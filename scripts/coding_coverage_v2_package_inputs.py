#!/usr/bin/env python3
"""Build an explicit immutable input tree and a separate private scoring tree.

This helper reads a frozen code worktree and the original data repository. It
never modifies either source, starts jobs, downloads models, or reads credentials.
The public deployment archive includes the selected mapper, which must later be
excluded from the compact review archive using excluded_heavy_inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

DECLARATION = 'configs/coding_pilot_v1/coverage_generalization_v2/declaration.json'
PRIVATE_PATH = 'data/coding_pilot_v1/private/coverage_generalization.json'
VISIBLE_PATH = 'data/coding_pilot_v1/visible/coverage_generalization.json'
PROVENANCE = 'evidence/coding_pilot_v1/coverage_v2_input_provenance'
PIN_SOURCE = 'evidence/coding_pilot_v1/coverage_generalization/runtime_receipts/coverage_recovery_20260918_02/evaluation_recovery'
CONTROL_GROUPS = {
    'full_path_numerical': ('results/coding_pilot_v1/post_progress01_numerical_20260917_03/numerical',
        ['complete.json', 'identity.json', 'native_gate.json', 'numerical_comparison.json']),
    'coverage_late_position': ('results/coding_pilot_v1/coverage_preflight_20260918_01/preflight',
        ['complete.json', 'identity.json', 'native_gate.json', 'late_numerical_controls.json',
         'preflight_result.json', 'preflight_updates.json', 'training_identity.json']),
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj, mode=0o644):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        os.chmod(path, mode); json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def scoped(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(p in ('..', '.') for p in path.parts):
        raise ValueError('Unsafe input relative path: ' + str(relative))
    result = Path(root)/Path(*path.parts)
    result.resolve().relative_to(root)
    for parent in [result, *result.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Input symlink is not permitted: ' + str(relative))
    if not result.is_file():
        raise ValueError('Missing required input: ' + str(relative))
    return result


def code_files(root):
    """Only tracked program/test files, plus explicit runtime documents/locks."""
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    chosen = [name for name in tracked if name and (
        (PurePosixPath(name).parts[0] in ('gearshift','scripts','tests') and name.endswith('.py')) or
        name in ('requirements.txt','requirements.lock.txt','pytest.ini','README.md','THIRD_PARTY_NOTICES.md') or
        (PurePosixPath(name).parent == PurePosixPath('.') and name.startswith('REPRODUCE_COVERAGE_GENERALIZATION_V2') and name.endswith('.md')))]
    forbidden = {'private', '.git', '.venv', '.pilot-venv', '__pycache__'}
    if any(forbidden.intersection(PurePosixPath(p).parts) for p in chosen):
        raise ValueError('Private/environment input found in tracked source selection')
    changed = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', *chosen], check=False)
    if changed.returncode:
        raise ValueError('Selected source files must be committed before staging')
    return sorted(chosen)


def _verify_controls(data_root):
    numerical_base = data_root/CONTROL_GROUPS['full_path_numerical'][0]
    complete = read(numerical_base/'complete.json')
    if complete.get('numerical_passed') is not True or complete['comparison_sha256'] != sha(numerical_base/'numerical_comparison.json'):
        raise ValueError('Established full-path numerical control is not passed/hash-bound')
    coverage_base = data_root/CONTROL_GROUPS['coverage_late_position'][0]
    result = read(coverage_base/'preflight_result.json')
    if not result.get('numerical_controls_passed') or not result.get('memory_and_gradient_checks_passed'):
        raise ValueError('Prior coverage numerical/gradient preflight did not pass')
    rows = read(coverage_base/'late_numerical_controls.json')
    if not rows or not all(r['passed'] for r in rows):
        raise ValueError('Prior late-position numerical controls did not all pass')
    for base, _ in CONTROL_GROUPS.values():
        if read(data_root/base/'native_gate.json').get('passed') is not True:
            raise ValueError('Prior native gate did not pass')


def _archive(folder, path, private=False):
    """Only regular files under the explicitly constructed tree enter archives."""
    with tarfile.open(path, 'w:gz', compresslevel=1) as archive:
        for item in sorted(folder.rglob('*')):
            if item.is_symlink():
                raise ValueError('Unexpected staging symlink')
            if item.is_file():
                info = archive.gettarinfo(str(item), arcname=str(item.relative_to(folder)))
                info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                info.mode = 0o600 if private else 0o644
                with item.open('rb') as stream:
                    archive.addfile(info, stream)
    if private:
        os.chmod(path, 0o600)
    return {'path': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size}


def package_inputs(code_root, data_root, output, declaration_path=DECLARATION, archives=False):
    code_root = Path(code_root).resolve(); data_root = Path(data_root).resolve(); output = Path(output).resolve()
    if output.is_relative_to(code_root) or output.is_relative_to(data_root):
        raise ValueError('Staging output must be outside both source repositories')
    if output.exists():
        raise ValueError('Use a new staging output directory; existing attempts remain untouched')
    declaration_source = scoped(code_root, declaration_path); d = read(declaration_source)
    if len(d['training_task_ids']) != 104 or len(d['validation_task_ids']) != 21 or len(set(d['training_task_ids']+d['validation_task_ids'])) != 125:
        raise ValueError('Expected exact disjoint104-training/21-validation membership')
    corpus_path = scoped(data_root, d['corpus_manifest'])
    if sha(corpus_path) != d['corpus_manifest_sha256']:
        raise ValueError('Frozen corpus manifest changed')
    corpus = read(corpus_path)
    for split, key in [('training','training_task_ids'), ('validation','validation_task_ids')]:
        if [r['task_id'] for r in corpus['histories'] if r['split'] == split] != d[key]:
            raise ValueError('Corpus membership/order differs from frozen declaration')
    _verify_controls(data_root)
    inputs = {}
    def add(relative, source_root, expected=None, destination=None, classification='public'):
        source = scoped(source_root, relative); dest = destination or relative
        if PurePosixPath(dest).is_absolute() or '..' in PurePosixPath(dest).parts or 'private' in PurePosixPath(dest).parts:
            raise ValueError('Unsafe/private public destination')
        h = sha(source); n = source.stat().st_size
        if expected and (h != expected['sha256'] or ('bytes' in expected and n != expected['bytes'])):
            raise ValueError('Required input bytes differ: ' + relative)
        if dest in inputs and inputs[dest]['sha256'] != h:
            raise ValueError('Conflicting public input bytes')
        inputs[dest] = {'source': source, 'sha256': h, 'bytes': n, 'classification': classification}
    for relative in code_files(code_root):
        add(relative, code_root, classification='source')
    if 'scripts/coding_coverage_v2_package_inputs.py' not in inputs:
        raise ValueError('Package helper must be committed in the frozen code worktree')
    add(declaration_path, code_root, classification='configuration')
    add('configs/coding_pilot_v1/pilot.json', code_root, {'sha256': corpus['config_sha256']}, classification='configuration')
    for name in ('schedules','panels','answer_seeds'):
        add(d[name+'_path'], code_root, {'sha256': d[name+'_sha256']}, classification='configuration')
    for relative, expected in d['unchanged_source_files'].items():
        add(relative, code_root, {'sha256': expected}, classification='source')
    add(d['corpus_manifest'], data_root, {'sha256': d['corpus_manifest_sha256']}, classification='corpus_manifest')
    for relative, expected in corpus['files'].items():
        if relative.endswith('.pt'):
            continue
        if not relative.endswith('.json') or not relative.startswith('results/coding_pilot_v1/history_recovery_20260916_01/'):
            raise ValueError('Unexpected file in compact frozen corpus')
        add(relative, data_root, expected, classification='source_trajectory')
    add(d['selected_checkpoint'], data_root, {'sha256': d['selected_checkpoint_sha256']}, classification='heavy_selected_mapper')
    prior_controls = {}
    for group, (base, names) in CONTROL_GROUPS.items():
        prior_controls[group] = {}
        for name in names:
            rel = str(PurePosixPath(base)/name); add(rel, data_root, classification='prior_numerical_control')
            prior_controls[group][name] = {'path': rel, 'sha256': inputs[rel]['sha256'], 'bytes': inputs[rel]['bytes']}
    pilot = read(scoped(code_root, 'configs/coding_pilot_v1/pilot.json')); models = pilot['models']; pins = {}
    for kind in ('source','receiver'):
        model_name = models[kind]['id'].split('/')[-1]
        for filename in ('config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json'):
            add('configs/coding_pilot_v1/reference/'+model_name+'/'+filename, code_root, classification='model_metadata')
        rel = PIN_SOURCE+'/'+kind+'_weight_pins.json'; old_pins = read(scoped(data_root, rel))
        if old_pins['revision'] != models[kind]['revision'] or not old_pins['sha256']:
            raise ValueError('Historical model-weight pins differ from frozen model revision')
        dest = PROVENANCE+'/'+kind+'_weight_pins.json'; add(rel, data_root, destination=dest, classification='historical_weight_pins')
        pins[kind] = {'model_id': models[kind]['id'], 'revision': models[kind]['revision'], 'path': dest,
                      'sha256': inputs[dest]['sha256'], 'bytes': inputs[dest]['bytes'], 'historical_source_path': rel}
    visible_source = scoped(data_root, VISIBLE_PATH); visible_all = read(visible_source)
    visible_lookup = {r['task_id']: r for r in visible_all}
    if len(visible_lookup) != len(visible_all) or not set(d['validation_task_ids']) <= set(visible_lookup):
        raise ValueError('Public validation prompts missing or duplicated')
    visible = [visible_lookup[t] for t in d['validation_task_ids']]
    private_source = scoped(data_root, PRIVATE_PATH); all_tests = read(private_source)
    if not set(d['validation_task_ids']) <= set(all_tests):
        raise ValueError('Private tests are missing declared validation tasks')
    private = {tid: all_tests[tid] for tid in d['validation_task_ids']}
    output.mkdir(parents=True); public = output/'repo'; private_root = output/'private'
    public.mkdir(); private_root.mkdir(mode=0o700)
    manifest = {}; classifications = {}
    for relative, item in sorted(inputs.items()):
        dest = public/relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(item['source'], dest)
        os.chmod(dest, 0o644)
        if sha(dest) != item['sha256']:
            raise ValueError('Copy verification failed: ' + relative)
        manifest[relative] = {'sha256': item['sha256'], 'bytes': item['bytes']}; classifications[relative] = item['classification']
    write(public/VISIBLE_PATH, visible)
    manifest[VISIBLE_PATH] = {'sha256': sha(public/VISIBLE_PATH), 'bytes': (public/VISIBLE_PATH).stat().st_size}
    classifications[VISIBLE_PATH] = 'validation_prompts_only'
    write(private_root/PRIVATE_PATH, private, mode=0o600)
    private_hash = sha(private_root/PRIVATE_PATH)
    code_commit = subprocess.check_output(['git','-C',str(code_root),'rev-parse','HEAD'], text=True).strip()
    receipt = {'schema_version': 1, 'code_commit': code_commit, 'declaration_path': declaration_path,
        'declaration_sha256': sha(public/declaration_path), 'input_manifest': dict(sorted(manifest.items())),
        'input_classifications': classifications, 'prior_controls': prior_controls, 'model_pins': pins,
        'private_tests_sha256': private_hash,
        'private_tests': {'staging_path': 'private/'+PRIVATE_PATH, 'path': PRIVATE_PATH, 'sha256': private_hash,
                          'bytes': (private_root/PRIVATE_PATH).stat().st_size, 'task_count': 21},
        'visible_filter': {'path': VISIBLE_PATH, 'original_sha256': sha(visible_source), 'task_ids': d['validation_task_ids'],
                           'filter_only_no_prompt_changes': True},
        'public_file_count': len(manifest), 'public_total_bytes': sum(r['bytes'] for r in manifest.values()),
        'excluded_heavy_inputs': {d['selected_checkpoint']: manifest[d['selected_checkpoint']]},
        'policy': {'source_repositories_modified': False, 'model_weights_included': False,
                   'paired_cache_tensors_included': False, 'private_tests_in_public_manifest': False,
                   'confirmation_task_text_or_tests_included': False, 'global_controller_or_watchdog_state_included': False,
                   'frozen_pilot_confirmation_ids': 'Historical metadata only; exact pilot config bytes retained for corpus identity.',
                   'review_exclusions': ['private/', PRIVATE_PATH, d['selected_checkpoint']]}}
    if archives:
        receipt['archives'] = {'inputs': _archive(public, output/'inputs.tar.gz'),
                               'private_tests': _archive(private_root, output/'private_tests.tar.gz', private=True)}
    write(output/'staging_receipt.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(); p.add_argument('--code-root', required=True); p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True); p.add_argument('--declaration', default=DECLARATION); p.add_argument('--archives', action='store_true')
    a = p.parse_args(); r = package_inputs(a.code_root, a.data_root, a.output, a.declaration, a.archives)
    print(json.dumps({'receipt': str(Path(a.output)/'staging_receipt.json'), 'code_commit': r['code_commit'],
                      'public_file_count': r['public_file_count'], 'public_total_bytes': r['public_total_bytes'],
                      'private_task_count': r['private_tests']['task_count'], 'private_tests_sha256': r['private_tests_sha256'],
                      'archives': r.get('archives')}, indent=2))


if __name__ == '__main__':
    main()
