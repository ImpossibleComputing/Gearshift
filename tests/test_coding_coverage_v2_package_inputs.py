import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts import coding_coverage_v2_package_inputs as package


def fixture(tmp_path):
    code = tmp_path/'code'; data = tmp_path/'data'; code.mkdir(); data.mkdir()
    def put(base, relative, obj):
        p = base/relative; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj) if isinstance(obj, (dict,list)) else obj); return p
    source = 'gearshift/core.py'; put(code, source, 'x = 1\n')
    put(code, 'scripts/coding_coverage_v2_package_inputs.py', 'pass\n')
    put(code, 'requirements.lock.txt', 'numpy==2.2.6\n')
    put(code, '.env', 'SECRET=must-not-copy\n')
    put(code, 'data/coding_pilot_v1/private/confirmation.json', {'must': 'not copy'})
    train = [f'train/{i}' for i in range(104)]; val = [f'validation/{i}' for i in range(21)]
    pilot = {'models': {kind: {'id': 'Qwen/model_'+kind, 'revision': kind+'_revision'} for kind in ('source','receiver')}}
    pilot_path = put(code, 'configs/coding_pilot_v1/pilot.json', pilot)
    histories = [{'task_id': t, 'split': 'training'} for t in train]+[{'task_id': t,'split':'validation'} for t in val]
    history = 'results/coding_pilot_v1/history_recovery_20260916_01/worker/tasks/train__0/source_history.json'
    hp = put(data, history, {'history': 'saved source reasoning'})
    corpus = {'histories': histories, 'config_sha256': package.sha(pilot_path), 'files': {history: {'sha256': package.sha(hp), 'bytes': hp.stat().st_size},
              'never-copy.pt': {'sha256': 'ignored-cache-tensors'}}}
    corpus_rel = 'results/coding_pilot_v1/partial_corpus_20260917_v1/corpus_manifest.json'; cp = put(data, corpus_rel, corpus)
    selected = 'results/coding_pilot_v1/recovery_train_20260917_04/train/mapper_step_0096.pt'; sp = put(data, selected, 'synthetic mapper')
    d = {'training_task_ids': train, 'validation_task_ids': val, 'corpus_manifest': corpus_rel, 'corpus_manifest_sha256': package.sha(cp),
         'selected_checkpoint': selected, 'selected_checkpoint_sha256': package.sha(sp), 'unchanged_source_files': {source: package.sha(code/source)}}
    for key in ('schedules','panels','answer_seeds'):
        rel = 'configs/'+key+'.json'; path = put(code, rel, {})
        d[key+'_path'] = rel; d[key+'_sha256'] = package.sha(path)
    put(code, package.DECLARATION, d)
    for group, (base, names) in package.CONTROL_GROUPS.items():
        for name in names:
            obj = {'passed': True}
            if name == 'preflight_result.json': obj = {'numerical_controls_passed': True, 'memory_and_gradient_checks_passed': True}
            if name == 'late_numerical_controls.json': obj = [{'passed': True}]
            put(data, base+'/'+name, obj)
        if group == 'full_path_numerical':
            (data/base/'complete.json').write_text(json.dumps({'numerical_passed': True, 'comparison_sha256': package.sha(data/base/'numerical_comparison.json')}))
    for kind in ('source','receiver'):
        put(data, package.PIN_SOURCE+'/'+kind+'_weight_pins.json', {'revision': kind+'_revision', 'sha256': {'weights.safetensors': 'pinned'}})
        for name in ('config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json'):
            put(code, 'configs/coding_pilot_v1/reference/model_'+kind+'/'+name, {'metadata': name})
    put(data, package.VISIBLE_PATH, [{'task_id': t, 'prompt': 'prompt '+t} for t in val+train[:4]])
    put(data, package.PRIVATE_PATH, {t: {'tests': ['hidden '+t]} for t in val+train[:4]})
    subprocess.run(['git','init','-q',str(code)],check=True)
    subprocess.run(['git','-C',str(code),'add','.'],check=True)
    subprocess.run(['git','-C',str(code),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Fixture'],check=True)
    return code, data, d


def test_package_excludes_private_and_unneeded_data_from_public_archive(tmp_path):
    code, data, d = fixture(tmp_path); output = tmp_path/'staging'
    original = {str(p): package.sha(p) for root in (code,data) for p in root.rglob('*') if p.is_file() and '.git' not in p.parts}
    receipt = package.package_inputs(code, data, output, archives=True)
    manifest = receipt['input_manifest']
    assert package.PRIVATE_PATH not in manifest
    assert set(package.read(output/'private'/package.PRIVATE_PATH)) == set(d['validation_task_ids'])
    assert len(package.read(output/'repo'/package.VISIBLE_PATH)) == 21
    assert receipt['private_tests_sha256'] == package.sha(output/'private'/package.PRIVATE_PATH)
    assert (output/'private'/package.PRIVATE_PATH).stat().st_mode & 0o777 == 0o600
    assert not (output/'repo/.env').exists()
    assert not (output/'repo/data/coding_pilot_v1/private').exists()
    assert [p for p in manifest if p.endswith('.pt')] == [d['selected_checkpoint']]
    assert set(receipt['excluded_heavy_inputs']) == {d['selected_checkpoint']}
    assert not any('controller' in p or 'sandbox_gate' in p for p in manifest)
    with tarfile.open(output/'inputs.tar.gz') as archive:
        assert set(archive.getnames()) == set(manifest)
        assert all(m.isfile() for m in archive.getmembers())
    with tarfile.open(output/'private_tests.tar.gz') as archive:
        assert archive.getnames() == [package.PRIVATE_PATH]
    assert original == {str(p): package.sha(p) for root in (code,data) for p in root.rglob('*') if p.is_file() and '.git' not in p.parts}


def test_package_rejects_changed_frozen_mapper_or_mutating_original(tmp_path):
    code, data, d = fixture(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        package.package_inputs(code, data, data/'staging')
    (data/d['selected_checkpoint']).write_text('changed')
    with pytest.raises(ValueError, match='Required input bytes'):
        package.package_inputs(code, data, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_package_rejects_source_symlink_or_uncommitted_code(tmp_path):
    code, data, _ = fixture(tmp_path)
    (code/'gearshift/core.py').write_text('changed')
    with pytest.raises(ValueError, match='committed'):
        package.package_inputs(code, data, tmp_path/'output')
    link = data/'link.json'; link.symlink_to(code/'configs/coding_pilot_v1/pilot.json')
    with pytest.raises(ValueError):
        package.scoped(data, 'link.json')
