#!/usr/bin/env python3
"""Build the compact local review bundle; never includes weights or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write,artifact

ROOT=Path(__file__).resolve().parents[1]


def private_operation_log(rel):
    """Controller conversations are not experimental source-model trajectories."""
    return (rel.parts[:3]==('evidence','phase2','studio_transfer') and
        (rel.name.startswith('controller_turn_') and rel.suffix in {'.jsonl','.txt'}
         or rel.name=='controller_last_message.txt'))


def exclusions():
    files={}
    tensors={path for dirname in ['results','evidence'] for path in (ROOT/dirname).rglob('*.pt')}
    for path in sorted(tensors):
        files[str(path.relative_to(ROOT))]={**artifact(path),
            'regeneration':'REPRODUCE_PHASE2.md; earlier artifacts also use REPRODUCE_FOLLOWUP.md and preserved pilot source. Regeneration records a new identity and is not promised byte-identical.'}
    config_names=['phase2_v1.json','phase2_final.json','phase2_cuda.json','phase2_4b.json','phase2_4b_cuda.json']
    configs=[read(ROOT/'configs'/name) for name in config_names if (ROOT/'configs'/name).exists()]
    models={(cfg[r],cfg[r+'_revision']) for cfg in configs for r in ['source','target']}
    for model,revision in sorted(models):
        folder=Path.home()/'.cache/huggingface/hub'/('models--'+model.replace('/','--'))/'snapshots'/revision
        for path in sorted(folder.glob('*.safetensors')):
            files[f'external_model_cache/{model}/{revision}/{path.name}']={**artifact(path),
                'regeneration':f'Download the public {model} snapshot at revision {revision} using the pinned Hugging Face tooling. No paid inference service.'}
    write(ROOT/'results/phase2_v1/excluded_checkpoints.json',dict(files=files,
        other_exclusions='Paired KV/feature tensor caches, environments, compiled bytecode, Git internals, model caches, temporary files and previous ZIPs. Original large-cache hashes remain in evidence/pilot/manifest.json.',
        verification='Hashes above were computed from local checkpoint bytes when building the compact bundle. Their presence in this list is not verification that omitted files are available to a reviewer.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--skip-checkpoint-inventory',action='store_true');a=p.parse_args()
    if not a.skip_checkpoint_inventory:exclusions()
    forbidden={'.git','.venv','venv','env','__pycache__','.pytest_cache','.cache','.mypy_cache','node_modules'}
    suffixes={'.py','.json','.jsonl','.csv','.md','.txt','.png','.svg','.html','.yaml','.yml','.toml','.ini','.patch','.diff'}
    roots=['gearshift','scripts','configs','tests','results','plots','evidence','gearshift_review','gearshift_phase2','.github','third_party']
    candidates=[p for dirname in roots for p in (ROOT/dirname).rglob('*') if p.is_file()]
    candidates += [p for p in ROOT.iterdir() if p.is_file()]
    candidates += [ROOT/'data'/name/'reasoning_trajectories.jsonl' for name in ['qwen3_1.7b_to_0.6b','qwen3_4b_to_0.6b']]
    files={}
    destination=ROOT/'gearshift_phase2_review.zip';temporary=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(set(candidates)):
            rel=path.relative_to(ROOT)
            if path.is_symlink() or forbidden&set(rel.parts):continue
            if private_operation_log(rel):continue
            if path.name in ['phase2_review_bundle_manifest.json','phase2_review_bundle_receipt.json','phase2_review_bundle_verification.json','CONTROL_DONE.json']:continue
            if rel.parts[:4]==('evidence','phase2','studio_transfer','finalization_runtime'):continue
            public_dataset=rel.as_posix()=='results/phase2_v1/datasets/HumanEvalPlus-OriginFmt-v0.1.10.jsonl.gz'
            if path.suffix not in suffixes and path.name!='.gitignore' and not public_dataset:continue
            if path.stat().st_size>64*1024**2:raise ValueError(f'Unexpected heavy compact record: {rel}')
            if path.name.startswith('.env') or path.name in ['auth.json','credentials.json']:raise ValueError('Credential filename in bundle candidates')
            # Read each record once: live operational heartbeats cannot race its manifest hash.
            data=path.read_bytes()
            if path.suffix in suffixes- {'.png'}:
                content=data.decode(errors='replace')
                if re.search(r'(?<![\w-])sk-(?:proj-)?[A-Za-z0-9_-]{32,}',content) or re.search(r'(?m)^-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----\r?$',content):
                    raise ValueError(f'Possible credential material; inspect {rel} without printing secrets')
            files[str(rel)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            archive.writestr('gearshift/'+str(rel),data)
        manifest=dict(schema=1,files=files,archive_prefix='gearshift/',
            excluded_checkpoints='results/phase2_v1/excluded_checkpoints.json',
            exclusions='No weights, mapper tensors, paired cache tensors, virtual environments, model caches, credentials, Git internals or private controller conversations. Experimental source-model trajectories and scientific judge attempts are retained.',
            regeneration='Tables and plots use compact records only: REPRODUCE_PHASE2.md. Missing weights are explicitly unverified.',
            publication_status='Local review bundle only. No upload, push, publication or project license choice.')
        write(ROOT/'phase2_review_bundle_manifest.json',manifest)
        archive.writestr('gearshift/phase2_review_bundle_manifest.json',(ROOT/'phase2_review_bundle_manifest.json').read_bytes())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC check failed')
        if len(archive.namelist())!=len(files)+1:raise ValueError('ZIP inventory mismatch')
        for required in ['PHASE2_RESULTS.md','REPRODUCE_PHASE2.md','requirements.lock.txt','results/phase2_v1/tasks/human_reservation.json']:
            if 'gearshift/'+required not in archive.namelist():raise ValueError('Missing required review artifact: '+required)
    temporary.replace(destination)
    receipt=dict(path=str(destination),**artifact(destination),files=len(files)+1,uncompressed_bytes=sum(d['bytes'] for d in files.values()),manifest=artifact(ROOT/'phase2_review_bundle_manifest.json'))
    write(ROOT/'phase2_review_bundle_receipt.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
