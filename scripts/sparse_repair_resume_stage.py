#!/usr/bin/env python3
"""Stage a frozen public CA resume copy ONLY after root's bound absence proof.

No provider, SSH, model, sampler, grading, subprocess, or source-file writes.
CLI paths are intentionally fixed. Staging never authorizes launch. Exit 2 means
copied safely but complete-sampler/outer-incomplete trace review is required.
"""
import argparse, contextlib, fcntl, hashlib, json, math, os, re, stat, sys, tarfile, time
from pathlib import Path
sys.dont_write_bytecode=True
SOURCE=Path('/workspace/GearshiftSparseRepair')
DEST=Path('/workspace/GearshiftSparseRepairResume01')
RESULT=Path('results/sparse_repair_01')
OLD_NAME='screen_camtl05'; OLD_POD='no83hxamjucm7n'; VOLUME='u4rtme34ja'
OLD_LEASE=RESULT/'resources'/OLD_NAME/'lease.json'
OLD_LEASE_SHA='64e72e7a4f5ae462a5f2dc3493ebf40df05536db0ba7fabe7969cce6a1f9c370'
OLD_PLAN=RESULT/'resources'/OLD_NAME/'screen_runtime_plan.json'
OLD_PLAN_SHA='58f2d8eaed1e1df0b429de2ac01f39068782efeab2d58d31b27a2c7fa5378b17'
OLD_JOB=RESULT/'resources'/OLD_NAME/'screen_job_plan.json'
OLD_JOB_SHA='c6b48163bef923c9c9832cc0ee1eab6356521edf78ccde75307062a0013324a1'
WRAPPER_SHA='10b9fffc26695f04c1974f9455421c3210e67903c0101c6a8881237b1d56f06e'
DECL='configs/coding_pilot_v1/sparse_repair_01/declaration.json'
DECL_SHA='2fd04f4e3a20f8c38f4193582ef704ffff3d3cb28aba8f212711fa8329e13c9c'
RUNTIME_SHA='1cbf87a33bbf56f8cafd0ffe9eddec6e07e5fb1293fd89ad1392389ab6b5092f'
IMPLEMENTATION_SHA='9659381ff6c381eccdfa9c7854e1eda06f26504a55c95984e01ce34f32e7efb9'
MANIFEST=RESULT/'planning/resume_stage_no83hxamjucm7n.json'
CONDITIONS=('D','H','P','N_5','M_5','R_5','N_10','M_10','R_10','N_25','M_25','R_25','R_random','R_recent')
REQUIRED={'answer.json','working_set.json','sampler/identity.json','sampler/answer_record.json',
          'sampler/resume.json','sampler/complete.json','sampler/completion_timing.json'}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def scoped(root,rel):
    root=Path(root);r=Path(rel)
    if r.is_absolute() or not r.parts or '..' in r.parts or 'private' in r.parts:raise ValueError('Unsafe/nonpublic path')
    p=root/r
    for q in (p,*p.parents):
        if q==root:break
        if q.is_symlink():raise ValueError('Linked public path forbidden')
    if root.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Linked/escaping root')
    return p

def tree(root,relative):
    folder=scoped(root,relative);out=[]
    if not folder.exists():return out
    for p in sorted(folder.rglob('*')):
        if p.is_symlink():raise ValueError('Symlink in source tree')
        if p.is_dir():continue
        if not p.is_file():raise ValueError('Nonregular source entry')
        if p.name.endswith(('.lock','.tmp')):continue
        rel=p.relative_to(root)
        if any(v.startswith('.') for v in rel.parts) or 'private' in rel.parts:raise ValueError('Hidden/private source entry')
        if p.suffix not in ('.json','.jsonl','.log','.md','.txt','.csv','.tsv') or p.stat().st_size>128*1024**2:
            raise ValueError('Noncompact public tree entry')
        out.append(str(rel))
    return out

def absence_proof(path,expected,old_lease):
    p=Path(path)
    if p.is_symlink() or sha(p)!=expected:raise ValueError('Root absence receipt hash differs')
    x=read(p);epoch=x.get('observed_absent_epoch')
    if (x.get('pod_id')!=OLD_POD or x.get('network_volume_preserved') is not True
        or type(epoch) not in (int,float) or not math.isfinite(epoch)
        or not old_lease['allocation_epoch']<=epoch<=time.time()+5):
        raise ValueError('Explicit observed old-pod absence receipt required')
    return {'path':str(p),'sha256':expected,'observed_absent_epoch':epoch,
            'provider_absence_attested_by_root_not_independently_queried_here':True}

def sampler(folder,outer,history):
    tree(folder,'sampler')  # Reject every linked nested input before any read.
    ip=folder/'sampler/identity.json';rp=folder/'sampler/resume.json'
    if not ip.exists():
        if rp.exists() or (folder/'sampler/answer_record.json').exists() or (folder/'sampler/complete.json').exists():raise ValueError('Orphan sampler transaction')
        return {'state':'not_checkpointed'}
    identity=read(ip)
    expected={'schema':1,'sampler':'answer','task_id':outer['task_id'],'stream':outer['stream'],
        'seed':outer['answer_seed'],'cap':4096,'model':{'model_id':'Qwen/Qwen3-8B',
        'revision':'b968826d9c46dd6066d109eabc6255188de91218','runtime_identity_sha256':RUNTIME_SHA,
        'declaration_sha256':DECL_SHA,'cache_identity_sha256':digest({**outer,'conditioning':digest(history)})},
        'conditioning':{'history_sha256':digest(history),'prefix_ids':history['prefix_ids'],'bridge_ids':history['bridge_ids']},
        'prefill_chunk':512,'continuation_forward_chunk':1,'sampling':{'temperature':.6,'top_p':.95,'top_k':20},
        'eos':[151643,151645],'closing_think':151668}
    if identity!=expected:raise ValueError('Sampler identity differs from frozen task/cache/runtime')
    if not rp.exists():
        if (folder/'sampler/complete.json').exists() or (folder/'sampler/answer_record.json').exists():raise ValueError('Sampler result without resume')
        return {'state':'not_checkpointed','identity_sha256':digest(identity)}
    r=read(rp);tokens=r.get('tokens');forward=r.get('forwarded_tokens')
    if r.get('resume_sha256')!=digest({k:v for k,v in r.items() if k!='resume_sha256'}) or r.get('identity_sha256')!=digest(identity):
        raise ValueError('Sampler checkpoint checksum/identity differs')
    if r.get('state') not in ('running','interrupted','complete') or not isinstance(tokens,list) or len(tokens)>4096 or any(type(v)is not int or v<0 for v in tokens):
        raise ValueError('Invalid committed sampler tokens/state')
    if any(v in (151643,151645) for v in tokens[:-1]) or type(forward)is not int or not 0<=forward<=len(tokens)-bool(tokens and tokens[-1] in (151643,151645)):
        raise ValueError('Invalid committed forward boundary')
    if not isinstance(r.get('logits_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',r['logits_sha256']):
        raise ValueError('Checkpoint lacks exact reconstruction-logit fingerprint')
    for key in ('rng_initial','rng_state'):
        if not isinstance(r.get(key),list) or not r[key] or any(type(v)is not int or not 0<=v<=255 for v in r[key]):raise ValueError('Invalid RNG state bytes')
    if r['state']=='complete':
        record=r.get('record')
        if (not isinstance(record,dict) or digest(record)!=r.get('record_sha256') or record.get('answer_ids')!=tokens or record.get('rng_final')!=r['rng_state']
            or record.get('task_id')!=outer['task_id'] or record.get('answer_seed')!=outer['answer_seed'] or record.get('sampler_identity_sha256')!=digest(identity)):
            raise ValueError('Completed sampler record differs from transaction')
        ap=folder/'sampler/answer_record.json';cp=folder/'sampler/complete.json'
        if ap.exists() and read(ap)!=record:raise ValueError('Materialized sampler record changed')
        if cp.exists():
            c=read(cp)
            if not ap.exists() or c!={'identity_sha256':digest(identity),'record_sha256':digest(record),
                'record_file_sha256':sha(ap),'record_file':'answer_record.json'}:raise ValueError('Sampler completion binding differs')
    elif (folder/'sampler/complete.json').exists() or (folder/'sampler/answer_record.json').exists():raise ValueError('Incomplete resume alongside materialized sampler')
    return {'state':r['state'],'identity_sha256':digest(identity),'resume_sha256':r['resume_sha256'],
            'committed_tokens':len(tokens),'forwarded_tokens':forward,'logits_sha256':r.get('logits_sha256')}

def task_state(source,task,d,implementation):
    tid=task['task_id'];rel=RESULT/'screen/tasks'/tid.replace('/','__');folder=scoped(source,rel)
    tree(source,str(rel))  # Reject symlinks before interpreting any nested JSON.
    h=read(scoped(source,task['history_path']));template=None;tp=folder/'prompt_only_template.json'
    if tp.exists():
        template=read(tp)
        if (template.get('task_id')!=tid or template.get('declaration_sha256')!=DECL_SHA or template.get('enable_thinking') is not False
            or '<think>\n\n</think>' not in template.get('rendered_template','')
            or template.get('prefix_ids',[])+template.get('bridge_ids',[])!=template.get('prompt_ids')):raise ValueError('P template mismatch')
    draws=[];completed={};corners=[];allowed=set()
    for condition in CONDITIONS:
        for seed in d['seeds'][tid]:
            dr=Path(condition)/('seed_'+str(seed['seed_index']));allowed.add(str(dr));dest=folder/dr
            if not dest.exists():draws.append({'draw':str(dr),'state':'not_started'});continue
            if template is None:raise ValueError('Draw exists without original P template')
            outer={'experiment_id':'sparse_repair_01','cohort':'development_screen','task_id':tid,
                'condition':condition,**seed,'declaration_sha256':DECL_SHA,
                'history_sha256':sha(tp) if condition=='P' else task['history_sha256'],
                'original_source_history_sha256':task['history_sha256'],'mapper_sha256':d['mapper']['sha256'],
                'implementation_sha256':implementation,'teacher_answer_prefix_supplied':False}
            if (dest/'draw_identity.json').exists() and read(dest/'draw_identity.json')!=outer:raise ValueError('Original draw identity mismatch')
            history={k:template[k] for k in ('task_id','prompt_ids','prefix_ids','bridge_ids')} if condition=='P' else h
            state=sampler(dest,outer,history);done=dest/'complete.json'
            if done.exists():
                receipt=read(done)
                if receipt.get('identity')!=outer or not REQUIRED<=set(receipt.get('files',{})):raise ValueError('Completed draw contract differs')
                for name,expected in receipt['files'].items():
                    if sha(scoped(dest,name))!=expected:raise ValueError('Completed draw descendant changed')
                answer=read(dest/'answer.json');record=read(dest/'sampler/answer_record.json')
                if (state['state']!='complete' or any(answer.get(k)!=v for k,v in outer.items())
                    or any(answer.get(k)!=record.get(k) for k in ('task_id','answer_seed','answer_ids','answer_text','rng_final'))
                    or read(dest/'working_set.json').get('semantic_trajectory_union_complete') is not True):raise ValueError('Completed answer/state differs')
                completed[str(dr/'complete.json')]=sha(done);state={**state,'outer_complete':True}
            elif state['state']=='complete':
                corners.append({'task_id':tid,'draw':str(dr),'resume_path':str((dest/'sampler/resume.json').relative_to(source)),
                    'reason':'Complete sampler, absent outer draw: preserve answer; existing trace rebuild lacks terminal-logit fingerprint comparison.',
                    'requires_explicit_zero_sampling_replay_fingerprint_review_before_launch':True,**state})
            draws.append({'draw':str(dr),**state})
    if folder.exists():
        actual={str(p.relative_to(folder)) for p in folder.glob('*/seed_*') if p.is_dir()}
        if not actual<=allowed:raise ValueError('Unexpected draw population')
    complete=folder/'task_complete.json'
    if complete.exists():
        r=read(complete);files=r.get('files',[])
        if (r.get('task_id')!=tid or r.get('declaration_sha256')!=DECL_SHA or r.get('implementation_sha256')!=implementation
            or r.get('completed_records')!=42 or r.get('expected_records')!=42 or len(completed)!=42
            or r.get('backing_cache_fingerprints_unchanged') is not True or r.get('prompt_only_template_sha256')!=sha(tp)
            or len(files)!=42 or {x['path']:x['sha256'] for x in files}!=completed):raise ValueError('Completed task closure differs')
    return {'task_id':tid,'task_complete':complete.exists(),'completed_draws':len(completed),'draws':draws,
            'complete_draw_sha256':completed,'corner_cases':corners,'source_tree':str(rel)}

def archives(source,old_lease,jobs):
    control=scoped(source,RESULT/'allocations'/OLD_POD)
    mp=scoped(source,RESULT/'allocations'/OLD_POD/'release_manifest.json')
    rp=scoped(source,RESULT/'allocations'/OLD_POD/'gpu_release_verified.json')
    files=sorted((control/'release_backups').glob('*/compact_*.tar.gz'))
    result={'release_manifest_present':mp.exists(),'release_receipt_present':rp.exists(),'archives':[],
            'original_checkout_will_remain_untouched':True}
    if mp.exists():
        manifest=read(mp)
        if manifest.get('pod_id')!=OLD_POD or manifest.get('experiment_id')!='sparse_repair_01':raise ValueError('Old release manifest identity differs')
        for row in manifest['files']:
            p=scoped(source/RESULT,row['path'])
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Old release manifest file differs')
        result['release_manifest_sha256']=sha(mp)
    if rp.exists():
        if not jobs.verify_release_receipt(old_lease):raise ValueError('Original release proof invalid')
        result.update(release_receipt_sha256=sha(rp),full_original_release_receipt_verified=True)
    for p in files:
        scoped(source,p.relative_to(source))
        with tarfile.open(p,'r:gz') as tar:rows=json.load(tar.extractfile('COMPACT_MANIFEST.json'))
        for row in rows:scoped(source/RESULT,row['path'])
        jobs.verify_archive(p,rows)
        result['archives'].append({'path':str(p.relative_to(source)),'bytes':p.stat().st_size,'sha256':sha(p),
                                  'internal_manifest_verified':True,'files':len(rows)})
    if not files:result['limitation']='No completed compact archives found; source files are preserved but archive backup is not claimed.'
    return result

def copy_independent(src,dst,expected):
    if sha(src)!=expected:raise ValueError('Source changed before copy')
    if dst.exists():
        if not dst.is_file() or dst.is_symlink() or sha(dst)!=expected:raise ValueError('Divergent destination; never overwrite')
        if (src.stat().st_dev,src.stat().st_ino)==(dst.stat().st_dev,dst.stat().st_ino) or dst.stat().st_nlink!=1:raise ValueError('Destination is hardlinked')
        return 'identical_existing_untouched'
    dst.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation: even an unexpected concurrent creator is never overwritten.
    fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o644)
    try:
        with os.fdopen(fd,'wb') as out,src.open('rb') as incoming:
            for chunk in iter(lambda:incoming.read(8*1024**2),b''):out.write(chunk)
            out.flush();os.fsync(out.fileno())
        if sha(dst)!=expected or sha(src)!=expected or dst.stat().st_nlink!=1:raise ValueError('Copy/source checksum or link count differs')
    except BaseException:
        dst.unlink(missing_ok=True);raise
    return 'copied_independent_bytes'

def validate_existing_tasks(dest,pending,source_rows,source_hashes):
    relative=RESULT/'screen/tasks';folder=scoped(dest,relative)
    if not folder.exists():return
    existing=tree(dest,str(relative))  # Includes a symlink/type check before reads.
    if {p.name for p in folder.iterdir() if p.is_dir()}-{t.replace('/','__') for t in pending}:
        raise ValueError('Destination has undeclared scientific task scope')
    for rel in existing:
        if rel not in source_rows:raise ValueError('Destination has source-absent scientific state: '+rel)
        if sha(scoped(dest,rel))!=source_hashes[rel]:raise ValueError('Destination scientific state differs')

def planning_paths(dest):
    planning=scoped(dest,RESULT/'planning')
    lockpath=scoped(dest,RESULT/'planning/resume_stage.lock')
    manifest=scoped(dest,MANIFEST)
    if lockpath.exists() and not lockpath.is_file():raise ValueError('Nonregular staging lock')
    planning.mkdir(parents=True,exist_ok=True)
    return lockpath,manifest

def stage(absence_path,absence_sha,new_lease_rel,new_lease_sha):
    source,dest=SOURCE,DEST
    if source.is_symlink() or dest.is_symlink() or source==dest:raise ValueError('Separate real checkout roots required')
    if not source.is_dir() or not dest.is_dir():raise ValueError('Old and separately armed new checkout roots must exist')
    if sha(scoped(source,OLD_LEASE))!=OLD_LEASE_SHA:raise ValueError('Old immutable lease differs')
    old=read(source/OLD_LEASE);proof=absence_proof(absence_path,absence_sha,old)
    if (old['pod_id']!=OLD_POD or old['network_volume_id']!=VOLUME or old['allowed_result_root']!=str(source/RESULT)):
        raise ValueError('Old allocation/root identity differs')
    newpath=scoped(dest,new_lease_rel)
    if sha(newpath)!=new_lease_sha:raise ValueError('New immutable lease differs')
    new=read(newpath)
    actual=dict(v.split(b'=',1) for v in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in v).get(b'RUNPOD_POD_ID',b'').decode()
    if (sys.platform!='linux' or os.geteuid()!=0 or new['pod_id']==OLD_POD or actual!=new['pod_id'] or new['gpu_count']!=1
        or new['network_volume_id']!=VOLUME or new['experiment_id']!='sparse_repair_01'
        or new['allowed_result_root']!=str(dest/RESULT) or time.time()>=new['deadline_epoch']-120):raise ValueError('Wrong/expired sibling allocation')
    if sha(source/OLD_PLAN)!=OLD_PLAN_SHA or sha(source/OLD_JOB)!=OLD_JOB_SHA or sha(source/'scripts/sparse_repair_job.py')!=WRAPPER_SHA:
        raise ValueError('Frozen original operational/scientific source differs')
    plan=read(source/OLD_PLAN);oldjob=read(source/OLD_JOB)
    if plan['declaration_sha256']!=DECL_SHA or digest(plan['implementation_hashes'])!=IMPLEMENTATION_SHA:raise ValueError('Original implementation identity differs')
    for name,expected in plan['implementation_hashes'].items():
        if sha(scoped(source,name))!=expected:raise ValueError('Frozen source changed')
    # Imports are only existing stdlib-only gate/receipt helpers. Bytecode writes
    # are disabled before imports, so the original checkout is not modified.
    sys.path.insert(0,str(source))
    from scripts import sparse_repair_screen as screen
    from scripts import sparse_repair_job as jobs
    from gearshift.coding_confirmation_lease import load_lease,control_root,atomic_json
    load_lease(source/OLD_LEASE,OLD_LEASE_SHA);load_lease(newpath,new_lease_sha)
    control=control_root(new);jobs.validate_guard(new,control)
    if (control/'lease_guard/supervisor_registration.json').exists():raise ValueError('New destination already has scientific ownership; do not stage')
    for rel in (RESULT/'workers',RESULT/'claims'):
        folder=scoped(dest,rel)
        if folder.exists() and any(folder.iterdir()):raise ValueError('Destination already has worker/claim state; do not stage')
    args=oldjob['argv'];option=lambda name:args[args.index(name)+1]
    mapper=option('--mapper');visible=option('--visible');weights=option('--weights-receipt');cal=option('--calibration-root')
    if sha(scoped(source,DECL))!=DECL_SHA:raise ValueError('Declaration changed')
    preview=read(source/DECL)
    for t in preview['calibration_tasks']:tree(source,str(Path(cal)/t['task_id'].replace('/','__')))
    for t in preview['screen_tasks']:scoped(source,t['history_path'])
    scoped(source,visible);scoped(source,weights)
    d,_,_,gate=screen.validate_inputs(source,DECL,DECL_SHA,scoped(source,mapper),visible,scoped(source,cal),plan['task_ids'])
    if len(plan['task_ids'])!=len(set(plan['task_ids'])) or len(d['screen_tasks'])!=12:raise ValueError('Original assignment population differs')
    owned=[t for t in d['screen_tasks'] if t['task_id'] in plan['task_ids']]
    with contextlib.ExitStack() as stack:
        # Existing source lock inodes are only opened read-only; never created,
        # renamed, unlinked, or copied. Absence proof is still mandatory.
        lockpaths=[source/RESULT/'allocations'/OLD_POD/'sparse_job.lock']
        for rel in oldjob['output_folders']:
            lockpaths.append(source/RESULT/'claims'/('sparse_output_'+hashlib.sha256(str(source/RESULT/rel).encode()).hexdigest()+'.lock'))
        for task in owned:
            lockpaths.append(source/RESULT/'claims'/('sparse_screen_'+task['task_id'].replace('/','__')+'.lock'))
            lockpaths.extend((source/RESULT/'screen/tasks'/task['task_id'].replace('/','__')).glob('*/*/sampler/.sampler.lock'))
        for p in lockpaths:
            scoped(source,p.relative_to(source))
            if p.exists():
                f=stack.enter_context(p.open('rb'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lockpath,manifest_path=planning_paths(dest)
        fd=os.open(lockpath,os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
        lock=stack.enter_context(os.fdopen(fd,'a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if manifest_path.exists():raise ValueError('Staging already completed; verify its immutable manifest, do not rerun')
        states=[task_state(source,t,d,IMPLEMENTATION_SHA) for t in owned]
        pending=[s['task_id'] for s in states if not s['task_complete']]
        backup=archives(source,old,jobs)
        rows=set();source_verified=set()
        for folder in ('gearshift','scripts'):
            rows.update(str(p.relative_to(source)) for p in (source/folder).glob('*.py'))
        rows.update([DECL,d['inputs']['protocol_path'],mapper,visible,weights])
        if sha(scoped(source,d['inputs']['protocol_path']))!=d['inputs']['protocol_sha256']:raise ValueError('Protocol changed')
        if sha(scoped(source,weights))!=screen.verify_weights.__globals__['WEIGHTS_RECEIPT_SHA256']:raise ValueError('Weight receipt changed')
        for key in ('development_membership','history_identity_source'):
            row=d['inputs'][key];p=scoped(source,row['path'])
            if p.exists():
                if sha(p)!=row['sha256']:raise ValueError('Frozen ancillary config differs')
                rows.add(row['path'])
        rows.update(t['history_path'] for t in d['screen_tasks'])
        for t in d['calibration_tasks']:rows.update(tree(source,str(Path(cal)/t['task_id'].replace('/','__'))))
        for s in states:
            paths=tree(source,s['source_tree']);source_verified.update(paths)
            if s['task_id'] in pending:rows.update(paths)
        source_verified.update(rows);source_verified.update(map(str,(OLD_LEASE,OLD_PLAN,OLD_JOB)))
        hashes={rel:sha(scoped(source,rel)) for rel in sorted(source_verified)}
        validate_existing_tasks(dest,pending,rows,hashes)
        sizes={rel:scoped(source,rel).stat().st_size for rel in rows}
        if sum(sizes.values())>4*1024**3:raise ValueError('Bounded staging exceeds 4 GiB; no blind copy')
        # Check ALL preexisting destination overlaps before copying ANY bytes.
        for rel in rows:
            p=scoped(dest,rel)
            if p.exists() and (not p.is_file() or sha(p)!=hashes[rel] or p.stat().st_nlink!=1
                or (p.stat().st_dev,p.stat().st_ino)==(scoped(source,rel).stat().st_dev,scoped(source,rel).stat().st_ino)):
                raise ValueError('Divergent or linked destination; armed guard/source will not be overwritten: '+rel)
        if not pending:rows=set()  # Nothing to resume: do not copy a new execution tree.
        copied=[]
        for rel in sorted(rows):
            action=copy_independent(scoped(source,rel),scoped(dest,rel),hashes[rel])
            copied.append({'path':rel,'bytes':sizes[rel],'sha256':hashes[rel],'action':action})
        if any(sha(scoped(source,r))!=h for r,h in hashes.items()):raise ValueError('Stopped source changed during staging')
        corners=[c for s in states for c in s['corner_cases']]
        receipt={'schema':1,'purpose':'frozen_public_infrastructure_resume_stage','old_pod_id':OLD_POD,
            'source_root':str(source),'destination_root':str(dest),'old_lease_sha256':OLD_LEASE_SHA,
            'new_lease_path':str(new_lease_rel),'new_lease_sha256':new_lease_sha,'new_pod_id':new['pod_id'],
            'old_absence_proof':proof,'source_backup_validation':backup,'declaration_sha256':DECL_SHA,
            'implementation_sha256':IMPLEMENTATION_SHA,'implementation_hashes':plan['implementation_hashes'],
            'code_commit':plan['code_commit'],'runtime_identity_sha256':RUNTIME_SHA,
            'pending_task_ids':pending,'task_states':states,'corner_cases_requiring_review':corners,
            'status':'NO_RESUME_NEEDED' if not pending else 'STAGED_REVIEW_REQUIRED' if corners else 'STAGED_NO_LAUNCH',
            'launch_authorized_by_this_receipt':False,'model_execution_performed':False,'private_values_accessed':False,
            'source_files_written':False,'old_control_or_locks_copied':False,'hardlinks_created':False,
            'files':copied,'source_checksums':hashes,'calibration_gate':gate,
            'model_weights_not_copied_or_executed':'Existing absolute /workspace/hf paths; existing screen runner must rehash all22 shards.',
            'created_epoch':time.time()}
        atomic_json(scoped(dest,MANIFEST),receipt)
        return {'manifest_path':str(dest/MANIFEST),'manifest_sha256':sha(dest/MANIFEST),'pending_task_ids':pending,
                'complete_sampler_outer_incomplete':len(corners),'files':len(copied),'status':receipt['status']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirm-old-pod-absent',action='store_true',required=True)
    p.add_argument('--absence-receipt',required=True);p.add_argument('--absence-sha256',required=True)
    p.add_argument('--new-lease-relative',required=True);p.add_argument('--new-lease-sha256',required=True)
    a=p.parse_args()
    result=stage(a.absence_receipt,a.absence_sha256,a.new_lease_relative,a.new_lease_sha256)
    print(json.dumps(result));sys.exit(2 if result['complete_sampler_outer_incomplete'] else 0)
