#!/usr/bin/env python3
"""Build immutable next-stage plans only from verified completed scientific gates.

This process never provisions resources. Exit20 is a bounded scientific stop;
other errors indicate missing/corrupt evidence and never authorize dispatch.
"""
import argparse,importlib.util,json,math,os,shutil,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha,digest,PREFIX
from gearshift.coding_parallel import safe_id,relative_path,task_shards,validate_plan,STAGE_GATES,file_in
from gearshift.coding_snapshot import verify_archive
from gearshift.coding_training import verify_history_roots,nominations
ROOT=Path(os.environ.get('GEARSHIFT_REPO_ROOT',Path(__file__).resolve().parents[1])).resolve()
STAGES=['memory','histories','initialization','training','development','confirmation_with_second_seed','seed_sensitivity','finished']


class ScientificStop(Exception):
    pass


def read(path):return json.loads(Path(path).read_text())
def local(rel):return ROOT/relative_path(rel)
def rel(path):return str(Path(path).resolve().relative_to(ROOT))

def immutable(path,obj):
    path=Path(path)
    if path.exists():
        if read(path)!=obj:raise ValueError('Immutable evidence differs: '+str(path))
    else:write(path,obj)


def copy_new(source,destination):
    source,destination=Path(source),Path(destination)
    if source.is_symlink() or not source.is_file():raise ValueError('Unsafe imported source')
    if destination.exists():
        if destination.is_symlink() or sha(source)!=sha(destination):raise ValueError('Refusing to replace existing evidence: '+str(destination))
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name(destination.name+'.import.tmp');shutil.copyfile(source,temporary)
        if sha(temporary)!=sha(source):raise ValueError('Import copy changed')
        os.replace(temporary,destination)


def import_run(run_id):
    """Verified compact+external artifacts retain their exact repository paths."""
    safe_id(run_id);control=ROOT/'evidence/coding_pilot_v1/control/parallel'/run_id
    plan=read(control/'plan.json');complete=read(control/'complete.json');identity=digest(plan)
    if not complete.get('passed') or complete.get('stage_identity')!=identity:raise ValueError('Process stage is incomplete or has a changed identity: '+run_id)
    expected={w['worker_id'] for w in plan['workers']}
    if len(complete['workers'])!=len(expected) or {w['worker_id'] for w in complete['workers']}!=expected or not all(w.get('passed') for w in complete['workers']):raise ValueError('Missing completed workers')
    roots=[];imports=[]
    for worker in plan['workers']:
        wid=worker['worker_id'];safe_id(wid);dest=ROOT/'evidence/coding_pilot_v1/parallel_backups'/run_id/wid
        receipt=read(control/wid/'backup_verified.json');archive=dest/'final.tar.gz'
        if not receipt.get('verified') or not receipt.get('worker_confirmed_absent') or receipt['stage_identity']!=identity or sha(archive)!=receipt['sha256']:raise ValueError('Final backup is not verified')
        worker_root=f'results/coding_pilot_v1/{run_id}/{wid}';evidence_root=f'evidence/coding_pilot_v1/parallel/{run_id}/{wid}'
        with tempfile.TemporaryDirectory(prefix='coding-import-') as tmp:
            stage=Path(tmp);manifest=verify_archive(archive,stage)
            allowed=[worker_root,evidence_root]
            if plan['stage']=='memory':allowed+=['results/coding_pilot_v1/memory_v2_both_gpu','results/coding_pilot_v1/memory_v2_source_cpu_diagnostic']
            count=0
            for name in sorted(manifest['files']):
                if any(name.startswith(prefix+'/') for prefix in allowed):copy_new(stage/name,local(name));count+=1
            for name in ['checkpoint_manifest.json','feature_manifest.json']:
                path=stage/worker_root/name
                if not path.exists():continue
                for member,expected_sha in read(path)['files'].items():
                    relative_path(member)
                    if not member.startswith(worker_root+'/') or Path(member).suffix not in ['.pt','.safetensors']:raise ValueError('Unexpected heavy artifact')
                    actual=receipt.get('mapper_checkpoints',{}).get(member)
                    source=dest/'mapper_checkpoints'/member
                    if actual is None or actual['sha256']!=expected_sha or sha(source)!=expected_sha:raise ValueError('External feature/checkpoint backup differs')
                    copy_new(source,local(member));count+=1
            status=read(stage/evidence_root/'worker_status.json')
            if status.get('state')!='complete' or status.get('stage_identity')!=identity or status.get('worker_id')!=wid:raise ValueError('Final scientific worker status is incomplete')
            imports.append({'worker_id':wid,'archive_sha256':sha(archive),'files_imported':count,'snapshot_epoch':manifest['epoch']})
        roots.append(worker_root)
    immutable(ROOT/'evidence/coding_pilot_v1/pipeline_imports'/(run_id+'.json'),{'run_id':run_id,'stage_identity':identity,'workers':imports})
    return plan,roots


def load_script(name):
    path=ROOT/'scripts'/name;spec=importlib.util.spec_from_file_location(name.replace('.','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def tree_files(root,suffixes=('.json',)):
    return [p for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in suffixes]


def verify_result_root(root):
    root=Path(root);identity=read(root/'identity.json');done=read(root/'complete.json');native=read(root/'native_gate.json')
    if done['identity_sha256']!=digest(identity) or native.get('passed') is not True or native['identity_sha256']!=digest(identity):raise ValueError('Result identity/native controls differ')
    return identity,done


def validate_evaluation_roots(roots,expected,stage,selection_sha=None):
    seen={};inputs=[]
    for name in roots:
        root=local(name);identity,done=verify_result_root(root)
        if done['stage']!=stage or selection_sha is not None and done.get('selection_sha256')!=selection_sha:raise ValueError('Evaluation stage/selection changed')
        for tid,want in done['tasks'].items():
            folder=root/'tasks'/tid.replace('/','__');row=read(folder/'complete.json')
            if tid in seen or sha(folder/'complete.json')!=want or row['identity_sha256']!=digest(identity):raise ValueError('Duplicate or changed evaluation task')
            for member,expected_sha in row['files'].items():
                if Path(member).name!=member or sha(folder/member)!=expected_sha:raise ValueError('Evaluation output changed')
            if stage=='confirmation' and set(row['pass'])!={'A','B','C','D'}:raise ValueError('Headline task lacks a required arm')
            if stage=='second_seed' and len(row['candidates'])!=1:raise ValueError('Second-seed task lacks its fixed candidate')
            seen[tid]=row
        inputs+=tree_files(root)
    if set(seen)!=set(expected) or len(seen)!=len(expected):raise ValueError('Evaluation cohort incomplete')
    return inputs,seen



def retained_storage_inputs(state):
    """Bind preserved storage to two immutable recovery copies at every stage."""
    refs=state.get('retained_storage',[]);inputs=set();seen=set()
    for ref in refs:
        if ref['volume_id'] in seen or not ref['name'].startswith(PREFIX):
            raise ValueError('Invalid retained-volume identity')
        seen.add(ref['volume_id'])
        receipt=file_in(ROOT,ref['recovery_receipt'])
        if sha(receipt)!=ref['sha256']:raise ValueError('Retained-volume recovery proof changed')
        proof=read(receipt);copies=proof.get('copies',[])
        if (proof.get('volume_id')!=ref['volume_id'] or proof.get('both_copies_verified') is not True
            or len(copies)!=2 or len(set(copies))!=2):
            raise ValueError('Retained volume needs two verified recovery copies')
        inputs.add(receipt)
        for name in copies:
            path=file_in(ROOT,name)
            if sha(path)!=proof['archive_sha256']:raise ValueError('Retained-volume backup changed')
            inputs.add(path)
    return json.loads(json.dumps(refs)),inputs


def build(stage,state,output):
    if stage not in STAGES:raise ValueError('Unsupported pipeline stage')
    os.chdir(ROOT)
    pipeline=safe_id(state['pipeline_id']);runs=dict(state.get('completed_runs',{}))
    if 'cap_recovery' not in runs and state.get('initial_recovery_run'):runs['cap_recovery']=state['initial_recovery_run']
    imported={name:import_run(run) for name,run in runs.items()}
    state['artifact_roots']={name:roots for name,(plan,roots) in imported.items()}
    if 'cap_recovery' not in imported:raise ValueError('Completed exact cap recovery is required')
    recovery_plan,recovery_roots=imported['cap_recovery']
    if len(recovery_roots)!=1:raise ValueError('Expected one baseline recovery worker')
    baseline_root=state.get('baseline_root',recovery_roots[0]);baseline=local(baseline_root);state['baseline_root']=baseline_root
    gate=read(baseline/'baseline_gate.json')
    if not gate.get('passed') or not gate.get('cap_gate'):raise ScientificStop('The complete amended baseline/cap gate failed; no training is permitted.')
    baseline_receipt=load_script('coding_memory_preflight.py').check_baselines(baseline)
    proof_root=ROOT/'evidence/coding_pilot_v1/pipeline'/pipeline/'proofs';proofs={};proof_inputs={}
    def proof(name,paths,**fields):
        inputs={rel(p):sha(p) for p in sorted(set(map(Path,paths)))}
        if not inputs:raise ValueError('Scientific proof has no inputs')
        path=proof_root/(name+'.json');immutable(path,{'passed':True,'gate':name,'inputs':inputs,**fields})
        proofs[name]={'path':rel(path),'sha256':sha(path)};proof_inputs[name]=inputs
    proof('baseline',tree_files(baseline),**baseline_receipt)
    memory_gate=None;histories=[];history_roots=[];initialization=None;initialization_sha=None;candidates=[];selection_path=None;selection=None;seed2=[];headline_roots=[]
    rank=STAGES.index(stage)
    if rank>=1:
        if 'memory' not in imported:raise ValueError('Completed memory stage required')
        roots=imported['memory'][1]
        if len(roots)!=1:raise ValueError('Expected one memory worker')
        result=read(local(roots[0])/'memory_gate.json')
        if not result.get('passed') or not result.get('training_ready'):raise ScientificStop('Actual full-context gradient memory verification failed; training stopped.')
        memory_gate=result['selected_probe_root']+'/memory_gate.json';state['memory_gate']=memory_gate;path=local(memory_gate);raw=read(path);identity=read(path.parent/'identity.json')
        if sha(path)!=result['probe_gate_sha256'] or not all(raw.get(k) for k in ['passed','training_ready','actual_training_runtime']):raise ValueError('Actual memory proof differs')
        if set(raw.get('objectives',{}))!={'ordinary_continuation','natural_handoff_boundary'} or any(r.get('gradient_predictions')!=32 for r in raw['objectives'].values()):raise ValueError('Both32-prediction training objectives must pass memory')
        if identity['baseline_identity_file_sha256']!=baseline_receipt['baseline_identity_file_sha256'] or identity['training_protocol_sha256']!=sha(ROOT/'configs/coding_pilot_v1/training_protocol_v1.json'):raise ValueError('Memory scope changed')
        for name in ['gearshift/core.py','gearshift/coding_inference.py','gearshift/coding_gradients.py','gearshift/coding_training.py']:
            if identity['implementation'][name]!=sha(ROOT/name):raise ValueError('Memory implementation changed')
        proof('memory',tree_files(path.parent)+tree_files(local(roots[0])),actual_training_runtime=True)
    if rank>=2:
        if 'histories' not in imported:raise ValueError('Completed history stage required')
        history_roots=imported['histories'][1];state['history_roots']=history_roots
        histories,features,history_receipts=verify_history_roots(ROOT,history_roots)
        if sum(o['split']=='training' for o in histories)!=128 or sum(o['split']=='validation' for o in histories)!=32:raise ValueError('History membership changed')
        proof('histories',[local(r)/'complete.json' for r in history_roots]+[local(r)/'identity.json' for r in history_roots],
            training_count=128,validation_count=32,history_receipts_sha256=digest(history_receipts))
    if rank>=3:
        if 'initialization' not in imported:raise ValueError('Completed initialization stage required')
        roots=imported['initialization'][1]
        if len(roots)!=1:raise ValueError('Only one shared initialization is allowed')
        initroot=local(roots[0]);identity,done=verify_result_root(initroot);initialization=roots[0]+'/mapper_initialization.pt';initialization_sha=sha(local(initialization));state['initialization']=initialization;state['initialization_sha256']=initialization_sha
        if done['initialization_sha256']!=initialization_sha:raise ValueError('Initialization checkpoint differs')
        import torch
        checkpoint=torch.load(local(initialization),map_location='cpu',weights_only=True)
        if checkpoint.get('ridge')!=.01 or checkpoint.get('training_task_count')!=128 or checkpoint.get('fresh_source_specific') is not True or checkpoint.get('history_receipts_sha256')!=digest(history_receipts):raise ValueError('Initialization training corpus or ridge changed')
        del checkpoint
        proof('initialization',[initroot/'identity.json',initroot/'complete.json',local(initialization)],initialization_sha256=initialization_sha)
    def training_candidates(worker_roots,seed):
        available=[];observed=set();inputs=[]
        for name in worker_roots:
            root=local(name);identity,outer=verify_result_root(root)
            run=root/'run';run_identity=read(run/'identity.json');done=read(run/'complete.json')
            if done['identity_sha256']!=digest(run_identity) or outer['run_complete_sha256']!=sha(run/'complete.json'):raise ValueError('Training transaction changed')
            if done['seed']!=seed or run_identity['initialization_sha256']!=initialization_sha:raise ValueError('Training seed or shared initialization changed')
            curve=read(run/'validation_curve.json') if (run/'validation_curve.json').exists() else []
            selected=nominations(curve)
            if selected!=done['nominees']:raise ValueError('Validation nomination changed')
            if not selected:raise ScientificStop(f"{done['objective']} seed{seed} reached its bounded training stop without an eligible validated checkpoint at256updates.")
            if done['objective'] in observed:raise ValueError('Duplicate objective training run')
            observed.add(done['objective']);inputs+=tree_files(run)
            for index,nominee in enumerate(selected):
                checkpoint=run/nominee['checkpoint']['path']
                if sha(checkpoint)!=nominee['checkpoint']['sha256']:raise ValueError('Nominated checkpoint changed')
                inputs.append(checkpoint)
                available.append({'candidate_id':f"{done['objective']}-s{seed}-n{index+1}",'training_root':rel(run),
                    'training_complete_sha256':sha(run/'complete.json'),'checkpoint':rel(checkpoint),'checkpoint_sha256':sha(checkpoint)})
        if seed==20260915 and observed!={'ordinary_continuation','natural_handoff_boundary'}:raise ValueError('Both matched objectives must finish')
        return available,inputs
    if rank>=4:
        if 'training' not in imported:raise ValueError('Completed first-seed training required')
        candidates,training_inputs=training_candidates(imported['training'][1],20260915)
        proof('training',training_inputs,candidate_count=len(candidates),nomination_rule='Shared boundary validation KL; no confirmation outcomes')
    if rank>=5:
        if 'development' not in imported:raise ValueError('Completed development selection required')
        evaluation=load_script('coding_evaluation_worker.py');selection_path=rel(proof_root.parent/'selected_recipe.json')
        development_ids=[r['task_id'] for r in read(ROOT/'data/coding_pilot_v1/visible/development.json')]
        if not local(selection_path).exists():evaluation.select_development([local(r) for r in imported['development'][1]],local(selection_path),development_ids)
        # Rebuild independently to verify any previously frozen selection, ignoring timestamp alone.
        with tempfile.TemporaryDirectory() as tmp:
            actual=evaluation.select_development([local(r) for r in imported['development'][1]],Path(tmp)/'selection.json',development_ids)
        selection=read(local(selection_path));state['selection_path']=selection_path;state['selection_sha256']=sha(local(selection_path))
        if {k:v for k,v in selection.items() if k!='frozen_epoch'}!={k:v for k,v in actual.items() if k!='frozen_epoch'}:raise ValueError('Frozen development selection differs')
        # Portable paths: selection must use repo-relative checkpoint descriptors.
        if Path(selection['checkpoint']).is_absolute():raise ValueError('Selection paths must remain portable')
        proof('selection',[local(selection_path)]+[local(r)/'complete.json' for r in imported['development'][1]],
            selected_objective=selection['objective'],selection_path=selection_path,selection_sha256=sha(local(selection_path)))
    if rank>=6:
        if 'confirmation_with_second_seed' not in imported:raise ValueError('Completed headline/second-training stage required')
        mixed_plan,mixed_roots=imported['confirmation_with_second_seed'];byid={Path(r).name:r for r in mixed_roots}
        headline_roots=[byid[w['worker_id']] for w in mixed_plan['workers'] if w['role']=='confirmation'];state['headline_roots']=headline_roots
        second_roots=[byid[w['worker_id']] for w in mixed_plan['workers'] if w['role']=='second_seed_training']
        confirmation_ids=[r['task_id'] for r in read(ROOT/'data/coding_pilot_v1/visible/confirmation.json')]
        _,headline=validate_evaluation_roots(headline_roots,confirmation_ids,'confirmation',sha(local(selection_path)))
        available,second_inputs=training_candidates(second_roots,20260916)
        # The first common-validation nominee is fixed before any seed2 outcomes.
        seed2=[available[0]]
        second_identity=read(local(seed2[0]['training_root'])/'identity.json')
        if second_identity['objective']!=selection['objective']:raise ValueError('Second seed changed selected recipe')
        proof('second_seed',second_inputs+[local(r)/'complete.json' for r in headline_roots],selected_objective=selection['objective'],seed=20260916,headline_tasks=200)
    if stage=='finished':
        if 'seed_sensitivity' not in imported:raise ValueError('Completed seed-sensitivity stage required')
        cfg=read(ROOT/'configs/coding_pilot_v1/pilot.json');ids=[r['platform']+'/'+r['question_id'] for r in cfg['mapper']['second_seed_confirmation_ids']]
        inputs,rows=validate_evaluation_roots(imported['seed_sensitivity'][1],ids,'second_seed',sha(local(selection_path)))
        result={'passed':True,'gate':'completed_bounded_execution','headline_task_count':200,'second_seed_task_count':40,
            'completed_runs':runs,'selection_path':selection_path,'selection_sha256':sha(local(selection_path)),
            'inputs':{rel(p):sha(p) for p in inputs},'remaining':['Final statistical analysis and compact review bundle']}
        immutable(output,result);return result
    approval_path=state.get('approval_path','evidence/coding_pilot_v1/control/parallel_500h_approved.json')
    approval_sha=sha(local(approval_path))
    if state.get('approval_sha256') not in [None,approval_sha]:raise ValueError('Pipeline approval changed')
    regions=state['region_candidates']
    if not regions:raise ValueError('No explicit regions')
    retained,retained_files=retained_storage_inputs(state)
    base_files=set(retained_files)
    for pattern in ['gearshift/*.py','scripts/coding_*.py','configs/coding_pilot_v1/**/*.json']:
        base_files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for path in [ROOT/'pyproject.toml',ROOT/'requirements.lock.txt',ROOT/'data/coding_pilot_v1/identity.json',local(approval_path),
        ROOT/'evidence/coding_pilot_v1/parallel_500h_approved.json',ROOT/'evidence/coding_pilot_v1/experiment_100h_approved.json']:
        if path.exists():base_files.add(path)
    base_files.update(tree_files(baseline))
    for split in ['training','validation','development']:base_files.add(ROOT/f'data/coding_pilot_v1/visible/{split}.json')
    required=STAGE_GATES[stage]
    for name in required:
        base_files.add(local(proofs[name]['path']));base_files.update(local(p) for p in proof_inputs[name])
    fields={'baseline_root':baseline_root,'approval_path':approval_path}
    if memory_gate:fields['memory_gate']=memory_gate
    history_files=set()
    if rank>=2:
        for name in history_roots:history_files.update(tree_files(local(name),('.json','.pt')))
    history_fields={'history_roots':history_roots,'initialization':initialization,'initialization_sha256':initialization_sha}
    cohort_files=[];cohort=[];workers=[]
    # Estimates are deliberately labeled forecasts, not measured training throughput.
    source_rows=[read(p.parent/'source_history.json') for p in baseline.glob('tasks/*/complete.json')]
    source_mean=sum(r['reasoning_seconds'] for r in source_rows)/40
    a_mean=sum(read(p.parent/'A.json')['answer_seconds'] for p in baseline.glob('tasks/*/complete.json'))/40
    d_mean=sum(read(p.parent/'D.json')['answer_seconds'] for p in baseline.glob('tasks/*/complete.json'))/40
    prefill_mean=sum(read(p.parent/'D.json').get('native_prefill_seconds',0) for p in baseline.glob('tasks/*/complete.json'))/40
    b_mean=sum(read(p.parent/'small_history.json')['reasoning_seconds']+read(p.parent/'B.json')['answer_seconds'] for p in baseline.glob('tasks/*/complete.json'))/40
    defaults={'memory':(1800,5400),'histories':(1200+20*(source_mean+a_mean),18000),
        'initialization':(1800,3600),'training':(10800,10800),
        'development':(1200+5*(source_mean+len(candidates)*(d_mean+30)),7200),
        'confirmation_with_second_seed':(1200+29*(source_mean+a_mean+2*d_mean+b_mean+prefill_mean+30),39600),
        'seed_sensitivity':(1200+5*(source_mean+d_mean+30),5400)}
    estimates=state.get('stage_limits_seconds',{}).get(stage,{})
    reference_estimate=float(estimates.get('estimated_seconds',defaults[stage][0]))
    actual_baseline_estimate=float(defaults[stage][0])
    estimate=max(reference_estimate,actual_baseline_estimate);maximum=float(estimates.get('maximum_seconds',defaults[stage][1]))
    if estimate>maximum:raise ScientificStop(f'The measured baseline-based {stage} forecast exceeds its configured worker ceiling; bounded dispatch stopped.')
    run_id=safe_id(state.get('run_ids',{}).get(stage,pipeline+'-'+stage))
    def add_worker(worker_id,script,ids,extra,role=None,inputs=None):
        i=len(workers);w={'worker_id':worker_id,'region':regions[i%len(regions)],'command':['scripts/'+script],
            'estimated_seconds':estimate,'maximum_seconds':maximum,'task_ids':ids,'worker_fields':{**fields,**extra},
            'region_candidates':list(dict.fromkeys(regions[i%len(regions):]+regions[:i%len(regions)]))[:3]}
        if role:w['role']=role
        if inputs is not None:w['input_files']=sorted(rel(p) for p in inputs)
        workers.append(w);return w
    if stage=='memory':add_worker('memory','coding_memory_worker.py',[],{})
    elif stage=='histories':
        cohort_files=['data/coding_pilot_v1/visible/training.json','data/coding_pilot_v1/visible/validation.json']
        cohort=[r['task_id'] for f in cohort_files for r in read(local(f))]
        for i,ids in enumerate(task_shards(cohort,8)):add_worker(f'history{i+1:02d}','coding_history_worker.py',ids,{})
    elif stage=='initialization':
        base_files.update(history_files);add_worker('ridge','coding_training_worker.py',[],{'history_roots':history_roots})
    elif stage=='training':
        base_files.update(history_files);base_files.add(local(initialization))
        for objective in ['ordinary_continuation','natural_handoff_boundary']:
            add_worker('ordinary' if objective=='ordinary_continuation' else 'boundary','coding_training_worker.py',[],{**history_fields,'objective':objective,'seed':20260915})
    elif stage=='development':
        cohort_files=['data/coding_pilot_v1/visible/development.json'];cohort=[r['task_id'] for r in read(local(cohort_files[0]))]
        base_files.add(ROOT/'data/coding_pilot_v1/private/development.json')
        for i,ids in enumerate(task_shards(cohort,8)):add_worker(f'dev{i+1:02d}','coding_evaluation_worker.py',ids,{'candidates':candidates})
    elif stage=='confirmation_with_second_seed':
        selected=next(c for c in candidates if c['checkpoint_sha256']==selection['checkpoint_sha256'])
        selection_fields={'selection_path':selection_path,'selection_sha256':sha(local(selection_path))}
        training_files=set(base_files)|history_files|{local(initialization)}
        # The seed2 trainer receives no confirmation prompt, private test or result.
        training_files={p for p in training_files if '/private/' not in rel(p)}
        confirmation_files=set(base_files)|{local(selected['checkpoint']),local(selected['training_root'])/'identity.json',local(selected['training_root'])/'complete.json'}
        cohort_files=['data/coding_pilot_v1/visible/confirmation.json'];cohort=[r['task_id'] for r in read(local(cohort_files[0]))]
        confirmation_files.update([local(cohort_files[0]),ROOT/'data/coding_pilot_v1/private/confirmation.json',ROOT/'data/coding_pilot_v1/private/development.json'])
        for i,ids in enumerate(task_shards(cohort,7)):
            add_worker(f'headline{i+1:02d}','coding_evaluation_worker.py',ids,{**selection_fields,'candidates':[selected]},'confirmation',confirmation_files)
        w=add_worker('replicate','coding_training_worker.py',[],{**history_fields,**selection_fields,'objective':selection['objective'],'seed':20260916},'second_seed_training',training_files)
        w['estimated_seconds']=10800;w['maximum_seconds']=10800
        base_files.update(training_files|confirmation_files)
    elif stage=='seed_sensitivity':
        cfg=read(ROOT/'configs/coding_pilot_v1/pilot.json');reserved={r['platform']+'/'+r['question_id'] for r in cfg['mapper']['second_seed_confirmation_ids']}
        rows=[r for r in read(ROOT/'data/coding_pilot_v1/visible/confirmation.json') if r['task_id'] in reserved]
        if len(rows)!=40:raise ValueError('Reserved seed subset changed')
        cohort_files=['data/coding_pilot_v1/visible/second_seed_confirmation.json'];immutable(local(cohort_files[0]),rows);cohort=[r['task_id'] for r in rows]
        base_files.update([local(cohort_files[0]),ROOT/'data/coding_pilot_v1/visible/confirmation.json',ROOT/'data/coding_pilot_v1/private/confirmation.json',ROOT/'data/coding_pilot_v1/private/development.json'])
        for name in headline_roots:base_files.update(tree_files(local(name)))
        for i,ids in enumerate(task_shards(cohort,8)):
            add_worker(f'seed{i+1:02d}','coding_evaluation_worker.py',ids,{'candidates':seed2,'headline_roots':headline_roots,
                'selection_path':selection_path,'selection_sha256':sha(local(selection_path))})
    files={rel(p):sha(p) for p in sorted(base_files)}
    plan={'schema':1,'run_id':run_id,'stage':stage,'approval_sha256':approval_sha,
        'image_digest':state.get('image_digest',recovery_plan['image_digest']),'gates':{k:proofs[k] for k in required},
        'files':files,'task_ids':cohort,'cohort_files':cohort_files,'workers':workers,
        'preserve_original_measurements':True,'scientific_scope_unchanged':True,
        'forecast':{'method':'Observed completed development source/A/B/D times with explicit30second mapper estimate and1200second setup allowance; training uses declared three-hour cap. Initialization/memory forecasts are unmeasured conservative engineering allowances.',
            'source_reasoning_mean_seconds':source_mean,'A_answer_mean_seconds':a_mean,'D_answer_mean_seconds':d_mean,'D_native_prefill_mean_seconds':prefill_mean,'B_total_mean_seconds':b_mean,
            'target_completion_epoch':state.get('target_completion_epoch'),'frozen_reference_estimate_seconds':reference_estimate,
            'completed_baseline_estimate_seconds':actual_baseline_estimate,'selected_estimate_seconds':estimate}}
    if retained:plan['retained_storage']=retained
    critical_estimate=max(w['estimated_seconds'] for w in workers)
    plan['forecast']['critical_path_estimate_seconds']=critical_estimate
    state.setdefault('frozen_stage_reference_estimates',{}).setdefault(stage,reference_estimate)
    state.setdefault('stage_limits_seconds',{}).setdefault(stage,{'maximum_seconds':maximum})['estimated_seconds']=critical_estimate
    validate_plan(plan,ROOT,approval_sha);immutable(output,plan);return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=STAGES);parser.add_argument('--state',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Plan destination already exists; no implicit rebuild/retry')
    state=read(args.state)
    try:result=build(args.stage,state,args.output)
    except ScientificStop as exc:
        immutable(args.output.with_suffix('.stop.json'),{'scientific_stop':True,'stage':args.stage,'reason':str(exc),'state_sha256':sha(args.state)})
        print(str(exc));raise SystemExit(20)
    latest=read(args.state)
    for key in ['artifact_roots','baseline_root','memory_gate','history_roots','initialization','initialization_sha256','selection_path','selection_sha256','headline_roots','frozen_stage_reference_estimates']:
        if key in state:latest[key]=state[key]
    if args.stage in state.get('stage_limits_seconds',{}):
        measured=state['stage_limits_seconds'][args.stage]
        latest.setdefault('stage_limits_seconds',{}).setdefault(args.stage,{'maximum_seconds':measured['maximum_seconds']})['estimated_seconds']=measured['estimated_seconds']
    write(args.state,latest)
    print(json.dumps({'stage':args.stage,'path':str(args.output),'sha256':sha(args.output),'workers':len(result.get('workers',[]))}))

if __name__=='__main__':main()
