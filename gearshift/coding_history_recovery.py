"""Exact recovery of interrupted history shards, preserving original measurements."""
import json,shutil
from pathlib import Path
from .coding_control import digest,sha,write
from .coding_parallel import file_in
from .coding_resume import verify_prefix
from .coding_training import safe_id,validate_history


def read(path):return json.loads(Path(path).read_text())


def parent_for(repo,context):
    repo=Path(repo);spec=context['spec']
    declaration=file_in(repo,spec['history_recovery_manifest'])
    if sha(declaration)!=spec['history_recovery_manifest_sha256']:raise ValueError('History recovery declaration changed')
    recovery=read(declaration)
    if recovery.get('all_final_records_recovered') is not True or recovery.get('new_random_draw_allowed') is not False:
        raise ValueError('Final interrupted history records must be recovered before restart')
    row=recovery['workers'][spec['worker_id']]
    if row['task_ids']!=spec['task_ids']:raise ValueError('Recovery shard membership/order changed')
    for name,expected in row['files'].items():
        if sha(file_in(repo,name))!=expected:raise ValueError('Recovery artifact changed: '+name)
    root=repo/row['parent_root']
    if row.get('no_generation_started'):
        if (root/'identity.json').exists() or list(root.glob('tasks/*')):raise ValueError('Empty-start proof contradicts saved outputs')
        return root,None
    identity=read(root/'identity.json');native=read(root/'native_gate.json')
    if not native.get('passed') or native.get('identity_sha256')!=digest(identity):raise ValueError('Parent native controls differ')
    for key in ['config_sha256','training_protocol_sha256','baseline','memory_gate_sha256','memory_identity_sha256','implementation','data_identity']:
        if identity[key]!=context['identity'][key]:raise ValueError('Recovery numerical/data scope changed: '+key)
    allowed={safe_id(t) for t in spec['task_ids']}
    if any(p.name not in allowed for p in (root/'tasks').iterdir()):raise ValueError('Unexpected interrupted history task')
    return root,identity


def constraints(folder,task_id,identity):
    prefixes={};full={}
    for partial,record,key,stream in [
        ('inflight_source_reasoning.json','source_history.json','reasoning_ids','source'),
        ('inflight_teacher_answer.json','teacher_answer.json','answer_ids','answer')]:
        path=folder/partial
        if path.exists():
            obj=read(path)
            if obj['task_id']!=task_id or obj['identity_sha256']!=digest(identity):raise ValueError('Interrupted prefix identity differs')
            prefixes[stream]=obj['tokens']
        path=folder/record
        if path.exists():
            obj=read(path)
            if obj['task_id']!=task_id:raise ValueError('Completed segment task differs')
            verify_prefix(obj[key],prefixes.get(stream,[]),final=True)
            prefixes[stream]=obj[key];full[stream]=obj
    return prefixes,full


def verify_replayed(actual,previous,stream):
    if stream=='source':
        keys=['task_id','prompt_ids','reasoning_ids','prefix_ids','bridge_ids','natural_boundary','reasoning_capped','early_eos','prefix_cache_length','seed','rng_initial','rng_after_reasoning']
    else:keys=['task_id','answer_ids','answer_seed','rng_initial','rng_final','answer_ended_eos','answer_capped','bridge_token_count']
    if any(actual.get(k)!=previous.get(k) for k in keys):raise ValueError('Completed history segment changed on exact replay')


def run_recovered(context,repo,worker_module,backends=None):
    import gearshift.coding_inference as inference
    repo=Path(repo);root=context['root'];spec=context['spec'];all_ids=list(spec['task_ids'])
    parent,identity=parent_for(repo,context);reused={};features={};saved={}
    for tid in all_ids:
        folder=parent/'tasks'/safe_id(tid)
        if not folder.exists():continue
        prefixes,full=constraints(folder,tid,identity);saved[tid]=(prefixes,full)
        receipt_path=folder/'complete.json'
        if not receipt_path.exists():continue
        receipt=read(receipt_path)
        if receipt['task_id']!=tid or receipt['identity_sha256']!=digest(identity):raise ValueError('Completed history transaction identity changed')
        for name,expected in {**receipt['files'],**receipt.get('feature_files',{})}.items():
            if Path(name).name!=name or sha(folder/name)!=expected:raise ValueError('Completed history artifact changed')
        validate_history({'task_id':tid,'split':receipt['split'],'source_history':full['source'],'teacher_answer':full['answer']})
        target=root/'tasks'/safe_id(tid);target.mkdir(parents=True,exist_ok=False)
        for name in receipt['files']:shutil.copyfile(folder/name,target/name)
        for name in receipt.get('feature_files',{}):
            shutil.copyfile(folder/name,target/name);features[str((target/name).relative_to(repo))]=sha(target/name)
        shutil.copyfile(receipt_path,target/'parent_complete.json')
        copied={**receipt,'identity_sha256':digest(context['identity']),
            'files':{**receipt['files'],'parent_complete.json':sha(target/'parent_complete.json')},
            'measurement_identity_sha256':receipt.get('measurement_identity_sha256',receipt['identity_sha256']),
            'reused_without_generation':True,'parent_receipt_sha256':sha(receipt_path),'parent_identity_sha256':digest(identity)}
        write(target/'complete.json',copied);reused[tid]=sha(target/'complete.json')
    remaining=[tid for tid in all_ids if tid not in reused];old_reason=inference.reason;old_answer=inference.answer;old_write=worker_module.write
    report={'parent_identity_sha256':digest(identity) if identity else None,'reused_task_ids':list(reused),'regenerated_task_ids':remaining,'segments':{}}
    def checked(kind,original):
        def call(*args,**kwargs):
            tid=args[2] if kind=='source' else args[3];prefixes,full=saved.get(tid,({},{}));callback=kwargs.get('callback')
            def progress(tokens):
                verify_prefix(tokens,prefixes.get(kind,[]))
                if callback:callback(tokens)
            kwargs['callback']=progress;result=original(*args,**kwargs);obj=result[0] if kind=='source' else result
            key='reasoning_ids' if kind=='source' else 'answer_ids';verify_prefix(obj[key],prefixes.get(kind,[]),final=True)
            original_obj=full.get(kind)
            if original_obj is not None:verify_replayed(obj,original_obj,kind)
            report['segments'].setdefault(tid,{})[kind]={'verified_prefix_tokens':len(prefixes.get(kind,[])),
                'original_complete_record_reused':original_obj is not None,'replay_seconds':obj.get('reasoning_seconds' if kind=='source' else 'answer_seconds')}
            write(root/'recovery.json',report)
            if original_obj is not None:return (original_obj,result[1]) if kind=='source' else original_obj
            return result
        return call
    def save(path,obj):
        path=Path(path)
        if path==root/'complete.json':obj={**obj,'tasks':{**reused,**obj['tasks']},'task_count':len(all_ids),'recovery_manifest_sha256':spec['history_recovery_manifest_sha256']}
        elif path==root/'feature_manifest.json':obj={**obj,'files':{**features,**obj['files']}}
        elif path==root/'progress.json' and 'completed_tasks' in obj:obj={**obj,'completed_tasks':len(reused)+obj['completed_tasks'],'total_tasks':len(all_ids)}
        if path.name in ['source_history.json','teacher_answer.json'] and path.parent.parent==root/'tasks':
            original=parent/'tasks'/path.parent.name/path.name
            if original.exists():
                if obj!=read(original):raise ValueError('Saved complete segment must remain unchanged')
                shutil.copyfile(original,path);return
        old_write(path,obj)
    publish=context['publish']
    def adjusted_publish(**values):
        if 'completed_tasks' in values:values['completed_tasks']+=len(reused)
        if 'total_tasks' in values:values['total_tasks']=len(all_ids)
        publish(**values)
    active={**context,'spec':{**spec,'task_ids':remaining},'publish':adjusted_publish}
    write(root/'recovery.json',report)
    if features:write(root/'feature_manifest.json',{'files':features})
    try:
        inference.reason=checked('source',old_reason);inference.answer=checked('answer',old_answer);worker_module.write=save
        worker_module.run(active,backends)
        done=read(root/'complete.json')
        if set(done['tasks'])!=set(all_ids) or done['task_count']!=len(all_ids):raise ValueError('Recovered shard remains incomplete')
        return done
    finally:
        inference.reason=old_reason;inference.answer=old_answer;worker_module.write=old_write
