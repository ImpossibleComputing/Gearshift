"""No provider calls, remote hosts, private tests, weights or real GPU execution."""
import copy
import io
import json
import os
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace

import pytest

from scripts import coding_confirmation_regional_generation as s
from scripts import coding_confirmation_dispatch as base
from gearshift import coding_confirmation_lease as guard


def lease(root='/workspace/GearshiftConfirmationPrimary/results/coding_pilot_v1/test'):
    return {'experiment_id':'test','pod_id':'newpod','allocation_epoch':100,'deadline_epoch':43300,
        'upper_hourly_usd':11.1,'gpu_count':2,'baseline_usd':100,'total_cap_usd':2500,
        'cleanup_reserve_usd':40,'network_volume_id':'public_volume','allowed_result_root':root,
        'control_relative':'allocations/newpod'}


def pod():
    return {'id':'newpod','name':'gearshift-confirmation-new','status':'RUNNING','gpu':{'id':'NVIDIA H200','count':2},
        'mounts':{'network':[{'volumeId':'public_volume','path':'/workspace'}]}}


def test_provider_receipt_contains_no_credentials_or_unneeded_fields():
    value=pod();value.update(env={'RUNPOD_API_KEY':'SECRET'},apiKey='SECRET')
    receipt=s.inspect_provider(value,lease(),'new',200)
    assert 'SECRET' not in json.dumps(receipt)
    assert receipt['network_volume_id']=='public_volume' and receipt['provider_gpu_count']==2


@pytest.mark.parametrize('kind',['pod','name','status','gpu','count','mount','extra_mount','root','expired','preallocation'])
def test_live_identity_and_mount_mismatch_rejected(kind):
    p=pod();l=lease();now=200
    if kind=='pod':p['id']='other'
    elif kind=='name':p['name']='foreign'
    elif kind=='status':p['status']='STOPPED'
    elif kind=='gpu':p['gpu']['id']='NVIDIA H100'
    elif kind=='count':p['gpu']['count']=1
    elif kind=='mount':p['mounts']['network'][0]['volumeId']='private_volume'
    elif kind=='extra_mount':p['mounts']['network'].append({'volumeId':'private_volume','path':'/private'})
    elif kind=='root':l['allowed_result_root']='/workspace/another/result'
    elif kind=='expired':now=43300
    else:now=99
    with pytest.raises(ValueError):s.inspect_provider(p,l,'new',now)


@pytest.mark.parametrize('name',['data/private/tasks.json','.runpod/config.toml','model/mapper.pt','models/a.safetensors','keys/.env','hidden_tests/test.json','reference_answers/code.py','../outside.json'])
def test_public_inputs_reject_private_heavy_or_escaped_names_before_read(tmp_path,name):
    with pytest.raises(ValueError):s.public_file(tmp_path,name)


def archive(items):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as a:
        for name,value in items.items():
            info=tarfile.TarInfo(name);info.size=len(value);a.addfile(info,io.BytesIO(value))
    stream.seek(0);return stream


def package(items,extra=None):
    rows={k:{'sha256':s.hashlib.sha256(v).hexdigest(),'bytes':len(v)} for k,v in items.items()}
    manifest=json.dumps({'files':rows}).encode();values={**items,'manifest.json':manifest,**(extra or {})}
    return archive(values),s.hashlib.sha256(manifest).hexdigest()


def test_exact_public_archive_is_idempotent_and_rejects_changed_remote_bytes(tmp_path):
    items={'scripts/a.py':b'print("public")','configs/private_test_identity.json':b'{"hash_only":true}'}
    stream,h=package(items);s.unpack(tmp_path,stream,'manifest.json',h)
    stream,h=package(items);s.unpack(tmp_path,stream,'manifest.json',h)
    (tmp_path/'scripts/a.py').write_text('changed')
    stream,h=package(items)
    with pytest.raises(ValueError,match='Remote immutable'):s.unpack(tmp_path,stream,'manifest.json',h)
    assert (tmp_path/'scripts/a.py').read_text()=='changed'


@pytest.mark.parametrize('bad',['private/tasks.json','mapper.pt','../escape.json','/absolute.json'])
def test_archive_validation_precedes_any_write(tmp_path,bad):
    stream,h=package({'scripts/good.py':b'public',bad:b'not public'})
    with pytest.raises(ValueError):s.unpack(tmp_path,stream,'manifest.json',h)
    assert list(tmp_path.iterdir())==[]


def test_archive_unlisted_member_and_linked_parent_rejected(tmp_path):
    stream,h=package({'good.json':b'{}'},{'evil.py':b'code'})
    with pytest.raises(ValueError,match='unlisted'):s.unpack(tmp_path,stream,'manifest.json',h)
    other=tmp_path/'elsewhere';other.mkdir();(tmp_path/'scripts').symlink_to(other,target_is_directory=True)
    stream,h=package({'scripts/good.py':b'code'})
    with pytest.raises(ValueError,match='Linked'):s.unpack(tmp_path,stream,'manifest.json',h)
    assert list(other.iterdir())==[]


def test_local_import_closure_omits_unrelated_bootstrap(tmp_path):
    (tmp_path/'scripts').mkdir();(tmp_path/'gearshift').mkdir()
    (tmp_path/'scripts/a.py').write_text('from gearshift import numerical\nimport legacy\n')
    (tmp_path/'scripts/legacy.py').write_text('import math\n')
    (tmp_path/'gearshift/numerical.py').write_text('from . import support\n')
    (tmp_path/'gearshift/support.py').write_text('pass\n')
    (tmp_path/'scripts/coding_confirmation_regional_bootstrap_launch.py').write_text('changed deployed old version\n')
    assert s.dependencies(tmp_path,{'scripts/a.py'})=={'scripts/a.py','scripts/legacy.py','gearshift/numerical.py','gearshift/support.py'}


def test_all_device_environment_removes_auth_and_single_gpu_mask(monkeypatch):
    monkeypatch.setenv('RUNPOD_API_KEY','SECRET');monkeypatch.setenv('HF_TOKEN','SECRET');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    env=s.all_device_environment(lease())
    assert 'CUDA_VISIBLE_DEVICES' not in env and 'RUNPOD_API_KEY' not in env and 'HF_TOKEN' not in env
    assert env['HF_HUB_OFFLINE']=='1' and env['RUNPOD_POD_ID']=='newpod'


def launch_fixture(tmp_path,monkeypatch):
    l=lease(str(tmp_path/'results'));control=guard.control_root(l);control.mkdir(parents=True)
    (control/'lease_guard').mkdir();monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'remote_preflight',lambda *a:({},l,{'visible_gpu_count':2}))
    identity={'pid':9876,'pgid':9876,'start_ticks':77,'boot_id':'boot'}
    monkeypatch.setattr(guard,'linux_process_identity',lambda *a:identity)
    monkeypatch.setattr(s,'verify_launch_process',lambda *a:True)
    calls=[]
    def start(command,**kw):
        assert not (control/'sharded_generation_supervisor').exists()
        assert (control/'sharded_launch_intent.json').exists()
        calls.append((command,kw));(control/'sharded_generation_supervisor').mkdir()
        s.bind_json(control/'lease_guard/supervisor_registration.json',{**identity,'experiment_id':'test','pod_id':'newpod'})
        return SimpleNamespace(pid=9876,poll=lambda:None)
    monkeypatch.setattr(s.subprocess,'Popen',start)
    return l,control,calls,identity


def test_explicit_launch_detaches_only_after_preflight_and_verifies_registration(tmp_path,monkeypatch):
    l,c,calls,identity=launch_fixture(tmp_path,monkeypatch)
    result=s.remote_launch('plan.json','a'*64)
    assert result['guard_registered'] and result['process_identity']==identity
    assert len(calls)==1 and calls[0][1]['start_new_session'] is True
    assert calls[0][0][-4:]==['--plan','plan.json','--plan-sha256','a'*64]
    again=s.remote_launch('plan.json','a'*64)
    assert again['already_running'] and len(calls)==1


@pytest.mark.parametrize('prior',['intent','owned_folder','registration'])
def test_ambiguous_or_previous_supervisor_blocks_duplicate_launch(tmp_path,monkeypatch,prior):
    l,c,calls,_=launch_fixture(tmp_path,monkeypatch)
    if prior=='intent':s.bind_json(c/'sharded_launch_intent.json',{'unknown':True})
    elif prior=='owned_folder':(c/'sharded_generation_supervisor').mkdir()
    else:s.bind_json(c/'lease_guard/supervisor_registration.json',{'pid':999})
    with pytest.raises(ValueError):s.remote_launch('plan.json','a'*64)
    assert not calls


def test_preflight_failure_never_launches(tmp_path,monkeypatch):
    l,c,calls,_=launch_fixture(tmp_path,monkeypatch)
    def fail(*a):raise ValueError('mount mismatch')
    monkeypatch.setattr(s,'remote_preflight',fail)
    with pytest.raises(ValueError):s.remote_launch('plan.json','a'*64)
    assert not calls and not (c/'sharded_launch_intent.json').exists()


def test_spawn_failure_leaves_intent_and_cannot_automatically_retry(tmp_path,monkeypatch):
    l,c,calls,_=launch_fixture(tmp_path,monkeypatch)
    def fail(*a,**k):raise OSError('transient launch failure')
    monkeypatch.setattr(s.subprocess,'Popen',fail)
    with pytest.raises(OSError):s.remote_launch('plan.json','a'*64)
    with pytest.raises(ValueError,match='ambiguous'):s.remote_launch('plan.json','a'*64)


@pytest.mark.parametrize('mutation',['pid_reused','different_plan','different_registration'])
def test_process_and_guard_registration_must_match_exact_launch(tmp_path,monkeypatch,mutation):
    l=lease(str(tmp_path/'results'));c=guard.control_root(l);(c/'lease_guard').mkdir(parents=True)
    identity={'pid':9876,'pgid':9876,'start_ticks':77,'boot_id':'boot'};current=dict(identity)
    proc=tmp_path/'proc';(proc/'9876').mkdir(parents=True)
    command=[s.SUPERVISOR,'--plan','plan.json','--plan-sha256','a'*64]
    if mutation=='pid_reused':current['start_ticks']=88
    elif mutation=='different_plan':command[2]='other.json'
    registration={**identity,'pod_id':'newpod','experiment_id':'test'}
    if mutation=='different_registration':registration['pid']=8765
    s.bind_json(c/'lease_guard/supervisor_registration.json',registration)
    (proc/'9876/cmdline').write_bytes(b'\0'.join(x.encode() for x in command))
    monkeypatch.setattr(guard,'linux_process_identity',lambda *a:current)
    with pytest.raises(ValueError):s.verify_launch_process(l,'plan.json','a'*64,identity,proc)


def test_bind_never_overwrites_or_follows_link(tmp_path):
    p=tmp_path/'proof.json';s.bind_json(p,{'a':1});s.bind_json(p,{'a':1})
    with pytest.raises(ValueError):s.bind_json(p,{'a':2})
    (tmp_path/'link.json').symlink_to(p)
    with pytest.raises(ValueError):s.bind_json(tmp_path/'link.json',{'a':1})

@pytest.mark.parametrize('gpu_names',[[],['NVIDIA H200'],['NVIDIA H200','NVIDIA H100']])
def test_actual_gpu_count_is_checked_unmasked_before_launch(tmp_path,monkeypatch,gpu_names):
    from scripts import coding_confirmation_generation_supervisor as supervisor
    from scripts import coding_confirmation_sharded_generate as shard
    l=lease(str(tmp_path/'results'));s.bind_json(tmp_path/'lease.json',l)
    control=guard.control_root(l);control.mkdir(parents=True)
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'REMOTE',str(tmp_path));monkeypatch.setattr(s.time,'time',lambda:200)
    monkeypatch.setattr(supervisor,'validate_guard',lambda *a:{'pid':123})
    dev=tmp_path.stat().st_dev
    monkeypatch.setattr(shard,'workspace_mount',lambda:{'mount_point':'/workspace','device':str(os.major(dev))+':'+str(os.minor(dev)),
        'filesystem_type':'nfs4','source':'durable-volume'})
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p: b'coding_confirmation_lease.py' if str(p)=='/proc/123/cmdline' else original(p))
    def run(command,**kw):
        assert 'CUDA_VISIBLE_DEVICES' not in kw['env'];return SimpleNamespace(stdout=json.dumps(gpu_names))
    monkeypatch.setattr(s.subprocess,'run',run)
    with pytest.raises(ValueError,match='H200 count'):s.remote_inspect('lease.json',s.sha(tmp_path/'lease.json'))


def test_preflight_checks_existing_mapper_bytes_and_initial_snapshot(tmp_path,monkeypatch):
    from scripts import coding_confirmation_sharded_generate as shard
    s.bind_file(tmp_path/'mapper.pt',b'synthetic mapper')
    c={'plan':{'lease_path':'lease.json','lease_sha256':'a'*64,'preferences':['source','small']},
       'declaration':{'primary_checkpoints':{'FIXED':{'mapper_path':'mapper.pt','mapper_bytes':16,'mapper_sha256':s.sha(tmp_path/'mapper.pt')}}}}
    l=lease();calls=[]
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(shard,'public_context',lambda *a:c)
    monkeypatch.setattr(shard,'validate_generation_store',lambda *a,**k:(l,{}))
    monkeypatch.setattr(s,'remote_inspect',lambda *a:{'visible_gpu_count':2})
    monkeypatch.setattr(shard,'verify_initial_dataset',lambda c:calls.append('exact_initial_dataset'))
    s.remote_preflight('plan.json','a'*64);assert calls==['exact_initial_dataset']
    (tmp_path/'mapper.pt').write_bytes(b'altered')
    with pytest.raises(ValueError,match='mapper bytes'):s.remote_preflight('plan.json','a'*64)
    assert calls==['exact_initial_dataset']


def test_launch_receipt_reconciliation_is_idempotent_but_identity_strict(tmp_path):
    p=tmp_path/'launch.json';v={'plan_path':'plan.json','plan_sha256':'a'*64,'pod_id':'pod','process_identity':{'pid':12}}
    s.save_launch(p,v);s.save_launch(p,{**v,'already_running':True})
    assert s.read(p)==v
    with pytest.raises(ValueError):s.save_launch(p,{**v,'pod_id':'other'})


def test_source_helpers_must_be_committed_before_actual_stage(monkeypatch):
    def git(command,**kw):
        if command[1]=='rev-parse':return 'c'*40+'\n'
        relative=command[2].split(':',1)[1]
        return b'different unreviewed helper' if relative==s.SELF else (s.ROOT/relative).read_bytes()
    monkeypatch.setattr(s.subprocess,'check_output',git)
    with pytest.raises(ValueError,match='Commit/review'):s.source_paths(s.ROOT,s.DECLARATION,s.ADAPTER,set())


def test_stage_defaults_to_no_launch_and_reuses_byte_bound_plan(tmp_path,monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('regional_shard_fixture',Path(__file__).with_name('test_coding_confirmation_sharded_generate.py'))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    c=fixture.fixture(tmp_path,monkeypatch)
    area=tmp_path/'resources/new';area.mkdir(parents=True)
    s.bind_file(area/'lease.json',(tmp_path/'lease.json').read_bytes())
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'REMOTE',str(tmp_path));monkeypatch.setattr(base,'EVIDENCE',tmp_path/'resources')
    monkeypatch.setattr(s,'DECLARATION','declaration.json');monkeypatch.setattr(s,'ADAPTER','source.json')
    monkeypatch.setattr(s,'SELF','helper.py');s.bind_file(tmp_path/'helper.py',b'# public operational helper')
    monkeypatch.setattr(s.time,'time',lambda:200)
    monkeypatch.setattr(s,'source_paths',lambda *a:(c['declaration'],c['operational_source'],{'helper.py',*a[-1]},'c'*40))
    calls=[];monkeypatch.setattr(base,'api',lambda path:pod())
    monkeypatch.setattr(s,'inspect_provider',lambda p,l,n,t:{'pod_id':l['pod_id'],'observed_epoch':t})
    def upload(p,paths,folder,label):
        calls.append(('upload',label,paths))
        assert all('private' not in Path(x).parts and not x.endswith('.pt') for x in paths)
    monkeypatch.setattr(s,'upload',upload)
    def remote(p,action,args):
        calls.append((action,args))
        assert action!='_launch'
        if action=='_inspect':return {'pod_id':'new_pod','visible_gpu_count':1,'local_mount':c['plan'] and s.read(tmp_path/'mount.json')['local_mount']}
        plan=s.read(tmp_path/args[1]);assert plan['execution_mode']=='generation' and plan['preferences']==['source']
        return {'visible_gpu_count':1,'preflight_passed':True}
    monkeypatch.setattr(s,'remote_call',remote)
    result=s.stage('new','partition.json','origin')
    assert result['generation_launched'] is False
    assert [x[0] for x in calls]==['upload','_inspect','upload','_preflight']
    plan=s.read(tmp_path/result['plan_path']);assert plan['lease_sha256']==s.sha(area/'lease.json')
    assert plan['partition_sha256']==s.sha(tmp_path/'partition.json')
    assert not (Path(c['top'])/'allocations/newpod/sharded_generation_supervisor').exists()
    calls.clear();again=s.stage('new','partition.json','origin')
    assert again==result and [x[0] for x in calls]==['_preflight']
