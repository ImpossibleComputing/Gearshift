import json
from pathlib import Path
import tarfile
import pytest
from scripts import sparse_repair_job as job
from gearshift.coding_confirmation_lease import sha256_file,verify_release_receipt


def setup(tmp_path):
    root=tmp_path/'results/sparse_repair_01';root.mkdir(parents=True)
    lease={'experiment_id':'sparse_repair_01','pod_id':'pod01','allocation_epoch':100.,'deadline_epoch':3700.,
           'upper_hourly_usd':5.55,'gpu_count':1,'baseline_usd':1360.,'other_reserved_usd':20.,
           'total_cap_usd':2500,'total_cap_gpu_hours':None,'cleanup_reserve_usd':40,
           'allowed_result_root':str(root),'control_relative':'allocations/pod01'}
    p=tmp_path/'lease.json';p.write_text(json.dumps(lease))
    plan={'experiment_id':'sparse_repair_01','job_id':'calibration-short-01','lease_path':'lease.json',
          'lease_sha256':sha256_file(p),'output_folders':['calibration'],
          'argv':[job.PYTHON,'scripts/sparse_repair_calibrate.py','--output','results/sparse_repair_01/calibration']}
    return root,lease,plan


def test_plan_calibration_exactscope_and_noautomaticretry(tmp_path):
    root,lease,p=setup(tmp_path)
    actual,_,folders=job.validate_plan(p,tmp_path)
    assert actual==lease and folders==[root/'calibration']
    with pytest.raises(ValueError,match='Automatic retries'):
        job.validate_plan({**p,'automatic_retries':1},tmp_path)
    with pytest.raises(ValueError,match='output scope'):
        job.validate_plan({**p,'output_folders':['another-worker']},tmp_path)


def test_screen_scopes_bind_tasks_worker_and_childlease(tmp_path):
    root,lease,p=setup(tmp_path)
    child={k:p[k] for k in ['lease_path','lease_sha256']};(tmp_path/'runtime.json').write_text(json.dumps(child))
    p['argv']=[job.PYTHON,'scripts/sparse_repair_screen.py','--plan','runtime.json','--task-id','foo/bar','--worker-id','w01']
    p['output_folders']=['screen/tasks/foo__bar','workers/w01']
    job.validate_plan(p,tmp_path)
    p['output_folders'].append('screen/tasks/unrelated')
    with pytest.raises(ValueError,match='only its tasks'):
        job.validate_plan(p,tmp_path)


def test_scope_rejects_escape_private_and_symlink(tmp_path):
    for path in ['../out','private/tests.json','/tmp/out']:
        with pytest.raises(ValueError):job.scoped(tmp_path,path)
    (tmp_path/'link').symlink_to('/tmp')
    with pytest.raises(ValueError,match='Linked'):job.scoped(tmp_path,'link/output.json')


def test_backup_rejects_heavy_and_private_not_silently_drop(tmp_path):
    root,lease,p=setup(tmp_path);out=root/'calibration';out.mkdir()
    (out/'cache.pt').write_bytes(b'not allowed')
    with pytest.raises(ValueError,match='Noncompact'):job.public_files(root,[out])
    (out/'cache.pt').unlink();(out/'private').mkdir();(out/'private/secret.json').write_text('{}')
    with pytest.raises(ValueError,match='Private'):job.public_files(root,[out])


def test_two_verified_archives_release_only_own_output(tmp_path):
    root,lease,p=setup(tmp_path);out=root/'calibration';out.mkdir();(out/'complete.json').write_text('{"passed":true}')
    other=root/'unrelated';other.mkdir();(other/'do-not-touch.json').write_text('{}')
    wrapper=root/'allocations/pod01/sparse_job/one';wrapper.mkdir(parents=True);(wrapper/'exited.json').write_text('{"returncode":0}')
    manifest=job.compact_release(root,lease,[out],wrapper,'complete')
    assert verify_release_receipt(lease)
    record=json.loads(manifest.read_text());archives=[root/r['path'] for r in record['files'] if r['path'].endswith('.tar.gz')]
    assert len(archives)==2 and sha256_file(archives[0])==sha256_file(archives[1])
    with tarfile.open(archives[0]) as a:
        assert not any('unrelated' in p for p in a.getnames())
        assert 'calibration/complete.json' in a.getnames()
    (out/'complete.json').write_text('{"passed":false}')
    with pytest.raises(ValueError,match='has changed'):verify_release_receipt(lease)


def test_sampler_advisory_lock_skipped_before_hidden_rejection(tmp_path):
    root,lease,p=setup(tmp_path);out=root/'calibration';out.mkdir()
    (out/'.sampler.lock').write_text('')
    (out/'complete.json').write_text('{}')
    assert [x.name for x in job.public_files(root,[out])]==['complete.json']
