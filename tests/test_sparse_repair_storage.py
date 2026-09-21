"""Exact public storage-binding checks only; no private values/provider calls."""
import json
from pathlib import Path
import shutil
import pytest
from gearshift import sparse_repair_storage as storage

ROOT=Path(__file__).resolve().parents[1]


def copy_evidence(tmp_path):
    binding=storage.load_binding(ROOT)
    for rel in (storage.BINDING_PATH,storage.DECLARATION_PATH,storage.CLOSURE_PATH,
                binding['provider_volume_receipt']['path']):
        path=tmp_path/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,path)
    return binding


def fallback():
    lease={'experiment_id':'sparse_repair_01','gpu_count':0,'network_volume_id':storage.FALLBACK_VOLUME,
        'private_storage_binding':storage.binding_reference(),'storage_region':storage.FALLBACK_REGION,
        'cpu_profile':storage.FALLBACK_PROFILE,'allowed_result_root':storage.RESULT_ROOT}
    plan={'private_volume_id':storage.FALLBACK_VOLUME,'private_storage_binding':storage.binding_reference(),
        'declaration_path':storage.DECLARATION_PATH,
        'declaration_sha256':'2fd04f4e3a20f8c38f4193582ef704ffff3d3cb28aba8f212711fa8329e13c9c'}
    return lease,plan


def test_exact_fallback_requires_no_private_bytes(tmp_path):
    binding=copy_evidence(tmp_path);lease,plan=fallback()
    assert storage.validate_cpu_lease(tmp_path,lease,plan)==binding
    assert not (tmp_path/'private').exists()
    assert storage.validate_cpu_lease(tmp_path,lease)==binding


def test_default_remains_original_without_new_binding_evidence(tmp_path):
    lease={'experiment_id':'sparse_repair_01','gpu_count':0,'network_volume_id':storage.ORIGINAL_VOLUME}
    assert storage.validate_cpu_lease(tmp_path,lease,{}) is None
    with pytest.raises(ValueError):storage.validate_cpu_lease(tmp_path,lease,{'private_volume_id':storage.FALLBACK_VOLUME})
    with pytest.raises(ValueError):storage.validate_cpu_lease(tmp_path,{**lease,'private_storage_binding':storage.binding_reference()})


@pytest.mark.parametrize('change',[
    {'network_volume_id':'arbitrary'}, {'network_volume_id':'zeicr9elbn'}, {'gpu_count':1},
    {'private_storage_binding':None},{'private_storage_binding':{'path':storage.BINDING_PATH,'sha256':'0'*64}},
    {'storage_region':'EU-RO-1'},{'cpu_profile':'cpu5c32'},{'allowed_result_root':'/workspace/elsewhere'},
])
def test_lease_rejects_unapproved_scope_before_evidence_read(tmp_path,change):
    lease,plan=fallback()
    with pytest.raises(ValueError):storage.validate_cpu_lease(tmp_path,{**lease,**change},plan)


@pytest.mark.parametrize('change',[
    {'private_volume_id':None},{'private_volume_id':storage.ORIGINAL_VOLUME},
    {'private_storage_binding':None},{'declaration_sha256':'0'*64},
    {'declaration_path':'other.json'},
])
def test_plan_requires_explicit_exact_binding(tmp_path,change):
    copy_evidence(tmp_path);lease,plan=fallback()
    with pytest.raises(ValueError):storage.validate_cpu_lease(tmp_path,lease,{**plan,**change})


@pytest.mark.parametrize('which',['binding','provider','declaration','closure'])
def test_any_frozen_public_evidence_mutation_rejected(tmp_path,which):
    binding=copy_evidence(tmp_path)
    rel={'binding':storage.BINDING_PATH,'provider':binding['provider_volume_receipt']['path'],
         'declaration':storage.DECLARATION_PATH,'closure':storage.CLOSURE_PATH}[which]
    with (tmp_path/rel).open('ab') as out:out.write(b' ')
    with pytest.raises(ValueError,match='hash differs'):storage.load_binding(tmp_path)


def test_symlink_rejected_before_read_even_if_same_public_bytes(tmp_path):
    copy_evidence(tmp_path);path=tmp_path/storage.BINDING_PATH;path.unlink();path.symlink_to(ROOT/storage.BINDING_PATH)
    with pytest.raises(ValueError,match='symlink'):storage.load_binding(tmp_path)
