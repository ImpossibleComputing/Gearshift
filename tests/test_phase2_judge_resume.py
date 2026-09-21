import copy
import importlib.util
from pathlib import Path
import pytest
from gearshift.phase2_io import artifact

spec=importlib.util.spec_from_file_location('resume',Path(__file__).resolve().parents[1]/'scripts/phase2_judge_resume.py')
resume=importlib.util.module_from_spec(spec);spec.loader.exec_module(resume)

def test_resume_allows_revision_only_and_rejects_source_or_policy_drift(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('judge.py').write_text('same scientific code\n')
    Path('snapshot').mkdir();Path('snapshot/judge.py').write_text('same scientific code\n')
    old={'source':{'git_sha':'original','files':{'judge.py':artifact('judge.py')},'snapshot':'snapshot'},'rubric':{'sha256':'frozen'},'cli_version':'same'}
    new=copy.deepcopy(old);new['source']['git_sha']='later-reporting-commit'
    assert resume.validate_provenance_only_resume(old,new)=={'previous_git_sha':'original','current_git_sha':'later-reporting-commit'}
    changed=copy.deepcopy(new);changed['rubric']['sha256']='changed'
    with pytest.raises(ValueError):resume.validate_provenance_only_resume(old,changed)
    changed=copy.deepcopy(new);changed['cli_version']='different'
    with pytest.raises(ValueError):resume.validate_provenance_only_resume(old,changed)
    Path('judge.py').write_text('changed scientific code\n')
    with pytest.raises(ValueError):resume.validate_provenance_only_resume(old,new)
    assert old['source']['git_sha']=='original'
