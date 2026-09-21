"""Reporting must compare training arms on common cases without pooling seeds."""
from scripts.phase2_report import objective_pairs
from gearshift.phase2_reporting import paired_cluster


def test_matched_training_arm_contrasts_keep_seeds_separate():
    conditions=['C','M/frozen','M/ordinary/1','M/boundary/1','M/ordinary/2','M/boundary/2']
    pairs=objective_pairs(conditions)
    assert ('M/boundary/1','M/ordinary/1') in pairs
    assert ('M/boundary/2','M/ordinary/2') in pairs
    assert ('M/boundary/1','M/ordinary/2') not in pairs
    rows=[dict(task_id=t,cluster_id=t,condition=c,score={'correct':v}) for t,c,v in [
        ('shared','M/boundary/1',1),('shared','M/ordinary/1',0),
        ('extra','M/ordinary/1',1),('shared','M/boundary/2',0)]]
    result=paired_cluster(rows,'M/boundary/1','M/ordinary/1','correct')
    assert result['n_clusters']==1
    assert result['n_task_variants']==1
    assert result['difference']==1
