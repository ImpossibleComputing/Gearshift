from scripts.phase2_evidence_unit_audit import alternatives


def test_unit_sensitivity_preserves_original_formula_and_exposes_conversion():
    task={'prompt':'Expected monthly volume is 28 thousand requests.\n[S2] Aster: monthly fixed fee 110 credits; rate 3 credits per unit.\n[S3] Birch: monthly fixed fee 230 credits; rate 4 credits per unit.',
        'hidden':{'decision':'Birch','monthly_total':342,'difference':148}}
    result=alternatives(task)
    assert result['per_thousand']=={'monthly_total':342,'difference':148}
    assert result['per_request']=={'monthly_total':112230,'difference':28120}
    assert task['hidden']['monthly_total']==342
    assert alternatives({'prompt':'A water authority needs 28 units monthly.'}) is None
