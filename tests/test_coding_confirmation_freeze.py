import pytest
from scripts.coding_confirmation_freeze import resolve_tasks, private_identity_from_gate


def manifest():
    return {'task_ids': [f'task/{i}' for i in range(200)],
            'original_reserved40_subset_ids': [f'task/{i}' for i in range(0, 200, 5)]}


def test_silence_or_default_selection_is_not_an_owner_decision():
    for value in ({}, {'decision': 'all_200'}, {'authority': 'direct_owner_reply', 'decision': 'all_200'}):
        with pytest.raises(ValueError, match='owner reply'):
            resolve_tasks(manifest(), value)


def test_all200_requires_explicit_owner_choice_and_preserves_original_order():
    reply = {'authority': 'direct_owner_reply', 'owner_reply_text': 'Use all 200', 'decision': 'all_200'}
    assert resolve_tasks(manifest(), reply) == manifest()['task_ids']


def test_protect40_retains_exact_remaining160_without_substitution():
    reply = {'authority': 'direct_owner_reply', 'owner_reply_text': 'Keep the40 untouched', 'decision': 'protect_40'}
    selected = resolve_tasks(manifest(), reply)
    assert len(selected) == 160 and not set(selected) & set(manifest()['original_reserved40_subset_ids'])
    assert selected == [x for x in manifest()['task_ids'] if x not in manifest()['original_reserved40_subset_ids']]


def test_membership_change_cannot_silently_fix_the_conflict():
    original = manifest(); original['original_reserved40_subset_ids'][-1] = 'new/task'
    with pytest.raises(ValueError, match='membership'):
        resolve_tasks(original, {'authority': 'direct_owner_reply', 'owner_reply_text': 'Use all200', 'decision': 'all_200'})


def gate():
    return {'passed': True, 'rows': [{'task_id': t, 'split': 'confirmation'} for t in manifest()['task_ids']],
            'identity': {'private_files': {'confirmation': 'e' * 64}}}


def test_selected_private_scope_preserves_original_file_and_excludes_reserved40():
    tids = manifest()['task_ids']; selected = tids[40:]
    receipt = private_identity_from_gate(gate(), tids, selected, 'test')
    assert receipt['task_ids'] == selected and receipt['task_count'] == 160
    assert receipt['private_file_task_ids'] == tids and receipt['private_file_task_count'] == 200
    assert receipt['private_tests_sha256'] == 'e' * 64
    assert not receipt['private_file_rewritten'] and not receipt['private_test_values_included']


@pytest.mark.parametrize('failure', ['membership', 'duplicate', 'unselected', 'not_passed', 'bad_hash'])
def test_invalid_historical_private_identity_or_new_task_cannot_be_frozen(failure):
    g = gate(); tids = manifest()['task_ids']; selected = tids[40:]
    if failure == 'membership': g['rows'][0]['task_id'] = 'new/task'
    elif failure == 'duplicate': g['rows'][0] = g['rows'][1]
    elif failure == 'unselected': selected.append('new/task')
    elif failure == 'not_passed': g['passed'] = False
    elif failure == 'bad_hash': g['identity']['private_files']['confirmation'] = 'bad'
    with pytest.raises(ValueError, match='Historical private-file identity'):
        private_identity_from_gate(g, tids, selected, 'test')
