#!/usr/bin/env python3
"""Reserve every split before inference; grading material is never in TaskSpec.visible()."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gzip
import json
from dataclasses import asdict
import numpy as np
from datasets import load_dataset
from gearshift.phase2_io import read,immutable,artifact,digest,snapshot
from gearshift.phase2_tasks import TaskSpec,CONTRACTS
from gearshift.reasoning import numeric_answer


def evidence(split,index,variant=None):
    rng=np.random.default_rng({'train':100000,'validation':200000,'development':300000,'characterization':400000,'confirmation':500000,'memory':600000}[split]+index)
    # Split-specific scenarios and units; exact task/parent seeds never cross splits.
    context={'train':('inventory deliveries','crates','warehouse'), 'validation':('water distribution','liters','water authority'),
             'development':('museum event staffing','visitors','museum'), 'characterization':('archive migration','thousand requests','archive'),
             'confirmation':('microgrid maintenance','kilowatt hours','energy cooperative'), 'memory':('emergency radio procurement','terminals','dispatch center')}[split]
    a,b=f'Aster-{index:03}',f'Birch-{index:03}'
    units=int(rng.integers(12,50));fixed_a=int(rng.integers(10,30))*10;fixed_b=fixed_a+int(rng.integers(8,16))*10
    rate_a=int(rng.integers(2,5));rate_b=rate_a+int(rng.integers(1,4))
    total_a=fixed_a+units*rate_a;total_b=fixed_b+units*rate_b;saving=total_b-total_a
    approved=bool(index%2) if variant is None else bool(variant)
    if split!='memory':approved=bool(rng.integers(2))
    choice=a if approved else b;total=total_a if approved else total_b
    sources=(f'[S1] The {context[2]} must choose one vendor for {context[0]}. Expected monthly volume is {units} {context[1]}. '
             f'Total monthly cost is the fixed fee plus rate multiplied by volume. Required certification is mandatory.\n'
             f'[S2] {a}: monthly fixed fee {fixed_a} credits; rate {rate_a} credits per unit. This older offer says certification is approved.\n'
             f'[S3] {b}: monthly fixed fee {fixed_b} credits; rate {rate_b} credits per unit. Certification is approved. It is the fallback if the other vendor is ineligible.\n'
             f'[S4] Later correction, overriding S2: {a} certification is {"approved" if approved else "revoked"}. '
             f'Quoted fees exclude a one-time onboarding cost that has not yet been priced. Monthly estimates are not total first-year costs.')
    qualifiers=['later correction overrides older offer','unpriced one-time onboarding','monthly not first-year']
    objective='Choose the lowest monthly-cost eligible option'
    if split=='train':
        a,b=f'Cargo-{index:03}',f'Dock-{index:03}';capacity_a=4;capacity_b=7
        total_a=fixed_a+((units+capacity_a-1)//capacity_a)*rate_a*10
        total_b=fixed_b+((units+capacity_b-1)//capacity_b)*rate_b*10
        limit=4;delay=3 if approved else 6
        sources=(f'[S1] A warehouse must ship {units} crates this month. Use whole trips, rounding trip count upward. Goods must arrive within {limit} days.\n'
          f'[S2] {a} carries {capacity_a} crates per trip, charges {rate_a*10} credits per trip plus {fixed_a} monthly credits, and originally quoted a two-day delivery.\n'
          f'[S3] {b} carries {capacity_b} crates per trip, charges {rate_b*10} credits per trip plus {fixed_b} monthly credits, and delivers in three days.\n'
          f'[S4] The later shipping bulletin supersedes the earlier quote: {a} now takes {delay} days. Spoilage insurance is required but is not included or priced. Do not treat the quoted freight bill as the complete insured cost.')
        eligible=[(a,total_a)] if approved else [];eligible.append((b,total_b));choice,total=min(eligible,key=lambda x:x[1])
        qualifiers=['round trips upward','later delivery correction controls eligibility','unpriced required spoilage insurance']
    elif split=='validation':
        a,b=f'Canal-{index:03}',f'Rill-{index:03}';included_a=8;included_b=15
        total_a=fixed_a+max(0,units-included_a)*rate_a;total_b=fixed_b+max(0,units-included_b)*rate_b
        quality=97 if approved else 89
        sources=(f'[S1] A water authority needs {units} units monthly. Only suppliers with a measured quality score at least 95 are eligible. Charges apply only to usage beyond the included allowance.\n'
          f'[S2] {a}: fixed fee {fixed_a}, included allowance {included_a}, then {rate_a} credits per unit. An old lab sheet showed quality 98.\n'
          f'[S3] {b}: fixed fee {fixed_b}, included allowance {included_b}, then {rate_b} credits per unit. Current quality is 99.\n'
          f'[S4] The replacement lab sheet reports {a} quality {quality}, superseding the old sheet. A temporary connection charge is excluded and still unknown; do not annualize the monthly comparison as a full project budget.')
        eligible=[(a,total_a)] if approved else [];eligible.append((b,total_b));choice,total=min(eligible,key=lambda x:x[1])
        qualifiers=['only usage above allowance is charged','latest measured quality controls eligibility','unknown connection charge']
    elif split=='development':
        a,b=f'Daystar-{index:03}',f'Lantern-{index:03}';night=units//3
        total_a=fixed_a+units*rate_a+night*(rate_a+3);total_b=fixed_b+units*rate_b+night*(rate_b+2)
        certified='licensed' if approved else 'unlicensed'
        sources=(f'[S1] A museum needs {units} daytime staff-hours and {night} nighttime staff-hours this month. A nighttime license is mandatory.\n'
          f'[S2] {a}: setup fee {fixed_a}, daytime rate {rate_a}, nighttime rate {rate_a+3} credits per hour. The early proposal claimed a valid license.\n'
          f'[S3] {b}: setup fee {fixed_b}, daytime rate {rate_b}, nighttime rate {rate_b+2} credits per hour; its nighttime license is valid.\n'
          f'[S4] The licensing register now lists {a} as {certified}, overriding the early proposal. Travel reimbursement is excluded and unpriced. These estimates cover this month only.')
        eligible=[(a,total_a)] if approved else [];eligible.append((b,total_b));choice,total=min(eligible,key=lambda x:x[1])
        qualifiers=['day and night hours have different rates','registry overrides old proposal','unpriced travel reimbursement']
    elif split=='confirmation':
        a,b=f'Helio-{index:03}',f'Volt-{index:03}';rebate_a=rate_a*30;rebate_b=rate_b*30
        total_a=units*rate_b*10-fixed_a-rebate_a;total_b=units*rate_a*10-fixed_b-rebate_b
        capacity=14 if approved else 8;minimum=12
        sources=(f'[S1] An energy cooperative compares two monthly operating plans for {units} production units. Net monthly benefit equals revenue per unit times production, minus operating fee, minus the required reserve. Choose the HIGHEST net benefit among plans whose storage capacity is at least {minimum} units.\n'
          f'[S2] {a}: revenue {rate_b*10} credits per production unit, operating fee {fixed_a}, reserve {rebate_a}. An old design assumed storage capacity 16.\n'
          f'[S3] {b}: revenue {rate_a*10} credits per production unit, operating fee {fixed_b}, reserve {rebate_b}; verified storage capacity 18.\n'
          f'[S4] The as-built inspection replaces the design assumption: {a} storage capacity is {capacity}. The estimates exclude an unknown decommissioning liability. Monthly operating benefit is not whole-life profit. In the final fields, Monthly total means the chosen plan\'s NET BENEFIT.')
        eligible=[(a,total_a)] if approved else [];eligible.append((b,total_b));choice,total=max(eligible,key=lambda x:x[1])
        objective='Choose the highest monthly-net-benefit eligible option'
        qualifiers=['maximize net benefit rather than minimize quoted charges','as-built capacity overrides the design','unknown decommissioning liability; not whole-life profit']
    if split!='memory':saving=abs(total_b-total_a)
    prompt=(sources+'\n\nWrite a 200–300 word decision memo. Choose the lowest monthly-cost eligible option, calculate both monthly totals and their difference, '
            'explain the correction and the unpriced exception, and attribute factual statements to [S1]–[S4]. '
            'End with three separate machine-readable lines: "Decision: <vendor>", "Monthly total: <number>", "Difference: <number>". '
            'Difference means the absolute difference between the two quoted monthly totals even if one vendor is ineligible.')
    if split!='memory':
        prompt=(sources+'\n\nWrite a 200–300 word decision memo. '+objective+', calculate both monthly totals and their absolute difference, explain the controlling update and the unpriced exception, and attribute claims to [S1]–[S4]. '
          'End with three separate machine-readable lines: "Decision: <vendor>", "Monthly total: <number>", "Difference: <number>". Use the definition of monthly total stated in the packet. Difference includes both alternatives even when one is ineligible.')
    suffix='' if variant is None else f'_v{int(variant)}'
    tid=f'{split}_evidence_{index:03}{suffix}'
    hidden=dict(decision=choice,monthly_total=total,difference=saving,totals={a:total_a,b:total_b},
        required_sources=['S1','S2','S3','S4'],required_qualifications=qualifiers,
        certification='approved' if approved else 'revoked',truth_source=sources)
    if split!='memory':hidden.pop('certification');hidden['eligibility_update_satisfied']=approved
    return TaskSpec(tid,'evidence',split,f'{split}_evidence_{index:03}',prompt,CONTRACTS['evidence'],1024,768,'evidence_fields',hidden,
                    dict(generator_seed=int({'train':100000,'validation':200000,'development':300000,'characterization':400000,'confirmation':500000,'memory':600000}[split]+index),scenario=context[0],variant=variant))


def writing(split,index):
    setting={'train':'mountain observatory','validation':'floating orchard','development':'underground library',
             'characterization':'tidal lighthouse','confirmation':'orbital greenhouse'}[split]
    name=f'Mira-{index:03}';number=3+index%5
    if index%2:
        prompt=(f'Write a 220–320 word story set in a {setting}. The narrator is {name}, an apprentice, speaking in first person. '
          f'A signal bell can ring only {number} times before its spring breaks. The narrator must spend the last ring to help a rival, '
          'sacrificing a personal opportunity; establish the opportunity before the sacrifice. No magic, dream twist, death, or omniscient narration. '
          'Use the phrase "a borrowed horizon" exactly once. End with the exact sentence "Tomorrow I will mend the bell." '
          'Aim for quiet tension, concrete sensory details and a causally earned ending; do not append commentary.')
        hidden=dict(kind='narrative',required=[name,'a borrowed horizon'],once='a borrowed horizon',ending='Tomorrow I will mend the bell.',
            facts=[f'bell permits {number} rings','last ring helps rival','personal opportunity established and sacrificed'],word_range=[220,320])
        other={
          'train':(f'Sana-{index:03}','third person, past tense','under a paper moon','She kept the envelope.',
                   'A courier discovers an unsigned apology inside a misdelivered envelope. She must decide whether to return it unopened or deliver it to its intended recipient, without learning whether forgiveness follows. Establish why a looming bus departure matters. No sacrifice involving a rival, signal bell, or magical event.'),
          'validation':(f'Ivo-{index:03}','second person, present tense','measured rain','The gate remains open.',
                   'A gardener is falsely accused of wasting water. The resolution must follow from a physical trace noticed early, without a confession or a narrator who knows other people\'s thoughts. Use no direct dialogue.'),
          'development':(f'Lena-{index:03}','third person limited, past tense','the margin remembers','She reshelved the map.',
                   'A librarian finds two incompatible catalog entries and helps a visitor find a misplaced map. The successful clue must be introduced before it is used. Include exactly one line of spoken dialogue and no unexplained coincidence.'),
          'confirmation':(f'Nila-{index:03}','first person, present tense','a small, stubborn orbit','Now the numbers can be trusted.',
                   f'A greenhouse technician has only {number} calibration seals. A concealed measurement error could protect her reputation but harm a future harvest. She must disclose the error and demonstrate a verifiable correction; the correction must consume a seal. No rival, bell, dream twist, or omniscient narrator.')}
        if split in other:
            character,perspective,motif,ending,outline=other[split]
            prompt=(f'Write a 220–320 word story set in a {setting}, centered on {character}, in {perspective}. '+outline+
                    f' Use the phrase "{motif}" exactly once. End with the exact sentence "{ending}". Aim for concrete details, restrained emotion, and a causally earned ending. Do not append commentary.')
            hidden=dict(kind='narrative',required=[character,motif],once=motif,ending=ending,facts=[outline,perspective],word_range=[220,320])
    else:
        prompt=(f'Explain the fictional Pebble-{index:03} message protocol to volunteer coordinators at a {setting} in 220–320 words. '
          f'The protocol has states IDLE, OFFERED, and CONFIRMED. A sender numbers offers consecutively. A receiver remembers the last {number} accepted offer IDs. '
          'Duplicates among remembered IDs are acknowledged but must never trigger the action again. An offer becomes CONFIRMED only when an acknowledgment returns to the sender. '
          'A lost acknowledgment can leave the receiver having acted while the sender remains OFFERED. IDs forgotten after the memory window can be repeated accidentally, so exactly-once delivery is not guaranteed. '
          'Give a concrete lost-acknowledgment example and one practical mitigation compatible with this specification. Do not claim that retries alone guarantee exactly once. '
          'Use the headings "How it works", "A failure example", and "A practical safeguard". Explain the technical terms without code.')
        hidden=dict(kind='explanation',required=['How it works','A failure example','A practical safeguard','OFFERED','CONFIRMED'],
            facts=['acknowledgment loss creates different sender/receiver states',f'only {number} IDs remembered','not exactly once','duplicates remembered cause no second action'],word_range=[220,320])
        protocols={
          'train':(f'Latch-{index:03}',f'A writer holds a lease lasting {number} ticks. Each lease has a monotonically increasing epoch. The storage node rejects a write from an older epoch even if the writer believes its lease is valid. A heartbeat may extend a current lease but cannot restore an old epoch. Clock delay can make the old writer act after a replacement begins; epoch checking at storage prevents stale writes but does not ensure every client can obtain a lease.',['lease','epoch'],['storage checks epoch','heartbeat cannot restore old epoch','safety does not imply availability']),
          'validation':(f'Parcel-{index:03}',f'A receiver buffer holds {number} messages. When full it drops the oldest undelivered message before storing the new arrival. Every stored message includes a sender epoch and sequence number; a sender restart increments the epoch and resets the sequence number. Sequence numbers alone cannot identify duplicates across restarts. The buffer policy guarantees bounded memory, not reliable delivery.', ['epoch','sequence'],['full buffer drops oldest undelivered message','epoch distinguishes restarts','bounded memory is not reliable delivery']),
          'development':(f'Lamp-{index:03}',f'Each frame has {number} data bits plus one parity bit chosen to make the total count of ones even. The receiver rejects odd parity and requests a retry. One flipped bit is detectable, but two flipped bits can pass parity. A retry does not establish whether a valid old frame is fresh. No encryption, sequence numbers or freshness mechanism are built in.', ['parity','retry'],['one bit flip detectable','two bit flips may pass','parity provides no freshness']),
          'confirmation':(f'Harbor-{index:03}',f'Three replicas store a value and a version vector. Each writer increments only its own vector component. If one vector is componentwise at least another and strictly larger somewhere, its value supersedes the other. Incomparable vectors are concurrent and BOTH values are kept until an application resolves them. A two-replica read quorum exposes available siblings but does not choose a semantically correct value. Losing two replicas blocks quorum reads; it does not erase stored data.', ['concurrent','quorum'],['incomparable versions retained together','read quorum does not resolve semantic conflicts','quorum loss does not erase data'])}
        if split in protocols:
            protocol,spec,required,facts=protocols[split]
            prompt=(f'Explain the fictional {protocol} protocol to volunteer coordinators at a {setting} in 220–320 words. Specification: '+spec+
                ' Give a concrete failure example and one practical mitigation compatible with this specification. Distinguish guarantees from limitations. Use the headings "How it works", "A failure example", and "A practical safeguard". Explain technical terms without code.')
            hidden=dict(kind='explanation',required=['How it works','A failure example','A practical safeguard']+required,facts=facts,word_range=[220,320])
    if index%2:prompt+=' Include the named central character\'s full name at least once in the story.'
    tid=f'{split}_writing_{index:03}'
    return TaskSpec(tid,'writing',split,tid,prompt,CONTRACTS['writing'],768,768,'writing_constraints',hidden,dict(setting=setting,template='narrative' if index%2 else 'protocol'))


def synthetic_code(split,index):
    specs=[
       'Return a new list keeping the first occurrence of each integer, in input order.',
       'Return run-length encoding of a string as a list of (character, count) tuples; the empty string yields an empty list.',
       'Given a list of inclusive integer intervals, merge overlapping intervals and return a sorted list of merged tuples; adjacent nonoverlapping intervals stay separate.',
       'Given a list of integers, return the longest strictly increasing contiguous run length; return zero for an empty list.',
       'Given a string containing parentheses and other characters, return whether parentheses are balanced; ignore other characters.',
       'Given a list of integers and a positive width, return sums of all contiguous windows of that width; return [] when width exceeds the list length.',
       'Given a list of strings, group case-insensitive anagrams preserving the order of first encountered groups and the order of strings within each group.',
       'Given a dictionary mapping nodes to lists of neighbors and a start node, return a list of reachable nodes in breadth-first order, using the supplied neighbor order and visiting each node once.']
    if split=='validation':
        specs=[
          'Given a list of integers, return a dictionary counting occurrences of each integer.',
          'Given a list of strings, return their longest common prefix, or the empty string if none exists.',
          'Given a nonnegative integer, return its binary representation without a prefix; zero is represented as "0".',
          'Given two sorted integer lists, merge them into one sorted list retaining duplicates.',
          'Given a list of integers and a target, return all pairs of distinct indices i < j whose values sum to target, sorted lexicographically.',
          'Given a string, return the first nonrepeating character or None if there is none.',
          'Given a rectangular matrix, return its transpose as a list of lists; an empty matrix yields an empty list.',
          'Given a string of words separated by whitespace, return a dictionary counting words case-insensitively.']
    tid=f'{split}_code_{index:03}'
    prompt=f'Write a Python function named solve_{index:03}. '+specs[index%len(specs)]+' Use only the Python standard library. Include an appropriate signature and docstring.'
    return TaskSpec(tid,'code',split,tid,prompt,CONTRACTS['code'],1024,512,'unscored_training',{},dict(template=index%len(specs)))


def main():
    cfg=read('configs/phase2_v1.json');root=Path(cfg['output'])/'tasks';root.mkdir(parents=True,exist_ok=True)
    ds=load_dataset('openai/gsm8k','main',revision=cfg['gsm8k_revision'])
    excluded=set(read('results/followup_v1/experiment_manifest.json')['identity']['test_ids'])
    for pair in ['qwen3_1.7b_to_0.6b','qwen3_4b_to_0.6b']:excluded.update(read(f'results/{pair}/reasoning_manifest.json')['selected_indices'])
    test_ids=[int(i) for i in np.random.default_rng(cfg['seed']).permutation(len(ds['test'])) if i not in excluded]
    old=read('results/followup_v1/experiment_manifest.json')['identity'];excluded_train=set(old['train_ids']+old['validation_ids'])
    train_ids=[int(i) for i in np.random.default_rng(cfg['seed']).permutation(len(ds['train'])) if i not in excluded_train]
    code_path=Path(cfg['output'])/'datasets/HumanEvalPlus-OriginFmt-v0.1.10.jsonl.gz'
    code=[json.loads(l) for l in gzip.decompress(code_path.read_bytes()).decode().splitlines()]
    code_order=np.random.default_rng(cfg['seed']).permutation(len(code)).tolist()
    counts={'development':{f:16 for f in CONTRACTS},'characterization':cfg['characterization_targets'],
        'confirmation':cfg['confirmation_targets'],'train':{f:32 for f in CONTRACTS},'validation':{f:8 for f in CONTRACTS}}
    splits={};test_cursor=train_cursor=code_cursor=0
    for split,count in counts.items():
        tasks=[]
        for family,n in count.items():
            for index in range(n):
                if family=='arithmetic':
                    official='train' if split in ['train','validation'] else 'test'
                    if official=='test':idx=test_ids[test_cursor];test_cursor+=1
                    else:idx=train_ids[train_cursor];train_cursor+=1
                    item=ds[official][idx];tid=f'{split}_gsm_{idx}'
                    task=TaskSpec(tid,family,split,f'gsm_{official}_{idx}',item['question'],CONTRACTS[family],2048,64,'numeric',
                                  dict(gold=numeric_answer(item['answer'].split('####')[-1])),dict(dataset='openai/gsm8k',official_split=official,index=idx))
                elif family=='code':
                    if split in ['train','validation']:task=synthetic_code(split,index)
                    else:
                        item=code[code_order[code_cursor]];code_cursor+=1;tid=f'{split}_'+item['task_id'].replace('/','_')
                        task=TaskSpec(tid,family,split,item['task_id'],item['prompt'],CONTRACTS[family],1024,512,'humanevalplus',
                            dict(test=item['test'],canonical_solution=item['canonical_solution'],entry_point=item['entry_point']),dict(dataset='HumanEvalPlus-OriginFmt',release='v0.1.10',official_id=item['task_id']))
                elif family=='evidence':task=evidence(split,index)
                else:task=writing(split,index)
                tasks.append(task)
        splits[split]=tasks
        immutable(root/f'{split}.json',[asdict(t) for t in tasks])
    memory=[evidence('memory',i,v) for i in range(12) for v in [0,1]]
    immutable(root/'memory.json',[asdict(t) for t in memory])
    cluster_sets={s:{t.cluster_id for t in ts} for s,ts in splits.items()}
    for a in cluster_sets:
        for b in cluster_sets:
            if a!=b:assert not cluster_sets[a]&cluster_sets[b]
    visible_hashes=[digest(t.visible()['prompt']) for ts in splits.values() for t in ts]
    assert len(visible_hashes)==len(set(visible_hashes))
    immutable(root/'reservation.json',dict(config=artifact('configs/phase2_v1.json'),
        source_code=snapshot(['scripts/phase2_prepare_tasks.py','gearshift/phase2_tasks.py','gearshift/phase2_io.py']),
        datasets=dict(gsm8k_revision=cfg['gsm8k_revision'],gsm8k_fingerprints={s:ds[s]._fingerprint for s in ds},
                      humanevalplus=dict(url='https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-OriginFmt.jsonl.gz',**artifact(code_path))),
        split_files={s:artifact(root/f'{s}.json') for s in splits},excluded_gsm_test_ids=sorted(excluded),
        counts=counts,unallocated_code_tasks=len(code)-code_cursor,access_policy=cfg['evaluation_access'],
        procedural_limit='Each split has distinct evidence calculation/eligibility rules, protocol specifications, and narrative plots. Samples within a split share procedural templates; these are bounded custom diagnostic families, not established broad benchmarks.'))
    print({s:len(t) for s,t in splits.items()})


if __name__=='__main__':main()
