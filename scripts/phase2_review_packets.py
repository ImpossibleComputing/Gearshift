#!/usr/bin/env python3
"""Reserve and export blind human review and rule-selected illustrative cases."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gearshift.phase2_io import read,write,immutable,artifact,digest

ROOT=Path('results/phase2_v1')


def reserve():
    tasks=read(ROOT/'tasks/characterization.json');rng=np.random.default_rng(20260915);chosen=[]
    for family in ['evidence','writing']:
        ids=sorted(t['task_id'] for t in tasks if t['family']==family)
        for i,index in enumerate(rng.permutation(len(ids))[:15]):
            chosen.append(dict(task_id=ids[index],family=family,left='M',right=['C','B/newturn','B/native'][i%3]))
    immutable(ROOT/'tasks/human_reservation.json',dict(seed=20260915,selected=chosen,
        definition='Thirty distinct original cases: fifteen per prose family, five per family for each of M/C, M/B/newturn, M/B/native. Selected without candidate outputs or judgments. This is a stratified sample of these declared comparisons, not all study outputs.',
        tasks=artifact(ROOT/'tasks/characterization.json'),human_labels='pending'))


def export():
    root=ROOT/'characterization';keys=read(root/'judging/condition_key.json');reservation=read(ROOT/'tasks/human_reservation.json')
    dest=ROOT/'human_review';chosen=[];answers=[]
    for i,selection in enumerate(reservation['selected'],1):
        candidates=[k for k in keys if k['task_id']==selection['task_id'] and k['left']==selection['left'] and k['right']==selection['right'] and k['orientation']==0]
        if len(candidates)!=1:raise ValueError('Expected reserved blind comparison missing')
        key=candidates[0];packet=read(root/'judging/packets'/f'{key["packet_id"]}.json')
        anonymous=f'human_{i:02d}';packet['packet_id']=anonymous
        immutable(dest/'blinded'/f'{anonymous}.json',packet)
        chosen.append(dict(human_packet_id=anonymous,**key,anonymous_packet_sha256=digest(packet)))
        answers.append(dict(packet_id=anonymous,A={d:None for d in ['task_fulfillment','correctness_consistency','coverage','clarity_style','acceptable']},
            B={d:None for d in ['task_fulfillment','correctness_consistency','coverage','clarity_style','acceptable']},preference=None,rationale=None))
    immutable(dest/'separate_condition_key.json',chosen);immutable(dest/'blinded/blank_labels.json',answers)
    human_viewer(dest/'blinded')
    (dest/'blinded/README.md').write_text('Read only this blinded directory when judging. Score each response on the included frozen rubric, then record preference (A, B, tie, neither_acceptable), acceptability and a brief rationale in a copy of blank_labels.json. All thirty labels are pending. Do not inspect the separate condition key, experiment reports or timing before labeling. The pairs were sampled before candidate inspection; disagreement cases are exported separately and are not part of the representative sample.\n')
    disagreement_keys=[]
    for stage in ROOT.iterdir():
        p=stage/'judging/order_disagreements.json'
        if not p.exists():continue
        original=read(stage/'judging/condition_key.json')
        for case in read(p):
            for key in original:
                if key['pair_id']!=case['pair_id']:continue
                packet=read(stage/'judging/packets'/f'{key["packet_id"]}.json')
                identifier=digest(dict(stage=stage.name,packet=packet['packet_id']))[:24];packet['packet_id']=identifier
                immutable(dest/'disagreements_blinded'/f'{identifier}.json',packet)
                disagreement_keys.append(dict(anonymous_packet_id=identifier,stage=stage.name,**key))
    write(dest/'disagreements_separate_key.json',disagreement_keys)
    write(dest/'status.json',dict(representative_pairs=len(chosen),human_labels='pending',disagreement_orientations=len(disagreement_keys),
        caution='Selectively identified order disagreements are diagnostic and must not enter representative human agreement percentages.'))
    print('Exported',len(chosen),'blinded human pairs;',len(disagreement_keys),'disagreement orientations')


def human_viewer(folder):
    packets=[read(p) for p in sorted(folder.glob('human_*.json'))]
    data=json.dumps(packets,ensure_ascii=False).replace('<','\\u003c')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Anonymous response review</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;color:#18232c;background:#fafafa}h1{font-size:26px}button,select,textarea{font:inherit}button{padding:7px 16px;margin:5px}pre{white-space:pre-wrap;word-break:break-word;font:15px/1.6 system-ui,sans-serif}article,details{background:white;border:1px solid #ced5da;border-radius:8px;padding:18px;margin:12px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}label{display:block;margin:9px 0}select{margin-left:12px}textarea{width:95%;min-height:90px}header{position:sticky;top:0;background:#fafafa;padding:10px 0;border-bottom:1px solid #ced5da}@media(max-width:750px){.pair{grid-template-columns:1fr}}</style>
<h1>Anonymous response review</h1><p>Thirty pairs, sampled before answer inspection. Read only this blinded packet. Score each dimension using the rubric, then record preference and a brief rationale. The form runs locally and makes no network requests. Download your labels when finished.</p>
<header><button id="prev">Previous</button><b id="number"></b><button id="next">Next</button><button id="download">Download labels</button><span id="progress"></span></header>
<details><summary>Frozen rubric and scoring anchors</summary><pre id="rubric"></pre></details>
<article><h2>Task and supporting evidence</h2><pre id="task"></pre></article><main class="pair" id="pair"></main>
<article><label>Preference<select id="preference"><option value="">Unscored</option><option>A</option><option>B</option><option value="tie">Tie</option><option value="neither_acceptable">Neither acceptable</option></select></label><label>Evidence-based rationale<textarea id="rationale"></textarea></label></article>
<script>const packets=PACKET_DATA;let index=0;const dimensions=['task_fulfillment','correctness_consistency','coverage','clarity_style'];const key='anonymous-review-PACKET_HASH';let labels;try{labels=JSON.parse(localStorage.getItem(key)||'null');}catch(e){}labels=labels||packets.map(p=>({packet_id:p.packet_id,A:{},B:{},preference:null,rationale:''}));
function save(){try{localStorage.setItem(key,JSON.stringify(labels));}catch(e){}document.getElementById('progress').textContent=labels.filter(r=>r.preference).length+' / '+packets.length+' preferences recorded';}
function render(){let p=packets[index],r=labels[index];document.getElementById('number').textContent=(index+1)+' / '+packets.length;document.getElementById('task').textContent=p.task;document.getElementById('rubric').textContent=JSON.stringify(p.rubric,null,2);let pair=document.getElementById('pair');pair.replaceChildren();for(let side of ['A','B']){let a=document.createElement('article'),h=document.createElement('h2'),text=document.createElement('pre');h.textContent='Response '+side;text.textContent=p.candidates[side];a.append(h,text);for(let d of dimensions){let label=document.createElement('label'),select=document.createElement('select');label.textContent=d.replaceAll('_',' ');select.append(new Option('Unscored',''));for(let n=0;n<=4;n++)select.append(new Option(n,n));select.value=r[side][d]??'';select.onchange=()=>{r[side][d]=select.value===''?null:Number(select.value);accept(r[side]);save();};label.append(select);a.append(label);}let label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=!!r[side].material_violation;check.onchange=()=>{r[side].material_violation=check.checked;accept(r[side]);save();};label.append(check,document.createTextNode(' Material factual or causal violation'));a.append(label);pair.append(a);}document.getElementById('preference').value=r.preference||'';document.getElementById('rationale').value=r.rationale||'';save();}
function accept(r){r.acceptable=dimensions.every(d=>typeof r[d]==='number')?dimensions.every(d=>r[d]>=3)&&!r.material_violation:null;}
document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);render();};document.getElementById('next').onclick=()=>{index=Math.min(packets.length-1,index+1);render();};document.getElementById('preference').onchange=e=>{labels[index].preference=e.target.value||null;save();};document.getElementById('rationale').oninput=e=>{labels[index].rationale=e.target.value;save();};document.getElementById('download').onclick=()=>{let link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify({human_labels:labels,packet_ids:packets.map(p=>p.packet_id)},null,2)],{type:'application/json'}));link.download='human-review-labels.json';link.click();URL.revokeObjectURL(link.href);};render();</script></html>'''
    (folder/'review.html').write_text(html.replace('PACKET_HASH',digest(packets)).replace('PACKET_DATA',data))


def examples():
    selected=[]
    for stage in ROOT.iterdir():
        p=stage/'objective_scores.json'
        if not p.exists() or stage.name not in ['characterization','confirmation_1p7_to_0p6','confirmation_4b_to_0p6']:continue
        scored=read(p)['rows'];groups=defaultdict(dict)
        for row in scored:groups[row['task_id']][row['condition']]=row
        buckets={}
        candidates=[]
        for task_id,rows in sorted(groups.items()):
            for mapped in ['M','M/frozen','M/initial','M/ordinary/20260915','M/boundary/20260915']:
                if mapped in rows and 'C' in rows:candidates.append((task_id,rows,mapped))
        for task_id,rows,mapped in candidates:
            family=rows[mapped]['family'];metric={'arithmetic':'correct','code':'correct','evidence':'all_fields_correct','writing':'all_literal_constraints'}[family]
            a=rows[mapped]['score'].get(metric);b=rows['C']['score'].get(metric)
            if a is None or b is None:continue
            bucket='both_pass_checks' if a and b else 'both_fail_checks' if not a and not b else 'mapped_only_pass' if a else 'native_only_pass'
            key=(mapped,family,bucket)
            if key in buckets:continue
            raw=read(stage/'questions'/f'{task_id}.json')
            item=dict(stage=stage.name,task_id=task_id,family=family,bucket=bucket,metric=metric,mapped=mapped,
                source_record=str(stage/'questions'/f'{task_id}.json'),source_record_artifact=artifact(stage/'questions'/f'{task_id}.json'),
                trajectory=raw['trajectory'],answers=[r for r in raw['rows'] if r['condition'] in [mapped,'C','B/newturn','B/native']],
                scores={c:rows[c]['score'] for c in [mapped,'C']})
            buckets[key]=item
        selected.extend(buckets.values())
    write(ROOT/'review_examples.json',dict(selection_rule='First lexical task ID within each stage × mapper condition × family × paired objective-check outcome. Includes frozen/initial and both first-seed adapted arms where available. The final reporting expansion to adapted arms was made after objective scoring; it is an illustrative rule, not a pre-outcome representative sample. No subjective quality is inferred from field or literal checks. Order disagreements have a separate export.',examples=selected))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['reserve','export','examples']);a=p.parse_args()
    {'reserve':reserve,'export':export,'examples':examples}[a.action]()
