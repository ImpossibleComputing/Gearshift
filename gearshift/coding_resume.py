"""Resume a recorded draw by deterministic recomputation, never by reselection."""
import json
from pathlib import Path
from gearshift.coding_control import sha

def verify_prefix(tokens,previous,final=False):
    n=min(len(tokens),len(previous))
    if tokens[:n]!=previous[:n] or (final and len(tokens)<len(previous)):
        raise ValueError('Recovered token stream diverged from the saved sample; no replacement draw is permitted')

def load_parent(root,task_id):
    path=Path(root)/'tasks'/task_id.replace('/','__');prefixes={};complete={}
    for p in path.glob('partial_*.json'):
        r=json.loads(p.read_text());prefixes[r['segment']]=r['token_ids']
    for name,segment,key in [('source_history','source_reasoning','reasoning_ids'),('small_history','small_reasoning','reasoning_ids'),
        ('A','answer_A','answer_ids'),('B','answer_B','answer_ids'),('D','answer_D','answer_ids')]:
        p=path/(name+'.json')
        if p.exists():
            r=json.loads(p.read_text());prefixes[segment]=r[key];complete[segment]={'tokens':r[key],'sha256':sha(p)}
    return prefixes,complete

def verify_segment(tokens,segment,prefixes,complete):
    verify_prefix(tokens,prefixes.get(segment,[]),final=True)
    if segment in complete and tokens!=complete[segment]['tokens']:
        raise ValueError('A completed recorded sample changed on recomputation')
