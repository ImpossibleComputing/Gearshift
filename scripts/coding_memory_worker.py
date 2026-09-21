#!/usr/bin/env python3
"""Run the gated memory probe with its one declared source-offload fallback."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import sha,write

ROOT=Path(__file__).resolve().parents[1]


def offload_eligible(gate):
    if gate.get('passed') is not False:
        return False
    return (gate.get('exception') in ['OutOfMemoryError','CUDAOutOfMemoryError']
        or 'Full-context memory/headroom gate failed' in gate.get('error','')
        or 'CUDA out of memory' in gate.get('error',''))


def main():
    spec=json.loads(Path(os.environ['GEARSHIFT_WORKER_SPEC']).read_text())
    out=ROOT/spec['result_root'];out.mkdir(parents=True,exist_ok=False)
    status_path=ROOT/spec['worker_status_path']
    state={'state':'running','phase':'full_context_memory','stage_identity':spec['stage_identity'],'worker_id':spec['worker_id']}
    stop=threading.Event()
    def heartbeat():
        while not stop.is_set():
            write(status_path,{**state,'heartbeat_epoch':time.time()});stop.wait(10)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    attempts=[]
    try:
        for offload in [False,True]:
            phase='memory_v2_source_cpu_diagnostic' if offload else 'memory_v2_both_gpu'
            state['phase']=phase
            probe=out/'probes'/phase
            command=[sys.executable,'scripts/coding_memory_preflight.py',
                '--baseline-root',spec['baseline_root'],'--allocation',spec['allocation_path'],
                '--approval','evidence/coding_pilot_v1/parallel_500h_approved.json',
                '--previous-approval','evidence/coding_pilot_v1/experiment_100h_approved.json',
                '--output-root',str(probe.relative_to(ROOT))]
            if offload:command.append('--source-on-cpu')
            allowance=spec['deadline_epoch']-time.time()-120
            if allowance<=0:raise TimeoutError('Memory worker deadline reached')
            result=subprocess.run(command,cwd=ROOT,timeout=allowance,check=False)
            path=probe/'memory_gate.json'
            if not path.exists():raise RuntimeError('Memory process ended without a gate receipt')
            gate=json.loads(path.read_text())
            attempts.append({'root':str(path.parent.relative_to(ROOT)),'gate_sha256':sha(path),
                             'passed':gate['passed'],'returncode':result.returncode})
            write(out/'attempts.json',attempts)
            if result.returncode==0 and gate['passed'] and gate.get('training_ready') is True:
                write(out/'memory_gate.json',{'passed':True,'selected_probe_root':str(path.parent.relative_to(ROOT)),
                    'source_weight_cpu_offload':offload,'training_ready':True,'probe_gate_sha256':sha(path),
                    'stage_identity':spec['stage_identity'],'attempts':attempts})
                state.update(state='complete',memory_gate_passed=True)
                return
            if offload or not offload_eligible(gate):
                write(out/'memory_gate.json',{'passed':False,'stage_identity':spec['stage_identity'],'attempts':attempts})
                state.update(state='complete',memory_gate_passed=False)
                return
    except BaseException as exc:
        state.update(state='failed',error=str(exc),exception=type(exc).__name__)
        raise
    finally:
        stop.set();thread.join();write(status_path,{**state,'heartbeat_epoch':time.time()})


if __name__=='__main__':main()
