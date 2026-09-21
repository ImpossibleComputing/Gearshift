#!/usr/bin/env python3
"""Durable Studio control turns; cloud budget enforcement must be a separate watchdog."""
from pathlib import Path
import datetime
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'evidence/phase2/studio_transfer'
from phase2_execution_authorization import DEADLINE, AUTHORIZATION_END_UTC


def write(name, obj):
    path = STATE / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def main():
    os.chdir(ROOT)
    cfg = json.loads((ROOT / 'configs/phase2_studio_runtime.json').read_text())
    os.environ.update(cfg['environment'])
    sys.path.insert(0, str(ROOT / 'scripts'))
    from phase2_usage_guard import included_usage
    cli = cfg['environment']['GEARSHIFT_JUDGE_CLI']
    env = {k: v for k, v in os.environ.items() if k in ['PATH', 'HOME', 'USER', 'TMPDIR', 'LANG', 'LC_ALL'] or k.startswith('GEARSHIFT_')}
    write('supervisor_process.json', dict(pid=os.getpid(), ppid=os.getppid(), host=os.uname().nodename,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), deadline_utc=AUTHORIZATION_END_UTC))
    turn = 0
    while time.time() < DEADLINE and not (STATE / 'CONTROL_DONE.json').exists():
        try:
            usage = included_usage()
        except Exception as exc:
            usage = dict(allowed=False, error=type(exc).__name__)
        write('controller_usage.json', usage)
        if not usage.get('allowed'):
            write('supervisor_status.json', dict(state='included_usage_unavailable', next_check_seconds=300,
                note='Independent scientific workers and cloud watchdog remain responsible for completion/cleanup. No credit fallback.'))
            for _ in range(10):
                if time.time() >= DEADLINE: break
                time.sleep(30)
            continue
        prompt = ('Continue the owner-authorized Gearshift research on this Studio. Read STUDIO_CONTROL_HANDOFF.md and STUDIO_STATUS.md if it exists, '
            'then inspect the durable current state. The laptop can be asleep. Complete the bounded research autonomously. '
            'FIRST establish and test the separate cloud budget/deadline watchdog before any paid pod is created, then finish independent-operation readiness. '
            'Do not stop at a plan. Work concretely for this turn, persist status/next steps, and yield when a healthy long-running job needs time; '
            'this supervisor will call you again in five minutes. Keep a turn under ten minutes where practical. No publishing, pushes, extra providers, '
            'credit resets, or spending beyond the documented cap. The owner removed the original time cutoff; read configs/phase2_execution_authorization.json. Create CONTROL_DONE.json only when the actual deliverables and cloud cleanup are complete.')
        last = STATE / 'controller_last_message.txt'
        if last.exists():
            prompt += '\n\nPrevious control turn summary:\n' + last.read_text()[-14000:]
        stem = STATE / f'controller_turn_{turn:04d}'
        while stem.with_suffix('.jsonl').exists():
            turn += 1
            stem = STATE / f'controller_turn_{turn:04d}'
        command = [cli, 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral', '--json', '-s', 'danger-full-access',
            '-c', 'approval_policy="never"', '-m', cfg['environment']['GEARSHIFT_JUDGE_MODEL'],
            '-c', 'model_reasoning_effort="xhigh"', '-c', 'features.plugins=false', '-c', 'features.apps=false',
            '-c', 'features.memories=false', '-c', 'features.hooks=false', '-o', str(last), '-']
        with stem.with_suffix('.jsonl').open('w') as output, stem.with_suffix('.stderr.txt').open('w') as error:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=output, stderr=error, text=True, start_new_session=True)
            write('supervisor_status.json', dict(state='control_turn_running', turn=turn, child_pid=process.pid))
            try:
                process.communicate(prompt, timeout=min(900, max(1, DEADLINE-time.time())))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
        write(f'controller_turn_{turn:04d}.json', dict(returncode=process.returncode,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        turn += 1
        if (STATE / 'CONTROL_DONE.json').exists(): break
        write('supervisor_status.json', dict(state='waiting_for_next_control_turn', next_check_seconds=300))
        for _ in range(10):
            if time.time() >= DEADLINE or (STATE / 'CONTROL_DONE.json').exists(): break
            time.sleep(30)
    write('supervisor_status.json', dict(state='complete' if (STATE / 'CONTROL_DONE.json').exists() else 'approval_required_at_deadline',
        ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
