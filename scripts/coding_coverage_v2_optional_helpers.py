#!/usr/bin/env python3
"""Optional, bounded helper acquisition on the primary pod; never controls training.

At most two extra GPU slots are purchased over this controller's entire lifetime.
Failed acquired slots are not replaced. A create with uncertain outcome blocks
another create until exact-name inventory reconciliation has completed twice.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_coverage_v2_lease import (
    atomic_json, control_root, delete_own_pod, load_lease, read_private_key, redacted_error,
    sha256_file, verify_lease,
)
import coding_coverage_v2_dispatch as dispatch


def read(path):
    return json.loads(Path(path).read_text())


def helper_lease(primary, pod_id, count, epoch):
    if count not in (1, 2) or epoch >= primary['deadline_epoch'] - 120:
        raise ValueError('no funded helper execution window remains')
    original_hours = (primary['deadline_epoch'] - primary['allocation_epoch']) / 3600
    value = {**primary, 'pod_id': pod_id, 'gpu_count': count,
             'allocation_epoch': epoch, 'upper_hourly_usd': 5.55 * count,
             'other_reserved_usd': 5.55 * (4-count) * original_hours,
             'other_reserved_gpu_hours': (4-count) * original_hours,
             'control_relative': 'allocations/' + pod_id}
    verify_lease(value)
    return value


def helper_plan(primary, lease_relative, count, lease_sha256):
    value = copy.deepcopy(primary)
    value.update(allocation_lease_path=lease_relative, allocation_lease_sha256=lease_sha256,
                 execution_role='evaluation_helper',
                 local_gpu_count=count)
    return value


def inventory_matches(intent, pods):
    matches = [pod for pod in pods if pod.get('name') == intent['name']]
    if len(matches) > 1:
        raise ValueError('ambiguous duplicate allocation name; acquisition paused')
    return matches[0] if matches else None


class Controller:
    def __init__(self, args, repo=ROOT):
        self.args, self.repo = args, Path(repo).resolve()
        self.plan_path, self.lease_path = self.repo / args.plan, self.repo / args.lease
        self.plan = read(self.plan_path)
        self.primary = load_lease(self.lease_path, self.plan['allocation_lease_sha256'])
        if (self.repo / self.plan['allocation_lease_path']).resolve() != self.lease_path.resolve():
            raise ValueError('controller primary lease differs from frozen dispatch plan')
        if self.primary['gpu_count'] != 2 or self.plan.get('execution_role', 'primary') != 'primary':
            raise ValueError('optional helpers require a two-GPU primary')
        if self.plan['experiment_id'] != self.primary['experiment_id']:
            raise ValueError('primary identity differs')
        self.root = Path(self.primary['allowed_result_root'])
        if self.root.resolve() != (self.repo / self.plan['result_root']).resolve():
            raise ValueError('shared scientific result root differs')
        self.directory = self.lease_path.parent / 'optional_helpers'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = read_private_key(args.key_file, self.root)
        self.ssh_key = Path(args.ssh_key or ('/root/.gearshift-helper-' + self.primary['experiment_id'] + '.key'))
        if not self.ssh_key.is_absolute() or self.ssh_key.resolve().is_relative_to(self.repo):
            raise ValueError('ephemeral SSH private key must remain outside the repository')
        self.prefix = 'gearshift-coverage-v2-autoeval-' + self.primary['experiment_id'].split('_')[-1] + '-'

    def event(self, state, **details):
        value = {'epoch': time.time(), 'experiment_id': self.primary['experiment_id'],
                 'state': state, **details}
        atomic_json(self.directory / 'status.json', value)
        with (self.directory / 'events.jsonl').open('a') as output:
            output.write(json.dumps(value, allow_nan=False) + '\n')
            output.flush(); os.fsync(output.fileno())

    def finished(self):
        return (time.time() >= self.primary['deadline_epoch'] - 900
                or (control_root(self.primary) / 'lease_guard/STOP').exists()
                or (self.root / 'execution_complete.json').exists()
                or all((self.root / 'evaluation/jobs' / job['job_id'] / 'generation_complete.json').exists()
                       for job in self.plan['expected_jobs']))

    def capacity(self):
        fields = ['g' + str(n) + ':lowestPrice(input:{gpuCount:' + str(n)
                  + ',secureCloud:true,dataCenterId:' + json.dumps(self.args.region)
                  + '}){stockStatus uninterruptablePrice}' for n in (1, 2)]
        query = '{gpuTypes(input:{id:"NVIDIA H200"}){' + ' '.join(fields) + '}}'
        request = urllib.request.Request('https://api.runpod.io/graphql',
            data=json.dumps({'query': query}).encode(), headers={
                'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json',
                'User-Agent': 'gearshift-optional-helpers'})
        with urllib.request.urlopen(request, timeout=25) as response:
            result = json.load(response)
        if result.get('errors'):
            raise RuntimeError('capacity query failed')
        rows = result['data']['gpuTypes'][0]
        atomic_json(self.directory / 'latest_capacity.json', {'epoch': time.time(), 'region': self.args.region, 'quotes': rows})
        return [n for n in (2, 1) if isinstance(rows.get('g'+str(n)), dict)
                and rows['g'+str(n)].get('stockStatus') in ('Low', 'Medium', 'High')
                and isinstance(rows['g'+str(n)].get('uninterruptablePrice'), (float, int))
                and 0 < rows['g'+str(n)]['uninterruptablePrice'] * 1.2 + .04*n <= 5.55*n]

    def ensure_ssh_key(self):
        if not self.ssh_key.exists():
            subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(self.ssh_key)],
                           stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=15)
        if self.ssh_key.is_symlink() or self.ssh_key.stat().st_mode & 0o077:
            raise ValueError('ephemeral SSH private key permissions are not private')
        public = self.ssh_key.with_suffix(self.ssh_key.suffix + '.pub').read_text().strip()
        if not public.startswith('ssh-ed25519 ') or '\n' in public:
            raise ValueError('invalid ephemeral public key')
        return public

    def attempts(self):
        return sorted(self.directory.glob('attempt_*'))

    def reconcile(self, pods):
        attempts = self.attempts()
        names = {read(path / 'intent.json')['name'] for path in attempts}
        if any(pod.get('name', '').startswith(self.prefix) and pod['name'] not in names for pod in pods):
            raise ValueError('unregistered helper namespace resource; acquisition paused')
        if any(pod.get('networkVolumeId') == self.primary['network_volume_id']
               and pod.get('id') != self.primary['pod_id'] and pod.get('name') not in names for pod in pods):
            raise ValueError('unregistered pod shares experiment volume; acquisition paused')
        for path in attempts:
            intent = read(path / 'intent.json')
            match = inventory_matches(intent, pods)
            if match:
                match = dispatch.cli(self.args, 'pod', 'get', match['id'])
                if match.get('gpuCount') != intent['gpu_count'] or match.get('networkVolumeId') != self.primary['network_volume_id']:
                    raise ValueError('provider helper identity differs from reservation')
                if not (path / 'pod.json').exists():
                    atomic_json(path / 'pod.json', dispatch.safe_pod(match))
                elif read(path / 'pod.json')['id'] != match['id']:
                    raise ValueError('provider reused a helper allocation name')
            elif not (path / 'pod.json').exists() and not (path / 'absent_confirmed.json').exists():
                first = path / 'first_absent.json'
                if not first.exists():
                    atomic_json(first, {'epoch': time.time()})
                    return False
                if time.time() - read(first)['epoch'] < 60:
                    return False
                atomic_json(path / 'absent_confirmed.json', {'epoch': time.time(),
                    'two_fresh_inventory_checks': True, 'provider_resources_confirmed_absent': True})
        return True

    def acquire(self, count):
        attempts = self.attempts()
        purchased = [path for path in attempts if (path / 'pod.json').exists()]
        if (len(purchased) >= 2 or count not in (1, 2)
                or count + sum(read(path / 'intent.json')['gpu_count'] for path in purchased) > 2):
            raise ValueError('lifetime optional-helper allocation ceiling reached')
        if any(not (path / 'pod.json').exists() and not (path / 'absent_confirmed.json').exists() for path in attempts):
            raise ValueError('previous ambiguous create has not been reconciled')
        folder = self.directory / f'attempt_{len(attempts)+1:04d}'
        folder.mkdir()
        name = self.prefix + f'{len(attempts)+1:04d}'
        reservation = helper_lease(self.primary, 'reservation_pending', count, time.time())
        intent = {'name': name, 'gpu_count': count, 'epoch': time.time(), 'reservation': reservation}
        atomic_json(folder / 'intent.json', intent)
        try:
            pod = dispatch.cli(self.args, 'pod', 'create', '--name', name, '--image', dispatch.IMAGE,
                '--gpu-id', 'NVIDIA H200', '--gpu-count', str(count), '--cloud-type', 'SECURE',
                '--data-center-ids', self.args.region, '--network-volume-id', self.primary['network_volume_id'],
                '--volume-mount-path', '/workspace', '--container-disk-in-gb', '40', '--ports', '22/tcp',
                '--env', json.dumps({'PUBLIC_KEY': self.ensure_ssh_key()}), '--ssh')
            atomic_json(folder / 'pod.json', dispatch.safe_pod(pod))
        except Exception as error:
            atomic_json(folder / 'create_failed.json', {'epoch': time.time(), **redacted_error(error),
                'provider_reconciliation_required': True})
        return folder

    def remote_args(self, pod, lease_path):
        ports = pod.get('portMappings') or {}
        endpoint = pod.get('ssh') or {}
        host = endpoint.get('ip') or pod.get('publicIp')
        port = endpoint.get('port') or ports.get('22') or ports.get('22/tcp')
        if not host or not port:
            raise ConnectionError('helper SSH endpoint is not ready')
        return SimpleNamespace(cli=self.args.cli, ssh_host=host, ssh_port=int(port),
            ssh_key=str(self.ssh_key), lease=str(lease_path.relative_to(self.repo)),
            provider_config=self.args.provider_config)

    def launch_remote(self, remote, command, log):
        launcher = 'import subprocess; f=open(' + repr(str(log)) + ", 'ab'); p=subprocess.Popen(" + repr(command)
        launcher += ',cwd=' + repr(str(self.repo)) + ',stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True); print(p.pid)'
        return int(dispatch.ssh(remote, 'python3 -c ' + shlex.quote(launcher)).decode().strip())

    def lock_probe(self, remote, folder):
        probe = folder / 'lock_probe'; probe.mkdir(exist_ok=True)
        script = str(self.repo / 'scripts/coding_coverage_v2_lock_probe.py')
        base = ['python3', script, '--lock', str(probe / 'shared.lock'), '--probe-id', folder.name]
        with (probe / 'holder.log').open('ab') as output:
            holder = subprocess.Popen(base + ['--mode', 'hold', '--ttl-seconds', '45',
                '--receipt', str(probe / 'held.json'), '--release-file', str(probe / 'release')],
                stdin=subprocess.DEVNULL, stdout=output, stderr=output)
        try:
            until = time.monotonic() + 10
            while not (probe / 'held.json').exists() and time.monotonic() < until:
                time.sleep(.2)
            if read(probe / 'held.json')['state'] != 'holding':
                raise ValueError('primary did not acquire shared lock')
            for expected, filename in [('blocked', 'remote_blocked.json'), ('acquired', 'remote_acquired.json')]:
                command = base + ['--mode', 'try', '--expect', expected, '--receipt', str(probe / filename)]
                dispatch.ssh(remote, shlex.join(command))
                receipt = read(probe / filename)
                if receipt.get('state') != expected or receipt.get('hostname') == read(probe / 'held.json')['hostname']:
                    raise ValueError('cross-pod shared lock control failed')
                if expected == 'blocked':
                    (probe / 'release').touch(); holder.wait(timeout=10)
            atomic_json(probe / 'verified.json', {'epoch': time.time(), 'cross_host_exclusion_verified': True,
                'receipts': {name: sha256_file(probe / name) for name in ['held.json', 'remote_blocked.json', 'remote_acquired.json']}})
        finally:
            (probe / 'release').touch()
            try: holder.wait(timeout=5)
            except subprocess.TimeoutExpired: holder.terminate(); holder.wait(timeout=5)

    def boot(self, folder):
        pod = read(folder / 'pod.json'); intent = read(folder / 'intent.json')
        lease_path, plan_path = folder / 'allocation_lease.json', folder / 'dispatch_plan.json'
        if not lease_path.exists():
            atomic_json(lease_path, helper_lease(self.primary, pod['id'], intent['gpu_count'], intent['epoch']))
        lease = read(lease_path)
        expected_plan = helper_plan(self.plan, str(lease_path.relative_to(self.repo)),
                                    intent['gpu_count'], sha256_file(lease_path))
        if not plan_path.exists():
            atomic_json(plan_path, expected_plan)
        elif read(plan_path) != expected_plan:
            raise ValueError('existing immutable helper plan differs from its lease/science')
        ready_end = min(time.time() + 900, self.primary['deadline_epoch'] - 300)
        while time.time() < ready_end:
            if self.finished():
                raise TimeoutError('no unfinished funded helper work remains')
            try:
                pod = dispatch.cli(self.args, 'pod', 'get', pod['id'])
                remote = self.remote_args(pod, lease_path)
                dispatch.verify_remote_identity(remote, lease)
                break
            except Exception:
                time.sleep(10)
        else:
            raise TimeoutError('helper SSH did not become ready within bounded staging window')
        prices = [float(pod[k]) for k in ('costPerHr', 'adjustedCostPerHr') if pod.get(k) is not None]
        if (pod.get('gpuCount') != lease['gpu_count'] or pod.get('networkVolumeId') != lease['network_volume_id']
                or not prices or not all(0 < price * 1.2 + .04 * lease['gpu_count'] <= lease['upper_hourly_usd'] for price in prices)):
            raise ValueError('live helper resources or quote differ from reservation')
        atomic_json(folder / 'ready_pod.json', dispatch.safe_pod(pod))
        guard = control_root(lease) / 'lease_guard'; guard.mkdir(parents=True, exist_ok=True)
        dispatch.ssh(remote, 'umask 077; cat > /root/.gearshift-runpod-key; chmod 600 /root/.gearshift-runpod-key', input=self.key.encode())
        if not (guard / 'status.json').exists():
            self.launch_remote(remote, ['python3', str(self.repo / 'gearshift/coding_coverage_v2_lease.py'),
                '--lease', str(lease_path), '--lease-sha256', sha256_file(lease_path),
                '--key-file', '/root/.gearshift-runpod-key'], guard / 'launcher.log')
        verified = dispatch.assert_guard_armed(remote, lease, wait_seconds=10)
        atomic_json(folder / 'guard_verified.json', verified)
        self.lock_probe(remote, folder)
        atomic_json(folder / 'bootstrap_launch_intent.json', {'epoch': time.time(), 'guard_verified': True,
            'cross_host_lock_verified': True, 'plan_sha256': sha256_file(plan_path)})
        pid = self.launch_remote(remote, ['python3', str(self.repo / 'scripts/coding_coverage_v2_bootstrap.py'),
            '--plan', str(plan_path.relative_to(self.repo)), '--lease', str(lease_path.relative_to(self.repo))],
            control_root(lease) / 'bootstrap_launcher.log')
        atomic_json(folder / 'bootstrap_dispatched.json', {'epoch': time.time(), 'pid': pid,
            'pod_id': lease['pod_id'], 'independent_of_controller': True})

    def preserve_failed_boot(self, folder, error):
        atomic_json(folder / 'boot_failed.json', {'epoch': time.time(), **redacted_error(error),
            'primary_unaffected': True})
        # A potentially started bootstrap is left to its independent lease guard.
        if (folder / 'bootstrap_launch_intent.json').exists():
            return
        pod = read(folder / 'pod.json')
        intent = read(folder / 'intent.json')
        lease = helper_lease(self.primary, pod['id'], intent['gpu_count'], intent['epoch'])
        live = dispatch.cli(self.args, 'pod', 'get', pod['id'])
        if live.get('id') != pod['id'] or live.get('networkVolumeId') != lease['network_volume_id']:
            raise ValueError('refusing cleanup of allocation without matching retained volume')
        atomic_json(folder / 'pre_execution_cleanup_proof.json', {'epoch': time.time(),
            'pod_id': pod['id'], 'network_volume_retained': True, 'bootstrap_never_dispatched': True,
            'durable_primary_plan_sha256': sha256_file(self.plan_path), 'failure_sha256': sha256_file(folder / 'boot_failed.json')})
        for _ in range(3):
            try:
                if delete_own_pod(lease, self.key):
                    atomic_json(folder / 'provider_absence_verified.json', {'epoch': time.time(), 'pod_id': pod['id']})
                    return
            except Exception as failure:
                self.event('helper_cleanup_retry', pod_id=pod['id'], **redacted_error(failure))
            time.sleep(5)

    def tick(self):
        pods = dispatch.cli(self.args, 'pod', 'list')
        if not self.reconcile(pods):
            self.event('awaiting_create_reconciliation'); return
        allocated = [path for path in self.attempts() if (path / 'pod.json').exists()]
        for path in allocated:
            if ((path / 'boot_failed.json').exists()
                    and not (path / 'bootstrap_launch_intent.json').exists()
                    and not (path / 'provider_absence_verified.json').exists()):
                self.preserve_failed_boot(path, RuntimeError('retrying preserved pre-execution cleanup'))
            if not any((path / name).exists() for name in ('bootstrap_launch_intent.json', 'boot_failed.json')):
                try: self.boot(path)
                except Exception as error: self.preserve_failed_boot(path, error)
        purchased = sum(read(path / 'intent.json')['gpu_count'] for path in allocated)
        if purchased >= 2 or len(allocated) >= 2 or self.finished():
            return
        available = [count for count in self.capacity() if count <= 2-purchased]
        if available:
            folder = self.acquire(available[0])
            if (folder / 'pod.json').exists():
                try: self.boot(folder)
                except Exception as error: self.preserve_failed_boot(folder, error)
        else:
            self.event('no_matching_helper_capacity')

    def run(self):
        with (self.directory / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while not self.finished():
                try: self.tick()
                except Exception as error: self.event('optional_acquisition_paused', **redacted_error(error))
                until = min(time.time() + 300, self.primary['deadline_epoch'] - 900)
                while time.time() < until and not self.finished():
                    time.sleep(min(5, max(0, until-time.time())))
            self.event('optional_acquisition_complete', primary_training_never_controlled=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--lease', required=True)
    parser.add_argument('--key-file', default='/root/.gearshift-runpod-key')
    parser.add_argument('--ssh-key')
    parser.add_argument('--cli', default='/root/.local/bin/runpodctl-2.14.0')
    parser.add_argument('--provider-config', default='/root/.runpod/config.toml')
    parser.add_argument('--region', default='AP-JP-1')
    Controller(parser.parse_args()).run()
