"""Matched optimization with one immutable full cache pair shared between arms.

Both arms use the unchanged differentiable continuation and KL objective. Only
GPU cache preparation is shared; no optimizer state or mapper gradients are.
"""
import gc, time
import torch
from .coding_coverage import ARMS, grouped, check_pair
from .coding_training import TrainingRuntime
from .coding_inference import sync
from .core import CacheExtractor


class CoverageRuntime(TrainingRuntime):
    def __init__(self, source, receiver, mapper, guard, telemetry):
        super().__init__(source, receiver, mapper, None, guard)
        self.telemetry = telemetry

    @torch.no_grad()
    def pair(self, obj, device=None):
        """Identical full512-token prefill; no disk cache or context cropping."""
        self.guard(); ids = obj['source_history']['prefix_ids']
        self.telemetry.sample(stage='shared_pair_reconstruction', task_id=obj['task_id'], sequence_length=len(ids))
        so = self.source.prefill_chunked(ids)
        sp = CacheExtractor.tensors(so.past_key_values); del so
        to = self.receiver.prefill_chunked(ids)
        tp = CacheExtractor.tensors(to.past_key_values); del to
        self.telemetry.sample(stage='shared_pair_ready')
        return sp, tp


def paired_update(items, lookup, runtimes, optimizers, arm_order):
    check_pair(items['FIXED'], items['ROTATING'], lookup)
    groups = {a: grouped(items[a]) for a in ARMS}
    for a in ARMS:
        optimizers[a].zero_grad(set_to_none=True)
        runtimes[a].cache_gradient_checks = []
    before = time.monotonic(); prep = 0.; losses = {a: 0. for a in ARMS}
    elapsed = {a: 0. for a in ARMS}; details = []
    for tid in groups['FIXED']:
        obj = lookup[tid]; sync(); start = time.monotonic()
        sp, tp = runtimes['FIXED'].pair(obj); sync(); pair_seconds = time.monotonic() - start
        prep += pair_seconds
        # Tensor version counters check that neither manual continuation changes
        # the shared source/native prefix. No gradients flow into these tensors.
        versions = [x._version for p in (*sp, *tp) for x in p]
        for arm in arm_order:
            runtime = runtimes[arm]; runtime.guard(); sync(); start = time.monotonic()
            runtime.telemetry.sample(stage='coverage_gradient', task_id=tid, arm=arm,
                sequence_length=len(obj['source_history']['prefix_ids'])+max(groups[arm][tid])+1)
            values = runtime.kl(obj, groups[arm][tid], 'natural_handoff_boundary', sp, tp)
            loss = values.sum() / 32
            losses[arm] += float(loss.detach()); loss.backward(); sync()
            elapsed[arm] += time.monotonic() - start
            del loss, values
            if versions != [x._version for p in (*sp, *tp) for x in p]:
                raise RuntimeError('Shared historical cache mutated by training')
        details.append({'task_id': tid, 'pair_preparation_seconds': pair_seconds,
                        'predictions_per_arm': len(groups['FIXED'][tid]), 'cache_versions_unchanged': True})
        del sp, tp; gc.collect()
    result = {}
    # Check BOTH arms before applying either optimizer. Checkpoint transactions
    # below are paired, so a later engineering interruption cannot compare an
    # update from one arm with an earlier update of the other.
    for arm in ARMS:
        runtime = runtimes[arm]
        checks = [r for part in runtime.cache_gradient_checks for r in part]
        if not checks or not all(r['finite'] and r['absolute_sum'] > 0 for r in checks):
            raise RuntimeError('Missing, zero or nonfinite cache gradient')
        if any(p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum() > 0 for p in runtime.mapper.parameters()):
            raise RuntimeError('Missing, zero or nonfinite mapper gradient')
        if any(p.grad is not None for b in [runtime.source, runtime.receiver] for p in b.model.parameters()):
            raise RuntimeError('Frozen language model received gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(runtime.mapper.parameters(), 1, error_if_nonfinite=True))
        result[arm] = {'kl': losses[arm], 'gradient_norm_before_clipping': norm,
                      'gradient_predictions': 32, 'cache_gradient_tensors_checked': len(checks),
                      'cache_gradients_finite_nonzero': True,
                      'continuation_forward_backward_seconds': elapsed[arm]}
    for arm in arm_order:
        runtimes[arm].guard(); sync(); start = time.monotonic()
        optimizers[arm].step(); optimizers[arm].zero_grad(set_to_none=True); sync()
        result[arm]['optimizer_seconds'] = time.monotonic() - start
    return {'arms': result, 'arm_order': list(arm_order), 'shared_cache_preparation_seconds': prep,
            'tasks': details, 'wall_seconds': time.monotonic() - before,
            'normalization_per_arm': 32, 'execution': 'Shared immutable full caches; independent gradients and optimizers.'}


@torch.no_grad()
def validation(runtime, histories, panels, publish=lambda **kw: None):
    rows = []; sync(); before = time.monotonic()
    for i, obj in enumerate(histories):
        runtime.guard(); tid = obj['task_id']; publish(stage='coverage_validation', task_id=tid, completed_tasks=i)
        sp, tp = runtime.pair(obj)
        row = {'task_id': tid}
        for name in ['legacy', 'broad']:
            positions = panels[tid][name]
            values = runtime.kl(obj, positions, 'natural_handoff_boundary', sp, tp, require_grad=False)
            row[name] = {'positions': positions, 'per_position_kl': values.flatten().cpu().tolist(), 'mean_kl': float(values.mean())}
            del values
        rows.append(row); del sp, tp; gc.collect()
    sync()
    return {'rows': rows, 'legacy_mean_task_kl': sum(r['legacy']['mean_kl'] for r in rows)/len(rows),
            'broad_mean_task_kl': sum(r['broad']['mean_kl'] for r in rows)/len(rows),
            'wall_seconds': time.monotonic()-before, 'aggregation': 'Equal task means; separate panels, not interchangeable magnitudes.'}
