"""Small ORACLE-ASSISTED current-query sparse/repair diagnostic primitives.

No model loading, generation, scheduler, training, or scorer lives here. These
functions DO NOT establish a CUDA/model numerical gate. All tensors retain their
original post-RoPE coordinates. Sparse access is a dense masked intervention,
not a sparse kernel or a measured latency/memory saving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from typing import Literal

import torch


@dataclass(frozen=True)
class HistoryLayout:
    prompt_length: int
    history_length: int
    page_size: int = 16

    def __post_init__(self):
        if not 0 < self.prompt_length <= self.history_length:
            raise ValueError("Require 0 < native prompt length <= historical length")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")

    @property
    def reasoning_length(self) -> int:
        return self.history_length - self.prompt_length

    def count(self, fraction: float) -> int:
        """ceil(p * reasoning_length); zero stays zero; no minimum-one surprise."""
        p = Decimal(str(fraction))
        if not p.is_finite() or not Decimal(0) <= p <= Decimal(1):
            raise ValueError("fraction must be finite and within [0, 1]")
        return int((p * self.reasoning_length).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class SelectionIdentity:
    task_id: str
    draw_seed: int
    layer: int
    step: int

    def seed(self, kv_head: int) -> int:
        if self.layer < 0 or self.step < 0 or kv_head < 0:
            raise ValueError("Negative layer, step, or KV-head identity")
        payload = json.dumps(["sparse-repair-01-random-v1", self.task_id,
                              self.draw_seed, self.layer, self.step, kv_head],
                             separators=(",", ":"), ensure_ascii=True)
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63)


def _check_tensor(t: torch.Tensor, name: str):
    if t.ndim != 4 or t.shape[0] != 1 or not t.is_floating_point():
        raise ValueError(f"{name} must be floating [1, heads, positions, dim]")


def _check_pair(pair: tuple[torch.Tensor, torch.Tensor], name: str):
    k, v = pair
    _check_tensor(k, name + " keys")
    _check_tensor(v, name + " values")
    if k.shape != v.shape or k.device != v.device or k.dtype != v.dtype:
        raise ValueError(name + " K/V geometry, dtype, or device differs")


@torch.no_grad()
def native_reasoning_mass(query: torch.Tensor, native_history_keys: torch.Tensor,
                          live_keys: torch.Tensor, layout: HistoryLayout, *,
                          scaling: float, query_position: int,
                          attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return [KV-head, reasoning-position] sums of current-query probabilities.

    Native historical keys include the original prompt/reasoning only. The tail
    comes solely from this run's live cache, including the current token. Native
    scoring normalizes across ALL available keys before GQA-head aggregation;
    normalizing only the reasoning suffix would alter aggregation across heads.
    At most [1, Q-heads, 1, context] scores exist, never an L x L matrix.
    """
    for t, n in [(query, "query"), (native_history_keys, "native"), (live_keys, "live")]:
        _check_tensor(t, n)
    if query.shape[-2] != 1:
        raise ValueError("Only single current-query decode operations are supported")
    if native_history_keys.shape[-2] != layout.history_length:
        raise ValueError("Native history must contain exactly the frozen prefix")
    if live_keys.shape[-2] <= layout.history_length or query_position != live_keys.shape[-2] - 1:
        raise ValueError("Live cache must include current token at its original absolute position")
    if native_history_keys.shape[:2] + native_history_keys.shape[-1:] != live_keys.shape[:2] + live_keys.shape[-1:]:
        raise ValueError("Native/live geometry differs")
    if query.shape[-1] != live_keys.shape[-1] or query.shape[1] % live_keys.shape[1]:
        raise ValueError("Invalid GQA geometry")
    if len({t.device for t in (query, native_history_keys, live_keys)}) != 1:
        raise ValueError("Mixed devices")
    if len({t.dtype for t in (query, native_history_keys, live_keys)}) != 1:
        raise ValueError("Mixed precision")
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("Invalid model scaling")
    groups = query.shape[1] // live_keys.shape[1]
    # This is deliberately a dense oracle scan. It is not a native answer prefix.
    keys = torch.cat((native_history_keys, live_keys[..., layout.history_length:, :]), dim=-2)
    q = query.float().reshape(1, live_keys.shape[1], groups, 1, query.shape[-1])
    scores = torch.matmul(q, keys.float().unsqueeze(2).transpose(-1, -2)).flatten(1, 2) * scaling
    if attention_mask is not None:
        if attention_mask.device != query.device or attention_mask.ndim != 4:
            raise ValueError("Mask must be same-device [1, 1 or Q-heads, 1, context]")
        if attention_mask.shape[0] != 1 or attention_mask.shape[1] not in (1, query.shape[1]) or attention_mask.shape[-2:] != (1, live_keys.shape[-2]):
            raise ValueError("Mask geometry differs")
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, -torch.inf)
        elif attention_mask.is_floating_point():
            scores = scores + attention_mask.float()
        else:
            raise ValueError("Mask must be bool-allowed or floating additive")
    probability = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probability).all():
        raise ValueError("Nonfinite/all-masked native scoring")
    return probability.reshape(1, live_keys.shape[1], groups, 1, live_keys.shape[-2]).sum(2)[0, :, 0, layout.prompt_length:layout.history_length]


@torch.no_grad()
def select_indices(mass: torch.Tensor, layout: HistoryLayout, fraction: float, *,
                   method: Literal["native_mass", "random", "recent"] = "native_mass",
                   identity: SelectionIdentity | None = None) -> torch.Tensor:
    """Sorted absolute indices per KV group; complete K/V pairs share one mask.

    Ranking ties use the smaller original position. Random uses a CPU generator
    per stable task/draw/layer/step/head identity under the pinned Torch version;
    worker identity, order, condition name and global RNG state are excluded.
    """
    if mass.ndim != 2 or mass.shape[0] == 0 or mass.shape[1] != layout.reasoning_length:
        raise ValueError("Mass must be [KV-head, historical reasoning position]")
    if not mass.is_floating_point() or not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("Invalid native probability mass")
    count = layout.count(fraction)
    if method == "native_mass":
        selected = torch.argsort(mass, dim=-1, descending=True, stable=True)[:, :count]
    elif method == "recent":
        selected = torch.arange(layout.reasoning_length - count, layout.reasoning_length,
                                device=mass.device).expand(mass.shape[0], -1)
    elif method == "random":
        if identity is None:
            raise ValueError("Random selection requires stable identity")
        selected = torch.stack([torch.randperm(layout.reasoning_length,
                              generator=torch.Generator(device="cpu").manual_seed(identity.seed(h)))[:count]
                                for h in range(mass.shape[0])]).to(mass.device)
    else:
        raise ValueError("Unknown selection rule")
    return selected.sort(dim=-1).values + layout.prompt_length


def _check_indices(indices: torch.Tensor, layout: HistoryLayout, heads: int, device: torch.device):
    if indices.ndim != 2 or indices.shape[0] != heads or indices.dtype != torch.long or indices.device != device:
        raise ValueError("Indices must be same-device int64 [KV-head, budget]")
    if indices.numel() and (indices.min() < layout.prompt_length or indices.max() >= layout.history_length):
        raise ValueError("Selection must lie wholly inside historical reasoning")
    if indices.shape[1] > 1 and not (indices[:, 1:] > indices[:, :-1]).all():
        raise ValueError("Indices must be unique and increasing per KV head")


@dataclass
class Intervention:
    keys: torch.Tensor
    values: torch.Tensor
    # Per-KV-head logical access mask; expand with repeat_interleave for Q heads.
    allowed: torch.Tensor
    selected_indices: torch.Tensor
    mode: str


@torch.no_grad()
def intervene(live_pair: tuple[torch.Tensor, torch.Tensor],
              native_history: tuple[torch.Tensor, torch.Tensor],
              mapped_history: tuple[torch.Tensor, torch.Tensor],
              layout: HistoryLayout, indices: torch.Tensor,
              mode: Literal["N", "M", "R"]) -> Intervention:
    """Return temporary dense K/V + mask without changing ANY backing tensor.

    Prompt and own answer/current state always come from live_pair. R starts
    from immutable mapped history on EVERY call, then replaces only this call's
    selected pairs. No selected repair is written into the persistent cache.
    N/M retain only selected reasoning for reads; their physical buffer remains
    dense. Full N/R refer to a matched native/native splice, not necessarily a
    whole-prefix prefill with identical BF16 execution shape.
    """
    for pair, name in [(live_pair, "live"), (native_history, "native"), (mapped_history, "mapped")]:
        _check_pair(pair, name)
    live = live_pair[0]
    if live.shape[-2] <= layout.history_length:
        raise ValueError("Live state must include current token")
    for historical in [native_history[0], mapped_history[0]]:
        if historical.shape[-2] != layout.history_length or historical.shape[:2] + historical.shape[-1:] != live.shape[:2] + live.shape[-1:] or historical.dtype != live.dtype or historical.device != live.device:
            raise ValueError("History length/geometry/dtype/device mismatch")
    _check_indices(indices, layout, live.shape[1], live.device)
    if mode not in ("N", "M", "R"):
        raise ValueError("Unknown intervention")
    source = native_history if mode == "N" else mapped_history
    result = []
    gather = indices[None, :, :, None].expand(1, -1, -1, live.shape[-1])
    for own, base, native in zip(live_pair, source, native_history):
        combined = torch.cat((own[..., :layout.prompt_length, :],
                              base[..., layout.prompt_length:, :],
                              own[..., layout.history_length:, :]), dim=-2)
        if mode == "R":
            combined.scatter_(2, gather, native.gather(2, gather))
        result.append(combined)
    allowed = torch.ones((1, live.shape[1], 1, live.shape[-2]), dtype=torch.bool, device=live.device)
    if mode in ("N", "M"):
        allowed[..., layout.prompt_length:layout.history_length] = False
        allowed.scatter_(-1, indices[None, :, None, :], True)
    return Intervention(*result, allowed, indices.clone(), mode)


@torch.no_grad()
def selected_accounting(indices: torch.Tensor, layout: HistoryLayout, *,
                        heads: int, head_dim: int, element_size: int) -> dict:
    """Logical selection plus explicitly ESTIMATED pages for two layouts.

    Group-page estimate: independent pages per KV head. Shared-page estimate:
    a touched token page contains all KV heads. Neither is measured GPU traffic
    or actual allocator pages. Last page is charged as one padded full page.
    """
    _check_indices(indices, layout, heads, indices.device)
    if head_dim <= 0 or element_size <= 0:
        raise ValueError("Invalid pair storage geometry")
    pair_bytes = 2 * head_dim * element_size
    pages = indices.div(layout.page_size, rounding_mode="floor")
    per_head_pages = sum(int(torch.unique(row).numel()) for row in pages)
    shared_pages = int(torch.unique(pages).numel())
    return {"selected_unique_kv_pairs": indices.numel(),
            "logical_selected_bytes": indices.numel() * pair_bytes,
            "selected_index_bytes": indices.numel() * indices.element_size(),
            "per_head_page_count_estimated": per_head_pages,
            "per_head_page_bytes_estimated": per_head_pages * layout.page_size * pair_bytes,
            "all_heads_shared_page_count_estimated": shared_pages,
            "all_heads_shared_page_bytes_estimated": shared_pages * layout.page_size * pair_bytes * heads,
            "page_size_positions_assumed": layout.page_size,
            "physical_page_or_traffic_measurement": False}


@dataclass
class WorkingSetTrace:
    """Bound selected-index records, but track full-answer cumulative pair union.

    One instance per task/draw/condition/layer. No union across heads or layers.
    The trace records only configured current-query step indices; consumers may
    not call missing local/model measurements 'zero'. update() introduces CPU
    transfer/synchronization, which must be charged in diagnostic wall time.
    """
    layout: HistoryLayout
    heads: int
    head_dim: int
    element_size: int
    trace_steps: tuple[int, ...] = (0, 1, 16, 64, 256, 1024, 4095)
    trace_heads: tuple[int, ...] | None = None
    seen: torch.Tensor = field(init=False, repr=False)
    records: list[dict] = field(default_factory=list, init=False)
    calls: int = field(default=0, init=False)
    last_step: int = field(default=-1, init=False)
    peak_instantaneous_pairs: int = field(default=0, init=False)

    def __post_init__(self):
        if self.heads <= 0 or self.head_dim <= 0 or self.element_size <= 0:
            raise ValueError("Invalid cache geometry")
        if self.trace_heads is not None and (len(set(self.trace_heads)) != len(self.trace_heads) or any(h < 0 or h >= self.heads for h in self.trace_heads)):
            raise ValueError("Invalid explicit-trace KV-head subset")
        self.seen = torch.zeros((self.heads, self.layout.reasoning_length), dtype=torch.bool)

    @torch.no_grad()
    def update(self, step: int, indices: torch.Tensor, mass: torch.Tensor | None = None):
        if step <= self.last_step:
            raise ValueError("Step must increase; resume must restore the cumulative trace")
        _check_indices(indices, self.layout, self.heads, indices.device)
        local = indices.detach().cpu()
        self.seen.scatter_(1, local - self.layout.prompt_length, True)
        self.calls += 1
        self.last_step = step
        self.peak_instantaneous_pairs = max(self.peak_instantaneous_pairs, local.numel())
        if step in self.trace_steps:
            traced_heads = list(range(self.heads)) if self.trace_heads is None else list(self.trace_heads)
            record = {"step": step, "traced_kv_heads": traced_heads,
                      "selected_indices_absolute": local[traced_heads].tolist(),
                      "accounting_scope": "all KV heads, not merely explicitly traced heads",
                      **selected_accounting(local, self.layout, heads=self.heads,
                          head_dim=self.head_dim, element_size=self.element_size)}
            if mass is not None:
                if mass.shape != self.seen.shape or not torch.isfinite(mass).all():
                    raise ValueError("Invalid mass trace")
                selected_mass = mass.gather(1, indices - self.layout.prompt_length).sum(1)
                total_mass = mass.sum(1)
                record["selected_attention_mass_sum_over_query_heads_per_group"] = selected_mass.cpu().tolist()
                record["all_reasoning_attention_mass_sum_over_query_heads_per_group"] = total_mass.cpu().tolist()
            self.records.append(record)

    def summary(self) -> dict:
        # Per-head cumulative counts may differ, unlike each fixed-step budget.
        pairs = int(self.seen.sum())
        per_head_pages = 0
        shared = set()
        for row in self.seen:
            pages = torch.unique((row.nonzero().flatten() + self.layout.prompt_length) // self.layout.page_size)
            per_head_pages += pages.numel()
            shared.update(pages.tolist())
        bytes_per_pair = 2 * self.head_dim * self.element_size
        return {"attention_operations": self.calls, "last_step": self.last_step,
                "peak_instantaneous_selected_pairs": self.peak_instantaneous_pairs,
                "cumulative_unique_kv_pairs": pairs,
                "cumulative_pairs_by_kv_head": self.seen.sum(1).tolist(),
                "cumulative_reasoning_pair_fraction": pairs / self.seen.numel() if self.seen.numel() else 0.0,
                "cumulative_logical_selected_bytes": pairs * bytes_per_pair,
                "cumulative_per_head_page_bytes_estimated": per_head_pages * self.layout.page_size * bytes_per_pair,
                "cumulative_all_heads_shared_page_bytes_estimated": len(shared) * self.layout.page_size * bytes_per_pair * self.heads,
                "cumulative_union_metadata_bytes": self.seen.numel() * self.seen.element_size(),
                "bounded_index_trace_records": len(self.records),
                "physical_page_or_traffic_measurement": False}

    def state_dict(self) -> dict:
        return {"layout": vars(self.layout), "heads": self.heads,
                "head_dim": self.head_dim, "element_size": self.element_size,
                "trace_steps": list(self.trace_steps),
                "trace_heads": list(self.trace_heads) if self.trace_heads is not None else None, "seen": self.seen.tolist(),
                "records": self.records, "calls": self.calls, "last_step": self.last_step,
                "peak_instantaneous_pairs": self.peak_instantaneous_pairs}

    @classmethod
    def from_state_dict(cls, state: dict) -> "WorkingSetTrace":
        result = cls(HistoryLayout(**state["layout"]), state["heads"], state["head_dim"],
                     state["element_size"], tuple(state["trace_steps"]),
                     tuple(state["trace_heads"]) if state.get("trace_heads") is not None else None)
        result.seen = torch.tensor(state["seen"], dtype=torch.bool).reshape_as(result.seen)
        result.records = state["records"]
        result.calls = state["calls"]
        result.last_step = state["last_step"]
        result.peak_instantaneous_pairs = state["peak_instantaneous_pairs"]
        return result


class SparseRepairController:
    """Minimal Transformers-4.57.6 Qwen3 attention-interface context manager.

    Install only on one receiver model in the owning process. It sees Qwen3's
    already-normalized, already-RoPE'd current query and cache-updated live K/V;
    it never replaces Qwen3 forward, cache.update, rotary code, or generation.
    The original SDPA callable implements both the disabled and modified paths.

    `probe_callback(layer, step, query, (live_k, live_v), mask, scaling)` is an
    optional bounded diagnostic callback. It MUST NOT mutate these tensors or
    feed a reference/future answer into generation. `profile=True` explicitly
    synchronizes device stage timers; use for calibration, not a speedup claim.
    """
    def __init__(self, model, native_history, mapped_history, layout: HistoryLayout, *,
                 mode="disabled", fraction=.1, selector="native_mass", task_id="",
                 draw_seed=0, trace_enabled=True, profile=False,
                 trace_steps=(0, 1, 16, 64, 256, 1024, 4095),
                 trace_layers=(0, 17, 35), trace_heads=(0, 7), probe_callback=None):
        self.model = model
        self.native_history = tuple(tuple(pair) for pair in native_history)
        self.mapped_history = tuple(tuple(pair) for pair in mapped_history)
        self.layout = layout
        if len(self.native_history) != len(self.mapped_history) or not self.native_history:
            raise ValueError("Frozen native/mapped layer counts differ or are empty")
        if getattr(model.config, "model_type", None) != "qwen3":
            raise ValueError("Only the inspected Qwen3 attention interface is supported")
        if len(self.native_history) != model.config.num_hidden_layers:
            raise ValueError("Frozen cache/model layer counts differ")
        for native, mapped in zip(self.native_history, self.mapped_history):
            for pair, name in [(native, "native"), (mapped, "mapped")]:
                _check_pair(pair, name)
                if pair[0].shape[-2] != layout.history_length:
                    raise ValueError("Frozen history length differs")
            if native[0].shape != mapped[0].shape or native[0].dtype != mapped[0].dtype or native[0].device != mapped[0].device:
                raise ValueError("Frozen cache geometry/dtype/device differs")
        self.trace_enabled = trace_enabled
        self.profile = profile
        self.trace_steps = tuple(trace_steps)
        self.trace_layers = tuple(trace_layers) if trace_layers is not None else None
        self.trace_heads = tuple(trace_heads) if trace_heads is not None else None
        self.probe_callback = probe_callback
        self._installed = False
        self._name = "gearshift_sparse_repair_01_" + str(id(self))
        self.set_condition(mode, fraction=fraction, selector=selector,
                           task_id=task_id, draw_seed=draw_seed)

    def set_condition(self, mode, *, fraction=None, selector=None,
                      task_id=None, draw_seed=None, reset=True):
        if mode not in ("disabled", "N", "M", "R"):
            raise ValueError("Unknown controller mode")
        self.mode = mode
        if fraction is not None:
            self.layout.count(fraction)
            self.fraction = fraction
        if selector is not None:
            if selector not in ("native_mass", "random", "recent"):
                raise ValueError("Unknown selector")
            self.selector = selector
        if task_id is not None:
            self.task_id = task_id
        if draw_seed is not None:
            self.draw_seed = draw_seed
        if reset:
            self.traces = {}
            self._last_step = {}
            self._calls = {}
            self._stage_seconds = {k: 0.0 for k in ("native_key_scan", "selection", "replacement_and_mask",
                                                    "attention", "trace_and_accounting", "probe_callback")}
            self._peak_temporary = {}

    def __enter__(self):
        import transformers
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        if transformers.__version__ != "4.57.6":
            raise ValueError("Hook is pinned to inspected Transformers 4.57.6")
        if self._installed:
            raise RuntimeError("Controller is already installed")
        if self.model.config._attn_implementation != "sdpa":
            raise ValueError("Receiver must use the original deployed sdpa path before hooking")
        self._original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self._old_implementation = self.model.config._attn_implementation
        # Matching mask registration preserves model-level SDPA causal-mask rules.
        ALL_ATTENTION_FUNCTIONS[self._name] = self._attention
        ALL_MASK_ATTENTION_FUNCTIONS[self._name] = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
        self.model.config._attn_implementation = self._name
        self._installed = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        self.model.config._attn_implementation = self._old_implementation
        ALL_ATTENTION_FUNCTIONS.pop(self._name, None)
        ALL_MASK_ATTENTION_FUNCTIONS.pop(self._name, None)
        self._installed = False
        return False

    def _timed(self, name, device, fn):
        if not self.profile:
            return fn()
        import time
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._stage_seconds[name] += time.perf_counter() - start
        return result

    @torch.no_grad()
    def _attention(self, module, query, key, value, attention_mask,
                   dropout=0.0, scaling=None, **kwargs):
        if module.training or dropout != 0.0:
            raise ValueError("Diagnostic requires frozen eval attention with zero dropout")
        layer = module.layer_idx
        if getattr(module, "sliding_window", None) is not None:
            raise ValueError("Sliding-window attention is outside this frozen diagnostic")
        if query.shape[-2] != 1:
            raise ValueError("Install the controller for single-token decode only, after prefill")
        if key.shape[-2] <= self.layout.history_length:
            raise ValueError("Missing current-token state or cache/position drift")
        step = key.shape[-2] - self.layout.history_length - 1
        if layer in self._last_step and step != self._last_step[layer] + 1:
            raise ValueError("Repeated/skipped decode operation; reset condition for a new prefix")
        self._last_step[layer] = step
        self._calls[layer] = self._calls.get(layer, 0) + 1
        scaling = module.scaling if scaling is None else scaling
        if self.probe_callback is not None:
            self._timed("probe_callback", query.device, lambda: self.probe_callback(
                layer, step, query, (key, value), attention_mask, scaling))
        def sdpa(k, v, mask):
            return self._original_sdpa(module, query, k, v, mask, dropout=dropout,
                                       scaling=scaling, **kwargs)
        if self.mode == "disabled":
            return self._timed("attention", query.device, lambda: sdpa(key, value, attention_mask))
        if self.selector == "native_mass":
            mass = self._timed("native_key_scan", query.device, lambda: native_reasoning_mass(
                query, self.native_history[layer][0], key, self.layout,
                scaling=scaling, query_position=key.shape[-2] - 1,
                attention_mask=attention_mask))
        else:
            # Cheap random/recency controls do not require oracle key scans.
            mass = torch.zeros((key.shape[1], self.layout.reasoning_length), device=key.device)
        indices = self._timed("selection", query.device, lambda: select_indices(
            mass, self.layout, self.fraction, method=self.selector,
            identity=SelectionIdentity(self.task_id, self.draw_seed, layer, step)))
        def repair_and_mask():
            changed = intervene((key, value), self.native_history[layer], self.mapped_history[layer],
                                self.layout, indices, self.mode)
            mask = attention_mask
            # Preserve exactly the dense SDPA execution path for R and full N/M.
            if self.mode in ("N", "M") and indices.shape[1] < self.layout.reasoning_length:
                allowed = changed.allowed.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
                if mask is None:
                    mask = allowed
                elif mask.dtype == torch.bool:
                    mask = mask & allowed
                else:
                    mask = mask.masked_fill(~allowed, -torch.inf)
            return changed, mask
        changed, mask = self._timed("replacement_and_mask", query.device, repair_and_mask)
        result = self._timed("attention", query.device, lambda: sdpa(changed.keys, changed.values, mask))
        def record():
            if layer not in self.traces:
                self.traces[layer] = WorkingSetTrace(self.layout, key.shape[1], key.shape[-1],
                    key.element_size(), self.trace_steps if self.trace_enabled and (self.trace_layers is None or layer in self.trace_layers) else (),
                    tuple(h for h in self.trace_heads if h < key.shape[1]) if self.trace_heads is not None else None)
            self.traces[layer].update(step, indices, mass if self.trace_enabled and self.selector == "native_mass" else None)
            sizes = {"dense_temporary_kv_bytes_exact": changed.keys.numel() * changed.keys.element_size() * 2,
                     "kv_group_allowed_mask_bytes_exact": changed.allowed.numel() * changed.allowed.element_size(),
                     "selected_indices_bytes_exact": changed.selected_indices.numel() * changed.selected_indices.element_size(),
                     "native_probability_mass_bytes_exact": mass.numel() * mass.element_size(),
                     "live_full_context_kv_bytes_exact": key.numel() * key.element_size() * 2}
            previous = self._peak_temporary.setdefault(layer, {})
            for name, size in sizes.items():
                previous[name] = max(previous.get(name, 0), size)
        self._timed("trace_and_accounting", query.device, record)
        return result

    def trace_records(self):
        return {str(layer): trace.records for layer, trace in sorted(self.traces.items())}

    def summary(self):
        layers = {str(layer): trace.summary() for layer, trace in sorted(self.traces.items())}
        pair_sum = sum(row["cumulative_unique_kv_pairs"] for row in layers.values())
        return {"mode": self.mode, "fraction": self.fraction, "selector": self.selector,
                "task_id": self.task_id, "draw_seed": self.draw_seed,
                "explicit_index_trace_layers": self.trace_layers,
                "explicit_index_trace_kv_heads": self.trace_heads,
                "explicit_index_trace_steps": self.trace_steps,
                "cumulative_accounting_scope": "every layer and every KV head at every operation",
                "oracle_assisted": self.mode != "disabled" and self.selector == "native_mass",
                "implementation": "temporary dense K/V; masked-dense SDPA for N/M; no persistent cache changes",
                "attention_calls_by_layer": {str(k): v for k, v in sorted(self._calls.items())},
                "layers": layers,
                "cumulative_unique_pairs_all_layers": pair_sum,
                "cumulative_logical_selected_bytes_all_layers": sum(r["cumulative_logical_selected_bytes"] for r in layers.values()),
                "frozen_native_history_bytes_exact": sum(x.numel()*x.element_size() for p in self.native_history for x in p),
                "frozen_mapped_history_bytes_exact": sum(x.numel()*x.element_size() for p in self.mapped_history for x in p),
                "per_layer_peak_buffer_sizes": self._peak_temporary,
                "buffer_accounting_limitations": "Logical tensor sizes, not allocator peaks or memory traffic; score/softmax scratch, original source cache, model weights, and allocator overhead must be recorded separately. Per-layer maxima must not be called concurrent temporary usage.",
                "stage_seconds": self._stage_seconds if self.profile else None,
                "stage_timing_profile_enabled": self.profile,
                "timing_limitations": "Profile timers synchronize every stage; trace accounting includes CPU transfers even when selected-index tracing is disabled. Measure actual whole-answer wall time separately."}
