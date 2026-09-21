from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def save_checkpoint(path,obj):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp)
    tmp.replace(path)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)


def sync(device):
    if str(device).startswith('mps'):
        torch.mps.synchronize()
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize()


def timed(fn, device):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter() - start) * 1000


def memory():
    r = {'rss_gb': psutil.Process().memory_info().rss / 2**30}
    if torch.backends.mps.is_available():
        r.update(mps_allocated_gb=torch.mps.current_allocated_memory() / 2**30,
                 mps_driver_gb=torch.mps.driver_allocated_memory() / 2**30)
    return r


def environment():
    return dict(platform=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, transformers=transformers.__version__,
                cpu=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                if platform.system() == 'Darwin' else platform.processor(),
                cpu_count=psutil.cpu_count(), total_ram_gb=psutil.virtual_memory().total / 2**30,
                available_ram_gb=psutil.virtual_memory().available / 2**30,
                cuda_available=torch.cuda.is_available(), mps_built=torch.backends.mps.is_built(),
                mps_available=torch.backends.mps.is_available(), memory=memory())


class CacheExtractor:
    @staticmethod
    def tensors(cache, device=None, clone=False):
        pairs = tuple((layer.keys, layer.values) for layer in cache.layers)
        return tuple(tuple((x.to(device) if device else x).clone() if clone
                           else (x.to(device) if device else x) for x in pair) for pair in pairs)

    @staticmethod
    def flatten(x):
        # [batch, kv_heads, positions, head_dim] -> [batch * positions, features]
        return x.transpose(1, 2).reshape(-1, x.shape[1] * x.shape[3])

    @staticmethod
    def unflatten(x, heads, dim):
        return x.reshape(1, -1, heads, dim).transpose(1, 2).contiguous()


class CacheInjector:
    @staticmethod
    def splice_prefix(native_prefix, mapped_full_history):
        """Suffix keys retain the absolute RoPE positions from mapping the FULL history."""
        if len(native_prefix) != len(mapped_full_history):
            raise ValueError('Cache layer counts differ')
        prefix = native_prefix[0][0].shape[-2]
        total = mapped_full_history[0][0].shape[-2]
        if not 0 < prefix <= total:
            raise ValueError('Invalid hybrid prefix length')
        pairs = []
        for native, mapped in zip(native_prefix, mapped_full_history):
            pair = []
            for a,b in zip(native,mapped):
                if (a.shape[-2] != prefix or b.shape[-2] != total or
                    a.shape[:2]+a.shape[-1:] != b.shape[:2]+b.shape[-1:] or
                    a.dtype != b.dtype or a.device != b.device):
                    raise ValueError('Hybrid cache geometry, precision or device mismatch')
                pair.append(torch.cat((a, b[...,prefix:,:]),dim=-2))
            pairs.append(tuple(pair))
        return tuple(pairs)

    @staticmethod
    def create(pairs, clone=True):
        cache = DynamicCache()
        for i, (k, v) in enumerate(pairs):
            cache.update(k.clone() if clone else k, v.clone() if clone else v, i)
        return cache


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class ModelBackend:
    def __init__(self, name, device='mps', dtype='float16', attention='sdpa', revision=None):
        self.name, self.device, self.dtype = name, device, getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=self.dtype, attn_implementation=attention).eval().to(device)
        self.model.requires_grad_(False)
        self.config = self.model.config
        eos=self.model.generation_config.eos_token_id
        if eos is None: eos=self.config.eos_token_id
        self.eos=set([eos] if isinstance(eos,int) else eos)
        self.input_token_count = 0

    def ids(self, ids):
        if isinstance(ids, torch.Tensor):
            return ids.to(self.device).reshape(1, -1)
        return torch.tensor(ids, dtype=torch.long, device=self.device).reshape(1, -1)

    def forward(self, ids, cache=None, all_logits=False):
        ids = self.ids(ids)
        past = 0 if cache is None else cache.get_seq_length()
        self.input_token_count += ids.numel()
        positions = torch.arange(past, past + ids.shape[1], device=self.device)
        return self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                          attention_mask=torch.ones((1, past + ids.shape[1]), dtype=torch.long, device=self.device),
                          position_ids=positions.unsqueeze(0), cache_position=positions,
                          logits_to_keep=0 if all_logits else 1)

    @torch.inference_mode()
    def prefill(self, ids):
        return self.forward(ids)

    @torch.inference_mode()
    def generate_from(self, cache, logits, max_tokens, stop_ids=None):
        generated = []
        stop_ids = self.eos if stop_ids is None else set(stop_ids) | self.eos
        for _ in range(max_tokens):
            token = int(logits[0, -1].argmax())
            generated.append(token)
            # Include every emitted token in cache: essential for an exact source handoff.
            out = self.forward([token], cache)
            cache, logits = out.past_key_values, out.logits
            if token in stop_ids:
                break
        return generated, cache, logits

    def rope(self, k, positions=None, inverse=False):
        if self.config.model_type not in ('qwen3', 'llama'):
            raise NotImplementedError('Add a verified RoPE plugin for this model family')
        if getattr(self.config, 'rope_scaling', None):
            raise NotImplementedError('Dynamic/scaled RoPE requires a separately verified inverse')
        if positions is None:
            positions = torch.arange(k.shape[-2], device=k.device)
        cos, sin = self.model.model.rotary_emb(k.float(), positions.reshape(1, -1))
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        x = k.float()
        return x * cos + rotate_half(x) * (-sin if inverse else sin)

    @torch.inference_mode()
    def introspect(self):
        c = self.config
        out = self.prefill(self.tokenizer.encode('Cache inspection: one two three.'))
        cache = out.past_key_values
        vocab = self.tokenizer.get_vocab()
        return dict(name=self.name, revision=c._commit_hash, model_type=c.model_type,
                    parameter_count=sum(p.numel() for p in self.model.parameters()),
                    dtype=str(self.dtype), device=self.device,
                    vocab_size=c.vocab_size, tokenizer_size=len(self.tokenizer),
                    generation_config_eos_token_ids=self.model.generation_config.eos_token_id,
                    effective_eos_token_ids=sorted(self.eos),
                    vocab_sha256=hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest(),
                    backend_tokenizer_sha256=hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                    special_tokens=self.tokenizer.special_tokens_map,
                    chat_template_sha256=hashlib.sha256(str(self.tokenizer.chat_template).encode()).hexdigest(),
                    layers=c.num_hidden_layers, attention_heads=c.num_attention_heads,
                    kv_heads=c.num_key_value_heads, head_dim=c.head_dim,
                    hidden_size=c.hidden_size, rope_theta=c.rope_theta,
                    rope_scaling=c.rope_scaling, max_position_embeddings=c.max_position_embeddings,
                    cache_class=f'{type(cache).__module__}.{type(cache).__name__}',
                    cache_layer_class=type(cache.layers[0]).__name__,
                    shapes=[[list(k.shape), list(v.shape)] for k, v in CacheExtractor.tensors(cache)],
                    cache_position=cache.get_seq_length(), memory=memory())


def distributions(reference, candidate):
    # Do metrics in float64 on CPU, including all padded vocabulary entries.
    a, b = reference.detach().cpu().double().reshape(-1, reference.shape[-1]), candidate.detach().cpu().double().reshape(-1, candidate.shape[-1])
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    p, q = la.exp(), lb.exp()
    lm = torch.logaddexp(la, lb) - np.log(2)
    topa, topb = a.topk(5).indices, b.topk(5).indices
    return dict(max_logit_difference=float((a-b).abs().max()),
                mean_logit_difference=float((a-b).abs().mean()),
                kl=float((p*(la-lb)).sum(-1).mean().clamp_min(0)),
                js=float((0.5*(p*(la-lm)+q*(lb-lm))).sum(-1).mean().clamp_min(0)),
                top1_agreement=float((a.argmax(-1)==b.argmax(-1)).double().mean()),
                top5_overlap=float((topa.unsqueeze(-1)==topb.unsqueeze(-2)).any(-1).double().mean()),
                logit_cosine=float(torch.nn.functional.cosine_similarity(a,b).mean()))


def tensor_metrics(y, p):
    y, p = y.float(), p.float()
    error = y-p
    mse = error.square().mean()
    variance = (y-y.mean(0)).square().mean().clamp_min(1e-12)
    return dict(mse=float(mse), r2=float(1-mse/variance),
                explained_variance=float(1-(error-error.mean(0)).square().mean()/variance),
                cosine=float(torch.nn.functional.cosine_similarity(y,p).mean()))
