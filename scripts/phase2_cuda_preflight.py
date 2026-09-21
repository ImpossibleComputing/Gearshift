#!/usr/bin/env python3
"""Real CUDA boundary gradients and emitted-EOS cache accounting before generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.phase2_io import read, write, artifact
from gearshift.phase2_inference import backends, adapter_from, bridge_ids
from gearshift.phase2_training import frozen_pair, sparse_case, prediction, make_trainable


def main():
    cfg = read('configs/phase2_cuda.json')
    assert cfg['device'] == 'cuda' and torch.cuda.is_available()
    source, target = backends(cfg)
    mapper = 'results/followup_v1/selected_mapper.pt'
    before = artifact(mapper)
    history = source.tokenizer.encode('A sealed box contains three blue counters and two red counters.')
    answer = target.tokenizer.encode('There are five counters in the sealed box. ' * 12)
    obj = dict(trajectory=dict(prompt_ids=history, source_reasoning_ids=[]),
        task=dict(family='writing', task_id='cuda_infrastructure_synthetic'),
        teachers=dict(boundary=dict(bridge_ids=bridge_ids(target.tokenizer, 'Describe the box.'), answer_ids=answer)))
    rows = []
    for anchor in [0, 32]:
        case = sparse_case(obj, 'boundary', anchor)
        sp, tp = frozen_pair(source, target, history)
        with torch.no_grad():
            reference = prediction(target, tp, case)
            whole = target.forward(history + case['prefix_ids'] + case['input_ids'], all_logits=True).logits.float().log_softmax(-1)
        error = float((reference - whole[:, -case['predictions']:]).abs().max())
        # FP16 split and whole-prefill kernels can round differently.
        assert torch.allclose(reference, whole[:, -case['predictions']:], atol=.05, rtol=.005), error
        adapter = adapter_from(mapper, source, target)
        params = make_trainable(adapter)
        mapped = adapter.transform(sp, 'functional')
        student = prediction(target, mapped, case)
        loss = (reference.exp() * (reference - student)).sum(-1).mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert any(p.grad.abs().sum() > 0 for p in params)
        assert all(p.grad is None for b in [source, target] for p in b.model.parameters())
        rows.append(dict(anchor=anchor, alignment_max_logprob_error=error, loss=float(loss.detach()), finite_mapper_gradients=True, frozen_lm_gradients=True))
        del adapter, params, mapped, student, loss
    eos_rows = []
    with torch.inference_mode():
        for name, b in [('source', source), ('target', target)]:
            for eos in sorted(b.eos):
                out = b.prefill(history)
                logits = torch.full_like(out.logits, -10000)
                logits[0, -1, eos] = 10000
                tokens, cache, _ = b.generate_from(out.past_key_values, logits, 4)
                assert tokens == [eos] and cache.get_seq_length() == len(history) + 1
                eos_rows.append(dict(backend=name, eos=eos, emitted_final_token_in_cache=True))
    assert artifact(mapper) == before
    write('results/phase2_v1/controls_cuda/gradient_eos.json', dict(passed=True, boundary_gradients=rows, eos=eos_rows,
        gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0), torch=torch.__version__, cuda=torch.version.cuda,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), checkpoint_unchanged=before))


if __name__ == '__main__':
    main()
