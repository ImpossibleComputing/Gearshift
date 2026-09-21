# Upstream licenses and provenance

These files preserve upstream terms. They are not licenses authored by Impossible
Computing and do not relicense third-party content as first-party Gearshift work.

| License file | Upstream material |
|---|---|
| `HumanEvalPlus_RELEASE_LICENSE.txt` | Official `evalplus/humanevalplus_release` Apache-2.0 license, including its supplemental statement that the datasets also comply with MIT because they build on OpenAI HumanEval. |
| `HumanEval_LICENSE.txt` | OpenAI HumanEval MIT license and copyright notice. |
| `EvalPlus_LICENSE.txt` | EvalPlus software license retained from the original research package; distinct from the dataset release license above. |
| `GSM8K_LICENSE.txt` | OpenAI GSM8K/grade-school-math MIT license and copyright notice. |

## Bundled HumanEval+ dataset

- File: [`results/phase2_v1/datasets/HumanEvalPlus-OriginFmt-v0.1.10.jsonl.gz`](../results/phase2_v1/datasets/HumanEvalPlus-OriginFmt-v0.1.10.jsonl.gz).
- Upstream: [official v0.1.10 release asset](https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-OriginFmt.jsonl.gz).
- Original upstream filename: `HumanEvalPlus-OriginFmt.jsonl.gz`; the local filename adds the version. Compressed dataset contents were not changed for this release preparation.
- Preserved file SHA-256: `daa7661c8189924068069b0872a440b491edb60f8bdf431d5957adc88d18bae5`.
- Contents: 164 upstream records with prompts, canonical solutions, entry points, and tests. These are intentionally included upstream-public benchmark fixtures, **not** Impossible Computing-authored tasks or private LiveCodeBench grading data.
- Derived task records created by `scripts/phase2_prepare_tasks.py` identify `humanevalplus` as their origin and preserve official task IDs. Upstream prompts, reference solutions, and tests in those records retain these upstream terms; task wrappers, model-generated responses, and research metrics are separate layers of provenance.
- [Official release repository](https://github.com/evalplus/humanevalplus_release) and [official Hugging Face dataset](https://huggingface.co/datasets/evalplus/humanevalplus).

The owner explicitly approved preserving these fixtures and existing history on
2026-09-21. The earlier snapshot remains unchanged in the private archive.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for other materials and the
public redistribution boundary.

## Copied Qwen metadata

`configs/coding_pilot_v1/reference/Qwen3-8B/` and `Qwen3-32B/` contain upstream
config, generation config, tokenizer config (chat template, not vocabulary) and
weight-index metadata. Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`
and Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137` are pinned.
Their respective `Qwen3-8B_LICENSE.txt` and `Qwen3-32B_LICENSE.txt` files were fetched
verbatim from those revisions on Hugging Face. They are third-party Apache-2.0
material, not Impossible Computing-authored files.
