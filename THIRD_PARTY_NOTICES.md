# Third-party notices and licensing scope

First-party Gearshift software is licensed under the standard, unmodified
[Apache License 2.0](LICENSE). Copyright 2026 Impossible Computing, Inc.
This applies only to material Impossible Computing has the right to license.
It does **not** relicense models, datasets, benchmark statements/tests, or copied
upstream source. Third-party material retains its original terms and notices.

## HumanEval and HumanEval+ — approved upstream-public fixtures

The owner explicitly approved retaining the already-public HumanEval+ fixtures,
canonical solutions, and tests on 2026-09-21. They are no longer a release blocker.

- Dataset: [`results/phase2_v1/datasets/HumanEvalPlus-OriginFmt-v0.1.10.jsonl.gz`](results/phase2_v1/datasets/HumanEvalPlus-OriginFmt-v0.1.10.jsonl.gz), 164 records, SHA-256 `daa7661c8189924068069b0872a440b491edb60f8bdf431d5957adc88d18bae5`.
- Provenance: [official EvalPlus dataset release repository](https://github.com/evalplus/humanevalplus_release), [v0.1.10 asset](https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-OriginFmt.jsonl.gz), and [official dataset card](https://huggingface.co/datasets/evalplus/humanevalplus).
- Preserve [HumanEval+ release license](third_party/HumanEvalPlus_RELEASE_LICENSE.txt): Apache-2.0 plus the upstream statement that the datasets additionally comply with MIT because they are based on OpenAI HumanEval.
- Preserve [OpenAI HumanEval MIT license and copyright](third_party/HumanEval_LICENSE.txt). The previously included [EvalPlus software license](third_party/EvalPlus_LICENSE.txt) is also retained, not substituted for the dataset-specific notice.
- Upstream-origin fields in derived phase-two task records remain third-party material. They are not authored or relicensed by Impossible Computing. See [file-level provenance](third_party/README.md).

## LiveCodeBench — fetched upstream, not redistributed

The coding experiments use `livecodebench/code_generation_lite`, revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, release `release_v6`.
The [pinned dataset card](https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/README.md)
identifies the license only as `cc`. The MIT license on LiveCodeBench software is
not permission to redistribute contest problem text from LeetCode, AtCoder and
Codeforces. Gearshift does not relicense or redistribute this problem/test material
or reversible prompt tokens. Historical copies remain private.

`scripts/fetch_livecodebench.py` retrieves the pinned official files locally and
verifies file hashes, row hashes and task IDs against the retained membership manifest.
Downloaded data remains third-party material under upstream terms and must not be
committed or bundled. Compact analysis reproduction does not need that download.

## GSM8K

Questions and official answers originate from [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k),
revision `740312add88f781978c0658806c59bc2815b9866`. Preserve the
[upstream MIT license and copyright](third_party/GSM8K_LICENSE.txt). No GSM8K sample payload is bundled in this public tree. Dataset
questions/reference answers remain third-party content. Official answers were used for scoring, not to
produce training histories.

## WikiText

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher's
[WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext), revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, derives from Wikipedia.
The pinned card lists CC BY-SA 3.0/GFDL in metadata and links CC BY-SA 4.0 in its
licensing section. WikiText samples and reversible sampled token IDs are omitted from this public
tree; references in configs/reports are provenance, not bundled dataset content.

## Qwen models and dependencies

Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, and Qwen3-32B were used in different
stages. Exact revisions are in the corresponding configs/manifests. Their upstream
model terms remain separate; the models are not Impossible Computing-authored or
relicensed by this repository. Large model weights are not included in the compact
review bundles. A generated output may reproduce upstream text; generation alone
is not proof of independent redistribution rights.

Python dependencies retain their own licenses. Historical versions appear in
`requirements.lock.txt` and experiment-specific runtime records. Third-party
package source trees and environments are not bundled as first-party source.

Copied Qwen3 model metadata under `configs/coding_pilot_v1/reference/Qwen3-8B/`
and `Qwen3-32B/` retains upstream Apache-2.0 terms. The exact pinned upstream
licenses are in `third_party/Qwen3-8B_LICENSE.txt` and `Qwen3-32B_LICENSE.txt`.
No model weights or tokenizer vocabulary is bundled.
