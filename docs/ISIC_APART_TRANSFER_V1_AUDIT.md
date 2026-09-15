# ISIC APART Transfer V1 — P0 audit

Run ID: `isic_apart_v1_20260915_p0_q8m4`  
Reference: `7a089e4b30a3efa6e2943aa4b99c1f5e60652b16`  
State: **BLOCKED_DATA_PROTOCOL + BLOCKED_TRAINING_SEMANTICS**

This is an executed preflight audit, not a completed medical-transfer experiment.
No formal trajectory, training batch, session checkpoint, or model test prediction
was produced. The user's requested GPU3 remains the sole formal-training target.
The user explicitly adopted the attachment's stop conditions; they are not
permissions inferred from an untrusted document.

## Workspace protection and reference

The original workspace is source-only and has no `.git`, so its `git rev-parse HEAD`
and Git status are unavailable rather than a fabricated clean HEAD. All 341
publishable files were copied into an isolated checkout. The exact reference tree
and commit object were recovered from GitHub and checked with Git's object tools.
The isolated branch starts at the reference commit above and is named
`exp/apart-isic-lt-transfer-v1`.

Five existing CSV files differ only in CRLF versus LF. No implementation/configuration
content difference was found. Those original bytes remain unchanged in the original
workspace and as unstaged differences in the isolated checkout. The source patch is
retained in the private audit bundle. Existing data, checkpoints, outputs, and
untracked runtime material in the original workspace were not changed or removed.

## Actual environment and storage

The live target is `my-gpu`, physical GPU3, NVIDIA A100-PCIE-40GB. At the recorded
2026-09-15 02:00:53 UTC probe it had 24,964 MiB free; utilization was 59%. Shared
GPU use was allowed, so nonzero utilization was not treated as a blocker. No existing
process was killed or restarted.

Actual Python is 3.12.7, PyTorch 2.6.0+cu124, torchvision 0.21.0, timm 1.0.15,
NumPy 1.26.4, and Pillow 10.4.0. The target artifact directory is on NFS4 with
approximately 353 TiB available according to `df -hT`; a small create/write/read/remove
probe passed. Full environment and checked weight-search roots are in the private
`environment.json`.

The image audit ran against existing DataP images on local CPU using bundled
Python 3.12.14 and Pillow 12.3.0. The original-component probe ran on the remote
CPU with PyTorch 2.6.0; neither operation allocated GPU memory.

No matching ViT pretrained weights were located in the checked project, Torch,
timm, Hugging Face, and shared-cache paths. The historical APART `/dev/shm` directory
is absent. The source calls `timm.create_model("vit_base_patch16_224", pretrained=True)`;
a model name alone does not identify the exact historical bytes across timm versions.
The historical SHA256 and ISIC pretraining-overlap status remain unknown. No weights
were downloaded and no random backbone was substituted. This is an additional
**BLOCKED_PRETRAINED_WEIGHTS** prerequisite, scoped to the paths actually checked.

## Fixed split and actual counts

Only the three reference `*_skin1_0.01.csv` files were used. No resampling, truncation,
replacement split, or extra training image was introduced.

| original finding ID | train | val | test |
|---:|---:|---:|---:|
| 0 | 12725 | 50 | 100 |
| 1 | 4372 | 50 | 100 |
| 2 | 3173 | 50 | 100 |
| 3 | 1788 | 50 | 100 |
| 4 | 717 | 50 | 100 |
| 5 | 478 | 50 | 100 |
| 6 | 103 | 50 | 100 |
| 7 | 89 | 50 | 100 |
| total | 23445 | 400 | 800 |

The suffix is `0.01`, but the actual maximum/minimum ratio is **142.98:1**.
The plan requires reporting the actual CSV distribution; it does not authorize
resampling to make that ratio exactly 100. The fixed evaluator groups are tail
`[7,6]`, middle `[5,4,3,2]`, and head `[1,0]`.

All **24,645** selected images resolved and decoded successfully; **9,541,488,783
bytes** were read/hashed during the explicitly requested integrity audit, taking
682.26 seconds. Image-ID intersections between each pair of splits are empty.
However, SHA256 grouping found **41 identical-content groups**, including **8
cross-split groups** (train/val 2, train/test 6, val/test 0), and **2 groups whose
identical bytes have conflicting numerical labels**. Content-level label conflicts
were computed from the saved manifest without rereading the images; the audit
tool and synthetic regression were then extended to flag this case directly.
No files or labels were corrected. These are independently sufficient data gates.
The full file/decode/hash outcome is retained in
the accompanying [aggregate audit evidence](isic_apart_transfer_v1/dataset_audit.json).
Integrity decoding of test images is permitted P0 access, and is separately recorded
from model test inference, which remains zero.

## Confirmed lesion leakage

Available local ISIC metadata was joined using exact `isic_id`, not filename
similarity or inferred patient identities. It contains 81,722 rows with 22,680 IDs
matching the selected image set. There are no duplicate or conflicting metadata
rows for matched IDs. Known lesion IDs cover **21,854 / 24,645 (88.68%)** selected
images. Patient IDs cover zero selected images, so patient isolation is unknown.

There are **693 distinct known lesions appearing in more than one split**:

| split pair | shared lesion count |
|---|---:|
| train / val | 243 |
| train / test | 475 |
| val / test | 75 |

These pair counts overlap and must not be added. This is a positive leakage
finding despite incomplete coverage. The adopted plan §3.2 requires
`BLOCKED_DATA_PROTOCOL`; missing patient IDs do not waive observed lesion leakage.
Metadata SHA256: `00457932f6f32acf134467de30273e12198518bbf8df3802c4603860259428c9`.
All lesion IDs and sample-level joins stay in the private bundle.

Matched diagnosis strings support the following partial semantic interpretation:
0 nevus, 1 melanoma, 2 basal cell carcinoma, 3 a group containing seborrheic
keratosis/solar lentigo/pigmented benign keratosis, 4 actinic keratosis,
5 squamous cell carcinoma, 6 vascular lesion, 7 dermatofibroma. Coverage is partial;
numeric finding IDs remain authoritative and `SEMANTIC_MAP_UNVERIFIED` is retained
for the full selected set. No disease-order guess was used to relabel samples.

Development exposure is **UNKNOWN**. The checked local reports/configs/runs and
remote project run-directory names did not reveal prior ISIC training; this is
not proof that the split was never used anywhere.

## Reproduced training incompatibility

The reference `PoolAssigner` uses `nn.Embedding(501, 16)`, hence valid indices
0–500. The original training path sets `weight = cls_num_list[targets]`, passes
it to `forward_pool`, casts to integer, and looks up that embedding. Classes
0–4 have counts larger than 500.

The [runnable preflight](../tests/test_apart_medical_preflight.py) executed the
unchanged AST-defined `PoolAssigner` on CPU using all eight real CSV counts:
**all five out-of-range counts raised `IndexError: index out of range in self`**;
478, 103, and 89 succeeded. This does not depend on unavailable pretrained weights.
Every specified repeat already includes out-of-range classes in S0.

Source trace: `models/apart.py` real-sample count lookup →
`AdapterVitNet.forward` → `VisionTransformer.forward_pool` →
`PoolAssigner.forward` / `cls_emb(weight)`.

Increasing the embedding to 12,726 rows would retain literal count indexing but
introduce new trainable parameters/initialization; clipping or rescaling counts
would change input semantics. Neither revision was applied. The user explicitly
requires a stop before substantial training-semantics changes.

This is **not** a confirmed label-dependent-logit finding. Source tracing indicates
`weight` affects `pool_id`, which is used by training losses, while the main/few
logits are computed independently of it. A full pretrained label-blind
counterfactual test was not run and is recorded as NOT_RUN, not PASS.

## Actual legacy optimizer behavior

Original methods were executed on a small CPU parameter holder, without rewriting
their logic. This is an optimizer-method probe, not a trained ViT.

- JSON `weight_decay=0` is stored under the misspelled parameter-group key
  `weight decay`; actual AdamW `weight_decay` is **0.01** in both groups.
- Actual initial group learning rates are **0.0003 / 0.003**, betas **(0.9,0.999)**.
- `init_cls=4 != increment=2` overwrites scheduler selection to cosine.
- The S0 scheduler binds to the active optimizer. Incremental optimizer recreation
  leaves the scheduler bound to the earlier optimizer, not the one taking steps.
- These findings are marked `legacy_effective`; no optimizer/scheduler correction
  was applied and no actual training epoch/LR trace exists for this blocked run.

## Engineering and budget status

Two synthetic audit regression tests passed: distinct IDs do not hide equal image
content across splits, and missing group IDs remain unknown while known overlap
blocks. The original-component preflight passed its reproduction assertions and
returned BLOCKED_TRAINING_SEMANTICS. These are not a P1 TECHNICAL_PASS.

Full 4→6→8 B/C smoke, full pretrained label-blindness, reload/resume, gradient,
evaluation-purity, and paired S0 equivalence tests remain NOT_RUN due to P0 gates.
No formal `PROTOCOL_LOCK.json` was issued.

The unchanged six-trajectory design requires 29,400 optimizer steps from the
actual counts and batch size 48, plus statistics extraction and evaluation.
GPU training use is **0 hours**; measured-throughput budget estimation is not yet
possible, so the 24 GPU-hour gate is **NOT_EVALUABLE**, not passed or exceeded.

## Minimal pending decisions

1. Supply an approved split resolving identical-content leakage, conflicting
   labels, and lesion overlap, or explicitly specify a revised protocol and how
   these findings should be handled. The current fixed-split V1 cannot pass;
   no split was silently rebuilt.
2. If proceeding under a revised protocol, explicitly approve the 12,726-row
   count-embedding capacity revision (same literal count inputs and losses),
   including reproducible B/C initialization and a fresh P1 check.
3. Identify/provide the exact approved pretrained weight file and provenance, or
   explicitly revise the historical-weight requirement to a newly specified
   checkpoint. No silent timm-default replacement is acceptable.

No Full Dynamic or other closed/new method was reopened.
