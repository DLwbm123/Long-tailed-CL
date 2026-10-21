# HyperKvasir BiomedCLIP hierarchy-information pilot

## Question and scope

User explicitly requested a small HyperKvasir true-hierarchy vs flat-label vs shuffled-hierarchy experiment. This is an independently presealed frozen-CLIP module ablation, not a HASTEN reproduction, not an OPSA retry, and not a test of incremental forgetting. No changes to prior models, results, tolerances or clocks. The earlier no-retraining request applied to the now-completed OPSA evaluation; this new request authorizes the lightweight fits below.

Use only the current HK13 support (S_old3140 rows) and already-authorized E986/Qgate388 cohorts. Maintain existing group/row identities, validate semantic name binding from training manifest and group/path disjointness. Patient independence remains UNKNOWN. No future classes, future/test images, ISIC images, new clinical annotations, LLM calls, or query-driven taxonomy edits. Query metadata can be audited before training, but no query image/feature is read until all nine models are sealed. These cohorts are historically exposed development evaluations, not an independent confirmatory test.

## Frozen taxonomy

Official HyperKvasir discovery categories motivate a shallow CATEGORY hierarchy. It does not assert disease progression is subtype entailment. Source: https://www.nature.com/articles/s41597-020-00622-y . The map below is a fixed operational restriction to the current13 categories, not a new expert-validated ontology.

- Anatomical landmark/view: ileum(2), retroflex stomach(7), retroflex rectum(8), normal z line(10).
- Pathological finding: hemorrhoids(1), UCgrade3(3), UCgrade2(4), oesophagitisB-D(9), oesophagitisA(11), UCgrade1-2(12).
- Bowel preparation quality: BBPS0-1(5), BBPS2-3(6).
- Therapeutic intervention: dyed resection margins(0).
- Pathological subfamilies: hemorrhoids{1}, ulcerative colitis{3,4,12}, esophagitis{9,11}.

Severity grades are siblings; no ordinal loss or invented grade3-to-grade2 edge. No anatomy/etiology mixing within one level. Original label spellings remain bound to existing manifests; parent term hemorrhoids uses standard spelling. Leaf text: existing literal `a photo of a {class_name}.`; parent text: `an endoscopic image of {parent_name}.` Exact strings and class-index mapping are in core.py and runtime PLAN/TEXT seals.

## Matched models and controls

Frozen BiomedCLIP PubMedBERT-256 / ViT-B/16 FP32, eval mode, both TF32 flags false, benchmark false, existing open_clip environment. Use one pinned official 783705670-byte checkpoint and tokenizer/config files; no duplicate HF cache or separately pretrained text weights. All bytes count. Shared frozen image/text features encode once. No bone/encoder tuning and no low-dimensional text-subspace projection. Model selection is based on the suitability review below, before any new evaluation. Adapter dimension remains512. Preprocessing follows the official model config, text context256. Image encoder architecture and pretraining both change relative to prior B/32; cross-backbone differences are descriptive, not a medical-pretraining causal estimate.

Each adapter has two512x512 linear matrices (visual/text), no biases, identity initialization; same524288 trainable parameters for all arms. Normalize transformed features, temperature10 logits. Leaf13-way CE in all arms; true/shuffled add0.5*(four-type CE + pathological-family3-way CE). Subfamily loss is summed over relevant rows and divided by full batch size. Flat sets hierarchy coefficient to0; internal texts remain available but cannot affect leaf loss. Common identity penalty0.001*(mean-square visual deviation + mean-square text deviation). This measures added hierarchical supervision; true vs shuffled is the critical control against merely adding a regularizer.

Seeds74510/74511/74512; each draws the same600 batches of64 per arm with class/group-equal weights, preserving within-seed order. AdamW lr0.001 cosine decay to near0, weight_decay0.0001, clip gradient norm1. No search, early stopping, best-epoch selection, repeated query scoring, or seed replacement. Total5400 updates/backward across9models. Three deterministic shuffled leaf permutations preserve exact branch/subfamily sizes and change every top-level class-parent association; permutations depend only on seed and fixed labels, never images or performance. Their construction is in core.py, sealed before runtime.

At inference every arm uses the SAME fixed half-and-half mixture of adapted leaf softmax and native CLIP group-mean-prototype softmax, both temperature10. Native visual information is retained. Additional descriptive references: existing OriginalF, same CLIP native NCM, zero-shot and a single support-only class/group-weighted ridge (alpha0.01, existing normalization, no tuning). Ridge/prototypes are shared baselines, not alternative selected methods. OriginalF class counts must reproduce existing sealed E/Qgate reference.

## Integrity and outcomes

Synthetic checks: valid size-preserving permutations with all coarse assignments changed; identity permutation gives identical true/shuffled loss; flat leaf loss is independent of parent text changes; finite gradients and weight normalization. Actual semantic/identity checks, finite gradients/parameters/features/scores, exact saved-model readback, support score readback<=1e-6, and frozen-backbone version checks. All9 models and baselines sealed before E/Qgate encoding. Keep each cohort, pooled full13-class recall/confusion/group-BA and paired rescues/harms; no raw/tensor/identity export.

Primary exploratory module gate: true minus flat AND true minus shuffled mean pooled BA at least+1percentage point across3paired seeds; at least2/3 seeds positive for each comparison; both E and Qgate mean differences nonnegative. Separate conservative safety gate: in EACH seed/cohort/class, true recall at most1percentage point below matched flat, matched shuffled and OriginalF. Explicitly report tiny-class denominators, especially class1; no statistical-significance or population-risk claim from three optimization seeds. No bootstrap/CI. `HIERARCHY_PILOT_POINT_SIGNAL` only if both gates pass; otherwise report `NO_SAFE_HIERARCHY_POINT_SIGNAL` and retain any descriptive effects. method_success always false; no broad CL or clinical efficacy inference.

## Budget, execution and delivery

New independent pilot clock starts at first actual launch; hard total7200s including retries, TERM at7190/KILL5, CPU7200s, RSS6GiB, file1GiB (only to accommodate the official 783705670-byte checkpoint; total storage budget unchanged). If a deployment repair is needed, preserve original start and all previous costs/output; no scientific retry or retuning. Round newly allocated/logical maximum plus32MiB reserve strictly below1500000000 bytes; existing window+outside1001040384 below17000000000. Expected new peak below1GB (FP32 shared weight copy including NFS allocation, small cached matrices/models). Hard coordinator guards independent of SSH/Codex; no dependence on halfhour monitor. NFS mount/free/write-read probe and GPU0 memory check before launch. All data/model/temp/cache under existing NFS project; neutral parent/worker argv and GPU process names checked; no other-process changes/history cleanup.

Source preseal/push/ref/anonymous access before actual. Sole existing30-minute monitor updated in place. On exit retain successes/failures, count wait4/coordinator costs once, retire only current PIDs, publish source/protocol/aggregates/cost/report with proxy/ref/anonymous verification and dual delivery receipts. No clinical data/models/identities/private outputs or third-party source redistribution. A negative module result is preserved without changing tree/prompt/rank/seed/threshold to rescue it.

## Model suitability decision (2026-10-22 Asia/Shanghai)

MedCLIP (EMNLP2022) pretrains on MIMIC-CXR and CheXpert chest radiographs (paper section4.3/table3), with no established gastrointestinal-endoscopy validation there. It is not selected for this trial. https://aclanthology.org/2022.emnlp-main.256.pdf

Select the public MIT-licensed Microsoft BiomedCLIP checkpoint: multimodal biomedical PMC-15M, PubMedBERT text and ViT-B/16 images. Official model card: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 . The card establishes broad biomedical coverage, not a quantified gastrointestinal subset or proven HyperKvasir superiority. A separate VCE adaptation paper https://arxiv.org/abs/2410.19944 provides a feasibility precedent only; capsule images, fine-tuning and its labels are not equivalent to this frozen-feature HK13 protocol. Pretraining overlap with these evaluation images cannot be ruled out. No claim of an endoscopy-specialist backbone, no HK-finetuned third-party checkpoint, and no outcome-based model selection.

Pinned model revision9f341de24bfb00180f1b847274256e9b65a3a32e; text architecture config revisiond673b8835373c6fa116d6d8006b33d48734e305d. Reuse the existing environment; no installs. Download is inside the original7200-second guard and same1.5GB round/17GB window limits. Transport retries share that clock. No automatic alternate model.

Keep taxonomy, literal prompts, shuffled permutations, seed IDs, sample orders, optimizer,5400 total updates and524288 trainable parameters exactly the same as the completed CLIP pilot. Minority analysis reports all13 denominators/recalls/paired rescues-harms, explicitly including the previously tiny classes1,2,12. Cross-backbone flat-to-flat and true-to-true comparisons are descriptive after sealing, with the old immutable aggregate results; do not use them for stopping or selection. Do not conflate backbone benefit with hierarchy benefit.

## Deployment correction before any image encoding or update

Initial attempt ended by WINDOW_STORAGE_LIMIT during checkpoint download: partial logical579862528 bytes occupied1159725056 bytes (NFS allocated blocks about2x), worker exit-15, zero image/text encodes and zero updates. Prior forecast used logical checkpoint size and understated allocation; retain failure/cost receipts. Delete only this experiment's incomplete unusable download after recording its size/allocation; no historical assets removed. Stream the same pinned checkpoint into bounded RAM instead of disk; strictly load all FP32 parameters, no fp16 conversion, compression approximation or alternate model. Remaining NFS configs/models remain subject to unchanged1.5GB round/17GB window, and RAM remains6GiB. Original deadline1792606321.2900686 is retained; prior worker CPU14.012553 seconds deducted from7200. One deployment continuation, not scientific retry. Peak originalwindow16088031580+outside1001040384=17089071964 briefly exceeded the window threshold before two-second guard termination; this allocation overshoot is disclosed, not reported as compliant. Actual first attempt round1.1598GB remained below1.5GB; global per-experiment50GB cap was not approached.
