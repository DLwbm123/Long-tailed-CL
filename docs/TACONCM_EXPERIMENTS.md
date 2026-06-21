# TaConCM-GPA CIFAR-100-LT Experiments

This document describes the reproducible six-stage ablation pipeline for
CIFAR-100-LT. The pipeline does not change the dataset protocol: the training
split is long-tailed, the test split stays balanced, and the task split is
written to each run directory.

## Six-Stage Ablation

1. `finetune`
   - Plain Finetune baseline.
   - No GPA anchoring and no TaConCM components.

2. `finetune_gpa`
   - Original GPA path.
   - GPA classifier initialization and GPA anchoring begin at
     `--gpa-anchor-start-phase 1`.
   - Anchor target is the raw GPA target.

3. `taconcm_stage1_calib_init_only`
   - GPA plus TaConCM prototype calibration for classifier initialization only.
   - Uses `--ltconcm-calibrate-prototypes`.
   - Uses `--ltconcm-anchor-target gpa_raw`.
   - Tail-balanced anchoring, T-DSM, and match loss are disabled.

4. `taconcm_stage2_calib_anchor`
   - Stage 1 plus calibrated prototype anchor targets.
   - Uses `--ltconcm-anchor-target calibrated`.
   - Tail-balanced anchoring, T-DSM, and match loss are disabled.

5. `taconcm_stage3_calib_tailanchor`
   - Stage 2 plus tail-balanced anchor weights.
   - Uses `--ltconcm-tail-anchor`.
   - T-DSM and match loss are disabled.

6. `taconcm_stage4_full`
   - Stage 3 plus T-DSM anchors and feature-structure match loss.
   - Uses `--ltconcm-anchor-target tdsm`, `--ltconcm-use-tdsm`, and
     `--ltconcm-use-match-loss`.

## Local Smoke

Run the full six-stage smoke locally:

```bash
bash scripts/run_cifar100lt_taconcm_ablation.sh \
  --smoke \
  --seed 0 \
  --run-root runs/taconcm_ablation_smoke_s0
```

The smoke mode uses:

- `--base-classes 10`
- `--incremental-steps 2`
- `--debug-num-classes 20`
- `--max-train-per-class 20`
- `--epochs 1`
- `--batch-size 64`

## Server Full Run

Use `DATA_ROOT` and `OUTPUT_ROOT` to keep data and outputs explicit:

```bash
DATA_ROOT=/path/to/cifar OUTPUT_ROOT=/path/to/runs \
bash scripts/run_cifar100lt_taconcm_ablation.sh \
  --full \
  --seed 0 \
  --rho 0.01 \
  --order shuffled \
  --lambda-gpa 0.12 \
  --run-root /path/to/runs/taconcm_ablation_full_s0
```

Full mode uses the standard CIFAR-100-LT split with `--base-classes 50` and
`--incremental-steps 5`. Extra `train.py` flags can be passed after the script
arguments if needed.

## Summary

Summarize and validate all six runs:

```bash
python3 tools/summarize_taconcm_ablation.py \
  --root runs/taconcm_ablation_smoke_s0
```

The summarizer checks that each run wrote required reproducibility metadata,
that phase-0 GPA anchoring is disabled when `--gpa-anchor-start-phase 1`, that
losses are finite, and that each ablation stage enables only its intended
components.

## Expected Output Files

Each per-stage output directory should contain:

- `config.yaml`
- `class_counts.json`
- `class_order.json`
- `metrics.jsonl`
- `summary.json`
- `summary.csv`
- `acc_matrix.csv`
- `run.log`
- `checkpoints/model_phase_*.pt`

`metrics.jsonl`, `summary.json`, `config.yaml`, and checkpoints include run
metadata such as method, ablation stage, seed, dataset, imbalance ratio, task
split, order mode, GPA flags, TaConCM flags, `gpa_anchor_start_phase`,
`ltconcm_anchor_target`, and the git commit hash when the checkout has one.

