# Experiments

This repository contains a CIFAR-100-LT smoke pipeline with deterministic
long-tail sampling, incremental task splits, Finetune, Finetune+GPA, and
TaConCM-GPA run artifacts.

## Local Smoke

```bash
bash scripts/run_smoke.sh
```

Equivalent Finetune command:

```bash
DATA_ROOT="$PWD/data" OUTPUT_ROOT="$PWD/runs" python3 train.py \
  --dataset cifar100_lt \
  --rho 0.01 \
  --order shuffled \
  --base-classes 10 \
  --incremental-steps 2 \
  --debug-num-classes 20 \
  --max-train-per-class 20 \
  --method finetune \
  --epochs 1 \
  --batch-size 64 \
  --seed 0 \
  --output runs/smoke_finetune
```

Success for this milestone means all phases complete, classifier dimensions grow
from 10 to 15 to 20, and the output directory contains `config.yaml`,
`class_counts.json`, `class_order.json`, `metrics.jsonl`, `summary.json`,
`summary.csv`, `acc_matrix.csv`, `run.log`, and checkpoints.

## GPA and TaConCM-GPA Smoke

For the six-stage TaConCM-GPA ablation pipeline, see
[`docs/TACONCM_EXPERIMENTS.md`](../docs/TACONCM_EXPERIMENTS.md).

```bash
python3 tools/check_ltconcm_calibrator.py
bash scripts/run_smoke_gpa.sh
```

Equivalent TaConCM-GPA minimal command:

```bash
DATA_ROOT="$PWD/data" OUTPUT_ROOT="$PWD/runs" python3 train.py \
  --dataset cifar100_lt \
  --rho 0.01 \
  --order shuffled \
  --base-classes 10 \
  --incremental-steps 2 \
  --debug-num-classes 20 \
  --max-train-per-class 20 \
  --method taconcm_gpa \
  --lambda-gpa 0.12 \
  --epochs 1 \
  --batch-size 64 \
  --seed 0 \
  --output runs/smoke_taconcm_gpa
```

The GPA/TaConCM smoke should additionally write finite `gpa_anchor_loss`,
`method_extra_loss`, prototype diagnostics, and TaConCM alpha/calibration fields
to `metrics.jsonl`.

## Full CIFAR-100-LT

Server helper:

```bash
DATA_ROOT=/path/to/cifar OUTPUT_ROOT=/path/to/runs bash scripts/run_cifar100lt_taconcm.sh \
  --method taconcm_gpa \
  --rho 0.01 \
  --incremental-steps 5 \
  --order shuffled \
  --seed 0 \
  --lambda-gpa 0.12
```

Use `--method finetune_gpa` for the GPA baseline and add
`--ltconcm-use-tdsm --ltconcm-use-match-loss` for the optional full
TaConCM-GPA mode.
