#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DATA_ROOT:=$ROOT_DIR/data}"
: "${OUTPUT_ROOT:=$ROOT_DIR/runs}"
export DATA_ROOT OUTPUT_ROOT

python3 train.py \
  --dataset cifar100_lt \
  --rho 0.01 \
  --order shuffled \
  --base-classes 10 \
  --incremental-steps 2 \
  --debug-num-classes 20 \
  --max-train-per-class 20 \
  --preview-dataset

python3 train.py \
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
  --output "$OUTPUT_ROOT/smoke_finetune"
