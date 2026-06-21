#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DATA_ROOT:=$ROOT_DIR/data}"
: "${OUTPUT_ROOT:=$ROOT_DIR/runs}"
export DATA_ROOT OUTPUT_ROOT

COMMON_ARGS=(
  --dataset cifar100_lt
  --rho 0.01
  --order shuffled
  --base-classes 10
  --incremental-steps 2
  --debug-num-classes 20
  --max-train-per-class 20
  --epochs 1
  --batch-size 64
  --seed 0
  --lambda-gpa 0.12
)

python3 train.py \
  "${COMMON_ARGS[@]}" \
  --method finetune_gpa \
  --output "$OUTPUT_ROOT/smoke_finetune_gpa"

python3 train.py \
  "${COMMON_ARGS[@]}" \
  --method taconcm_gpa \
  --output "$OUTPUT_ROOT/smoke_taconcm_gpa"

