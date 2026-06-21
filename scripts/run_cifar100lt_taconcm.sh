#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DATA_ROOT:=$ROOT_DIR/data}"
: "${OUTPUT_ROOT:=$ROOT_DIR/runs}"
export DATA_ROOT OUTPUT_ROOT

METHOD="taconcm_gpa"
RHO="0.01"
INCREMENTAL_STEPS="5"
ORDER="shuffled"
SEED="0"
LAMBDA_GPA="0.12"
EPOCHS="100"
BATCH_SIZE="128"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD="$2"
      shift 2
      ;;
    --rho)
      RHO="$2"
      shift 2
      ;;
    --incremental-steps|--incremental_steps|--tasks)
      INCREMENTAL_STEPS="$2"
      shift 2
      ;;
    --order)
      ORDER="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --lambda-gpa|--lambda_gpa)
      LAMBDA_GPA="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size|--batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

RUN_NAME="cifar100lt_rho${RHO//./}_steps${INCREMENTAL_STEPS}_${ORDER}_${METHOD}_s${SEED}_lgpa${LAMBDA_GPA//./}"

python3 train.py \
  --dataset cifar100_lt \
  --method "$METHOD" \
  --rho "$RHO" \
  --order "$ORDER" \
  --base-classes 50 \
  --incremental-steps "$INCREMENTAL_STEPS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --lambda-gpa "$LAMBDA_GPA" \
  --output "$OUTPUT_ROOT/$RUN_NAME" \
  "${EXTRA_ARGS[@]}"

