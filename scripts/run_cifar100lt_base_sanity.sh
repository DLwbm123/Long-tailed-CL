#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
CACHE_DIR="${XDG_CACHE_HOME:-}"
CKPT_ROOT=""
RUN_ROOT=""
SEED=0
EPOCHS=100
RHO=0.01
ORDER="shuffled"
BASE_CLASSES=50
COUNT_ASSIGNMENT="original_id"
SCHEDULER="constant"
LR=0.1
BATCH_SIZE=128
WEIGHT_DECAY=5e-4
MOMENTUM=0.9
MILESTONES="60,80"
GAMMA=0.1
NUM_WORKERS=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --run-root|--run_root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --data-root|--data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --cache-dir|--cache_dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --ckpt-dir|--ckpt_dir)
      CKPT_ROOT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --rho)
      RHO="$2"
      shift 2
      ;;
    --order)
      ORDER="$2"
      shift 2
      ;;
    --base-classes|--base_classes)
      BASE_CLASSES="$2"
      shift 2
      ;;
    --count-assignment|--count_assignment)
      COUNT_ASSIGNMENT="$2"
      shift 2
      ;;
    --scheduler)
      SCHEDULER="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --batch-size|--batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --weight-decay|--weight_decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --momentum)
      MOMENTUM="$2"
      shift 2
      ;;
    --milestones)
      MILESTONES="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --num-workers|--num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$OUTPUT_ROOT/base_sanity_s${SEED}"
fi
METHOD_OUT="$RUN_ROOT/${COUNT_ASSIGNMENT}_${SCHEDULER}"
if [[ -z "$CKPT_ROOT" ]]; then
  CKPT_ROOT="$METHOD_OUT/checkpoints"
else
  CKPT_ROOT="$CKPT_ROOT/$(basename "$RUN_ROOT")/${COUNT_ASSIGNMENT}_${SCHEDULER}"
fi

mkdir -p "$METHOD_OUT" "$CKPT_ROOT"
if [[ -n "$CACHE_DIR" ]]; then
  mkdir -p "$CACHE_DIR"
fi

CMD=(
  "$PYTHON_BIN" train.py
  --dataset cifar100_lt
  --method finetune
  --base-only
  --rho "$RHO"
  --order "$ORDER"
  --base-classes "$BASE_CLASSES"
  --incremental-steps 5
  --count-assignment "$COUNT_ASSIGNMENT"
  --scheduler "$SCHEDULER"
  --milestones "$MILESTONES"
  --gamma "$GAMMA"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --momentum "$MOMENTUM"
  --weight-decay "$WEIGHT_DECAY"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --data-root "$DATA_ROOT"
  --run-root "$RUN_ROOT"
  --output-dir "$METHOD_OUT"
  --ckpt-dir "$CKPT_ROOT"
  --no-use-swanlab
)

if [[ -n "$CACHE_DIR" ]]; then
  CMD+=(--cache-dir "$CACHE_DIR")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "running base sanity count_assignment=$COUNT_ASSIGNMENT scheduler=$SCHEDULER output=$METHOD_OUT"
"${CMD[@]}"
