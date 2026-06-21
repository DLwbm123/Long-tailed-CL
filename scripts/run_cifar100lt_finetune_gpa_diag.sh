#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
CACHE_DIR="${XDG_CACHE_HOME:-}"
CKPT_ROOT=""
RUN_ROOT=""
SEED=0
MODE="smoke"
USE_SWANLAB=0
SWANLAB_PROJECT="LongTailed-CL-TaConCM"
SWANLAB_MODE="online"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE="smoke"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
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
    --use-swanlab|--use_swanlab)
      USE_SWANLAB=1
      shift
      ;;
    --swanlab-project|--swanlab_project)
      SWANLAB_PROJECT="$2"
      shift 2
      ;;
    --swanlab-mode|--swanlab_mode)
      SWANLAB_MODE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$OUTPUT_ROOT/finetune_gpa_diag_${MODE}_s${SEED}"
fi
if [[ -z "$CKPT_ROOT" ]]; then
  CKPT_ROOT="$RUN_ROOT/checkpoints"
fi

mkdir -p "$RUN_ROOT" "$CKPT_ROOT"
if [[ -n "$CACHE_DIR" ]]; then
  mkdir -p "$CACHE_DIR"
fi

COMMON_ARGS=(
  --dataset cifar100_lt
  --rho 0.01
  --order shuffled
  --seed "$SEED"
  --data-root "$DATA_ROOT"
  --run-root "$RUN_ROOT"
  --diagnose-task-hist
  --gpa-anchor-reduction mean
)

if [[ "$MODE" == "smoke" ]]; then
  COMMON_ARGS+=(
    --base-classes 10
    --incremental-steps 2
    --debug-num-classes 20
    --max-train-per-class 20
    --epochs 1
    --batch-size 64
  )
else
  COMMON_ARGS+=(
    --base-classes 50
    --incremental-steps 5
    --epochs 100
    --batch-size 128
  )
fi

if [[ -n "$CACHE_DIR" ]]; then
  COMMON_ARGS+=(--cache-dir "$CACHE_DIR")
fi
if [[ "$USE_SWANLAB" -eq 1 ]]; then
  COMMON_ARGS+=(
    --use-swanlab
    --swanlab-project "$SWANLAB_PROJECT"
    --swanlab-mode "$SWANLAB_MODE"
  )
fi

for METHOD in finetune finetune_gpa; do
  METHOD_OUT="$RUN_ROOT/$METHOD"
  METHOD_CKPT="$CKPT_ROOT/$(basename "$RUN_ROOT")/$METHOD"
  mkdir -p "$METHOD_OUT" "$METHOD_CKPT"

  METHOD_ARGS=(
    --method "$METHOD"
    --output-dir "$METHOD_OUT"
    --ckpt-dir "$METHOD_CKPT"
  )
  if [[ "$METHOD" == "finetune_gpa" ]]; then
    METHOD_ARGS+=(--gpa-freeze-old-head --gpa-restore-old-head-after-step)
  else
    METHOD_ARGS+=(--no-gpa-freeze-old-head --no-gpa-restore-old-head-after-step)
  fi

  echo "running method=$METHOD output=$METHOD_OUT ckpt=$METHOD_CKPT"
  CMD=("$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}")
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi
  "${CMD[@]}"
done

"$PYTHON_BIN" tools/summarize_finetune_gpa_repro.py --root "$RUN_ROOT" --allow-failures
