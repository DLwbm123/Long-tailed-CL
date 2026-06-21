#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DATA_ROOT:=$ROOT_DIR/data}"
: "${OUTPUT_ROOT:=$ROOT_DIR/runs}"
: "${PYTHON_BIN:=python3}"
export DATA_ROOT OUTPUT_ROOT

MODE="smoke"
SEED="0"
RHO="0.01"
ORDER="shuffled"
LAMBDA_GPA="0.12"
EPOCHS=""
BATCH_SIZE=""
RUN_ROOT=""
DATA_ROOT_ARG="$DATA_ROOT"
CACHE_DIR_ARG="${CACHE_DIR:-}"
CKPT_ROOT_ARG="${CKPT_ROOT:-}"
SWANLAB_FLAG=""
SWANLAB_PROJECT=""
SWANLAB_MODE=""
SWANLAB_RUN_NAME=""
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
    --rho)
      RHO="$2"
      shift 2
      ;;
    --order)
      ORDER="$2"
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
    --run-root|--run_root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --data-root|--data_root)
      DATA_ROOT_ARG="$2"
      DATA_ROOT="$2"
      shift 2
      ;;
    --cache-dir|--cache_dir)
      CACHE_DIR_ARG="$2"
      shift 2
      ;;
    --ckpt-dir|--ckpt_dir)
      CKPT_ROOT_ARG="$2"
      shift 2
      ;;
    --use-swanlab|--use_swanlab|--no-use-swanlab|--no-use_swanlab)
      SWANLAB_FLAG="$1"
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
    --swanlab-run-name|--swanlab_run_name)
      SWANLAB_RUN_NAME="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$MODE" == "smoke" ]]; then
  : "${EPOCHS:=1}"
  : "${BATCH_SIZE:=64}"
  : "${RUN_ROOT:=$OUTPUT_ROOT/taconcm_ablation_smoke_s${SEED}}"
  COMMON_DATA_ARGS=(
    --base-classes 10
    --incremental-steps 2
    --debug-num-classes 20
    --max-train-per-class 20
  )
else
  : "${EPOCHS:=100}"
  : "${BATCH_SIZE:=128}"
  : "${RUN_ROOT:=$OUTPUT_ROOT/taconcm_ablation_full_s${SEED}}"
  COMMON_DATA_ARGS=(
    --base-classes 50
    --incremental-steps 5
  )
fi

mkdir -p "$RUN_ROOT"

METHODS=(
  finetune
  finetune_gpa
  taconcm_stage1_calib_init_only
  taconcm_stage2_calib_anchor
  taconcm_stage3_calib_tailanchor
  taconcm_stage4_full
)

for method in "${METHODS[@]}"; do
  DEPLOY_ARGS=(--data-root "$DATA_ROOT_ARG")
  if [[ -n "$CACHE_DIR_ARG" ]]; then
    DEPLOY_ARGS+=(--cache-dir "$CACHE_DIR_ARG")
  fi
  if [[ -n "$CKPT_ROOT_ARG" ]]; then
    DEPLOY_ARGS+=(--ckpt-dir "$CKPT_ROOT_ARG/$(basename "$RUN_ROOT")/$method")
  fi
  if [[ -n "$SWANLAB_FLAG" ]]; then
    DEPLOY_ARGS+=("$SWANLAB_FLAG")
  fi
  if [[ -n "$SWANLAB_PROJECT" ]]; then
    DEPLOY_ARGS+=(--swanlab-project "$SWANLAB_PROJECT")
  fi
  if [[ -n "$SWANLAB_MODE" ]]; then
    DEPLOY_ARGS+=(--swanlab-mode "$SWANLAB_MODE")
  fi
  if [[ -n "$SWANLAB_RUN_NAME" ]]; then
    DEPLOY_ARGS+=(--swanlab-run-name "${SWANLAB_RUN_NAME}_${method}")
  elif [[ "$SWANLAB_FLAG" == "--use-swanlab" || "$SWANLAB_FLAG" == "--use_swanlab" ]]; then
    DEPLOY_ARGS+=(--swanlab-run-name "$(basename "$RUN_ROOT")_${method}")
  fi

  "$PYTHON_BIN" train.py \
    --dataset cifar100_lt \
    --method "$method" \
    --rho "$RHO" \
    --order "$ORDER" \
    "${COMMON_DATA_ARGS[@]}" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --lambda-gpa "$LAMBDA_GPA" \
    --gpa-anchor-start-phase 1 \
    --output "$RUN_ROOT/$method" \
    "${DEPLOY_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
done

echo "ablation_run_root=$RUN_ROOT"
