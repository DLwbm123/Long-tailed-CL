#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-/dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23}"
RUN_ROOT="${RUN_ROOT:-/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_module_matrix_$(date +%Y%m%d-%H%M%S)}"
CKPT_ROOT="${CKPT_ROOT:-/dev/shm/wangbomin/LongTailedCL/checkpoints/$(basename "$RUN_ROOT")}"
CACHE_DIR="${CACHE_DIR:-/dev/shm/wangbomin/LongTailedCL/cache}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-5}"
BASE_CLASSES="${BASE_CLASSES:-13}"
INCREMENTAL_STEPS="${INCREMENTAL_STEPS:-5}"
MAX_PHASES="${MAX_PHASES:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.01}"
SCHEDULER="${SCHEDULER:-cosine}"
ORDER="${ORDER:-shuffled}"

METHODS="${METHODS:-finetune finetune_gpa taconcm_gpa taconcm_stage1_calib_init_only taconcm_stage2_calib_anchor taconcm_stage3_calib_tailanchor taconcm_stage4_full}"

mkdir -p "$RUN_ROOT" "$CKPT_ROOT" "$CACHE_DIR"

echo "run_root=$RUN_ROOT"
echo "ckpt_root=$CKPT_ROOT"
echo "data_root=$DATA_ROOT"
echo "methods=$METHODS"

for method in $METHODS; do
  out_dir="$RUN_ROOT/$method"
  ckpt_dir="$CKPT_ROOT/$method"
  mkdir -p "$out_dir" "$ckpt_dir"
  log="$out_dir/launch.log"
  echo "[$(date '+%F %T')] start method=$method" | tee "$log"
  CUDA_VISIBLE_DEVICES="$GPU" \
  XDG_CACHE_HOME="$CACHE_DIR" \
  TORCH_HOME="$CACHE_DIR/torch" \
  "$PYTHON_BIN" train.py \
    --dataset hyper_kvasir23 \
    --data-root "$DATA_ROOT" \
    --method "$method" \
    --order "$ORDER" \
    --seed "$SEED" \
    --base-classes "$BASE_CLASSES" \
    --incremental-steps "$INCREMENTAL_STEPS" \
    --max-phases "$MAX_PHASES" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --scheduler "$SCHEDULER" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device cuda \
    --output "$out_dir" \
    --ckpt-dir "$ckpt_dir" \
    --cache-dir "$CACHE_DIR" \
    --no-download \
    2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] done method=$method status=$status" | tee -a "$log"
  if [ "$status" -ne 0 ]; then
    echo "$method failed with status $status" >> "$RUN_ROOT/FAILED"
  fi
done

"$PYTHON_BIN" tools/summarize_hyperkvasir_module_matrix.py --root "$RUN_ROOT" | tee "$RUN_ROOT/module_summary.csv"
