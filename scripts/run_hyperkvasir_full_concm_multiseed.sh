#!/usr/bin/env bash
set -euo pipefail

BASE=/remote-home/wangbomin/LongTailedCL
CODE="$BASE/code"
OUT=${FULL_CONCM_OUTPUT_ROOT:-$BASE/outputs/hyperkvasir23_full_concm_20260710_gpu1}
PY=/root/anaconda3/bin/python
PHYSICAL_GPU=${PHYSICAL_GPU:-1}

mkdir -p "$OUT"/{cache,configs,csv,json,logs,manifests,runs}

GPU_UUID=$(nvidia-smi -i "$PHYSICAL_GPU" --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "^${GPU_UUID},"; then
  echo "Physical GPU${PHYSICAL_GPU} is occupied; refusing to share or switch GPU." >&2
  exit 75
fi

run_one() {
  local seed="$1"
  local mode="$2"
  local run_dir="$OUT/runs/seed${seed}_${mode}"
  local log="$OUT/logs/seed${seed}_${mode}.log"
  local checkpoint="$BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260709_s${seed}_5ep/phase0_with_anchors.pt"
  local cache="$OUT/cache/seed${seed}_feature_bank.pt"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $run_dir" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" "$PY" "$CODE/tools/run_hyperkvasir_full_concm.py" \
    --mode "$mode" \
    --seed "$seed" \
    --data-root "$BASE/data/hyper-kvasir23" \
    --base-checkpoint "$checkpoint" \
    --feature-cache "$cache" \
    --output-dir "$run_dir" \
    --start-session 0 --end-session 5 \
    --device cuda --batch-size 128 --num-workers 4 \
    --projector-hidden 2048 --projector-dim 128 \
    --base-projector-epochs 5 --increment-projector-epochs 5 \
    --mpc-epochs 20 --mpc-lr 0.01 --projector-lr 0.01 \
    --mpc-alpha 0.75 --covariance-gamma 16 --covariance-beta 0.6 \
    --lambda-dsm 0.2 2>&1 | tee "$log"
}

cd "$CODE"

# Seed3 is the engineering and collapse smoke. All jobs are strictly serial.
run_one 3 locked_nc
run_one 3 full_dynamic
run_one 3 nc_anchored

run_one 1 locked_nc
run_one 1 full_dynamic
run_one 1 nc_anchored

run_one 2 locked_nc
run_one 2 full_dynamic
run_one 2 nc_anchored

"$PY" "$CODE/tools/summarize_hyperkvasir_full_concm.py" --root "$OUT" \
  2>&1 | tee "$OUT/logs/summarize.log"
