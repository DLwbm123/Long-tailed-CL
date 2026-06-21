#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-hyper_kvasir23}"

case "$DATASET" in
  hyper_kvasir23)
    DATA_ROOT="${DATA_ROOT:-/Volumes/DataP/CL_medical_classification/hyper-kvasir23}"
    ;;
  isic2019_lt)
    DATA_ROOT="${DATA_ROOT:-/Volumes/DataP/CL_medical_classification/ISIC2019_FoProKD/ISIC_2019_Training_Input}"
    ;;
  *)
    echo "Unsupported DATASET=$DATASET" >&2
    exit 1
    ;;
esac

"$PYTHON_BIN" train.py \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --preview-dataset \
  "$@"
