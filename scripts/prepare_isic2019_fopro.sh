#!/usr/bin/env bash
set -euo pipefail

DEST="${DEST:-/Volumes/DataP/CL_medical_classification/ISIC2019_FoProKD}"
SPLIT_SRC="${SPLIT_SRC:-src/datasets/fopro_splits/ham2019}"
IMAGE_ZIP="$DEST/ISIC_2019_Training_Input.zip"
IMAGE_DIR="$DEST/ISIC_2019_Training_Input"
URL="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"

mkdir -p "$DEST/ham2019"

if [ ! -d "$SPLIT_SRC" ]; then
  echo "Missing split directory: $SPLIT_SRC" >&2
  exit 1
fi
cp "$SPLIT_SRC"/*.csv "$DEST/ham2019/"
find "$DEST/ham2019" -type f -name '._*' -delete

if [ ! -d "$IMAGE_DIR" ]; then
  NEED_DOWNLOAD=1
  if [ -f "$IMAGE_ZIP" ] && unzip -tq "$IMAGE_ZIP" >/dev/null 2>&1; then
    NEED_DOWNLOAD=0
  fi

  if [ "$NEED_DOWNLOAD" -eq 1 ]; then
    echo "Downloading or resuming $URL"
    if command -v curl >/dev/null 2>&1; then
      curl -L -C - "$URL" -o "$IMAGE_ZIP"
    elif command -v wget >/dev/null 2>&1; then
      wget -c "$URL" -O "$IMAGE_ZIP"
    else
      echo "Neither curl nor wget is available." >&2
      exit 1
    fi
  fi

  if ! unzip -tq "$IMAGE_ZIP" >/dev/null 2>&1; then
    echo "Zip is incomplete or corrupt after download: $IMAGE_ZIP" >&2
    exit 1
  fi

  echo "Extracting $IMAGE_ZIP"
  unzip -q "$IMAGE_ZIP" -d "$DEST"
fi

IMAGE_COUNT=$(find "$IMAGE_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')
SPLIT_COUNT=$(find "$DEST/ham2019" -type f -name '*.csv' ! -name '._*' | wc -l | tr -d ' ')

echo "ISIC FoPro-KD data root: $DEST"
echo "Image dir: $IMAGE_DIR"
echo "Image files: $IMAGE_COUNT"
echo "Split CSVs: $SPLIT_COUNT"
echo
echo "Example preview command:"
echo "python3 train.py --dataset isic2019_lt --data-root '$IMAGE_DIR' --preview-dataset"
