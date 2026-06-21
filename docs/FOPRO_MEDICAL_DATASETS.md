# FoPro-KD Medical Dataset Integration

Date: 2026-06-20

## Scope

This repo now supports two FoPro-KD style medical long-tailed image datasets in the existing `train.py` / `src/engine/finetune.py` pipeline:

- `hyper_kvasir23`
- `isic2019_lt`

The training algorithm is unchanged. The patch only adds dataset protocols, transforms, split metadata, and CLI/config plumbing.

## Paper Protocol Notes

FoPro-KD reports long-tailed medical image recognition, not class-incremental learning. Therefore medical datasets default to a one-phase all-class recognition setup:

- `base_classes = num_classes`
- `incremental_steps = 0`

If a class-incremental split is needed, explicitly pass `--base-classes` and `--incremental-steps`.

Paper details used here:

- ISIC-LT:
  - 8 classes.
  - Long-tail train splits with imbalance factors `0.01`, `0.005`, `0.002`, corresponding to 1:100, 1:200, 1:500.
  - Validation and test splits are balanced with 50 and 100 images per class.
  - FoPro-KD public repository provides 5 sampled train/val/test split CSVs.
- HyperKvasir:
  - 23 classes and 10,662 gastrointestinal tract images.
  - Original long-tailed distribution is used.
  - Because the official test set covers only 12 classes, FoPro-KD follows BalMixUp and uses stratified 5-fold cross-validation.
  - Head / Medium / Tail groups follow the paper thresholds: Head `>700`, Tail `<70`, Medium in between.
- Shared training setup from the paper:
  - Images resized to 224x224.
  - Random crop and horizontal flip augmentation.
  - ResNet-18 target model and batch size 32 in the paper. This repo still uses the existing ResNet32 unless the model code is changed separately.

## Local Data Roots

Validated local HyperKvasir root:

```text
/Volumes/DataP/CL_medical_classification/hyper-kvasir23
```

Current DataP state:

- `hyper-kvasir23` exists and previews correctly.
- A FoPro-KD-compatible extracted ISIC 2019 image root was not found.
- `/Volumes/DataP/CL_medical_classification/ISIC 24_33` exists, but it is HDF5/metadata and does not match FoPro-KD `ISIC_2019_Training_Input` image-file layout.

## Added CLI

New datasets:

```bash
--dataset hyper_kvasir23
--dataset isic2019_lt
```

Medical-specific options:

```bash
--medical-fold 1
--medical-imb-factor 0.01
--medical-split-root src/datasets/fopro_splits/ham2019
--medical-label-column finding
--medical-eval-split test
--medical-image-size 224
--medical-normalization imagenet
```

For ISIC, `--medical-imb-factor` supports:

| FoPro split suffix | imbalance ratio |
|---:|---:|
| `0.01` | 1:100 |
| `0.005` | 1:200 |
| `0.002` | 1:500 |

## Preview Commands

HyperKvasir:

```bash
python3 train.py \
  --dataset hyper_kvasir23 \
  --data-root "/Volumes/DataP/CL_medical_classification/hyper-kvasir23" \
  --preview-dataset \
  --order ordered
```

ISIC 2019 after data preparation:

```bash
python3 train.py \
  --dataset isic2019_lt \
  --data-root "/Volumes/DataP/CL_medical_classification/ISIC2019_FoProKD/ISIC_2019_Training_Input" \
  --medical-imb-factor 0.01 \
  --medical-fold 1 \
  --preview-dataset
```

## Data Preparation

Prepare ISIC 2019 FoPro-KD layout on DataP:

```bash
bash scripts/prepare_isic2019_fopro.sh
```

Default target:

```text
/Volumes/DataP/CL_medical_classification/ISIC2019_FoProKD
```

The script:

1. Copies bundled FoPro-KD split CSVs to `ham2019/`.
2. Downloads `ISIC_2019_Training_Input.zip` from the ISIC 2019 challenge S3 URL.
3. Extracts to `ISIC_2019_Training_Input/`.
4. Prints image and split counts.

## HyperKvasir Preview Result

The local DataP preview produced:

| field | value |
|---|---:|
| num_classes | 23 |
| train_samples | 8519 |
| test_samples | 2143 |
| fold | 1 / 5 |
| max train count | 918 |
| min train count | 4 |
| image_size | 224 |
| frequency thresholds | many > 700, few < 70 |

Class counts from fold 1 train split:

```text
barretts=32
barretts-short-segment=42
bbps-0-1=516
bbps-2-3=918
cecum=807
dyed-lifted-polyps=801
dyed-resection-margins=791
esophagitis-a=322
esophagitis-b-d=208
hemorrhoids=4
ileum=7
impacted-stool=104
polyps=822
pylorus=799
retroflex-rectum=312
retroflex-stomach=611
ulcerative-colitis-grade-0-1=28
ulcerative-colitis-grade-1=160
ulcerative-colitis-grade-1-2=8
ulcerative-colitis-grade-2=354
ulcerative-colitis-grade-2-3=22
ulcerative-colitis-grade-3=106
z-line=745
```

## Caveats

- FoPro-KD uses ResNet-18 and medical long-tailed recognition. This repo currently uses the existing expandable ResNet32 and CIL engine. Dataset support is ready, but paper-level FoPro-KD model/training parity is not claimed.
- ISIC split CSVs are bundled from `xmed-lab/FoPro-KD`; ISIC image files are intentionally kept on DataP.
- HyperKvasir split is deterministic stratified 5-fold generated from local class folders because FoPro-KD does not publish HyperKvasir CSV split files in its repository.
