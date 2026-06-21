#!/usr/bin/env python3
"""Entry point for the minimal CIFAR-100-LT Finetune pipeline."""

from __future__ import annotations

import argparse
import os

from src.datasets import DATASET_CHOICES
from src.engine.finetune import preview_dataset, run_finetune
from src.utils.seed import default_device

METHOD_CHOICES = [
    "finetune",
    "finetune_gpa",
    "taconcm_gpa",
    "taconcm_stage1_calib_init_only",
    "taconcm_stage2_calib_anchor",
    "taconcm_stage3_calib_tailanchor",
    "taconcm_stage4_full",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal CIFAR-100-LT class-incremental Finetune runner."
    )
    parser.add_argument("--dataset", default="cifar100_lt", choices=DATASET_CHOICES)
    parser.add_argument(
        "--method",
        default="finetune",
        choices=METHOD_CHOICES,
    )

    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--order", default="shuffled", choices=["ordered", "shuffled"])
    parser.add_argument(
        "--count-assignment",
        "--count_assignment",
        dest="count_assignment",
        choices=["original_id", "shuffled_rank"],
        default="original_id",
        help="Bind long-tail counts by original CIFAR id or by shuffled class-order rank.",
    )
    parser.add_argument(
        "--class-order-source",
        "--class_order_source",
        dest="class_order_source",
        choices=["random_seed", "official_ltcil", "file"],
        default="random_seed",
        help="Class order source. Default preserves seed-based order.",
    )
    parser.add_argument("--class-order-file", "--class_order_file", dest="class_order_file", default=None)
    parser.add_argument(
        "--cifar-normalization",
        "--cifar_normalization",
        dest="cifar_normalization",
        choices=["current", "ltcil_official"],
        default="current",
        help="CIFAR-100 normalization constants. Default preserves current repo behavior.",
    )
    parser.add_argument(
        "--medical-fold",
        "--medical_fold",
        dest="medical_fold",
        type=int,
        default=1,
        help="FoPro-KD medical split fold. ISIC uses provided 5 split CSVs; HyperKvasir uses stratified 5-fold CV.",
    )
    parser.add_argument(
        "--medical-imb-factor",
        "--medical_imb_factor",
        dest="medical_imb_factor",
        type=float,
        default=0.01,
        help="FoPro-KD ISIC-LT imbalance factor: 0.01, 0.005, or 0.002 for 1:100, 1:200, or 1:500.",
    )
    parser.add_argument(
        "--medical-split-root",
        "--medical_split_root",
        dest="medical_split_root",
        default=None,
        help="Directory containing FoPro-KD ham2019 split CSVs. Defaults to bundled src/datasets/fopro_splits/ham2019.",
    )
    parser.add_argument(
        "--medical-label-column",
        "--medical_label_column",
        dest="medical_label_column",
        choices=["finding", "finding_name"],
        default="finding",
        help="FoPro-KD ISIC split label column. Default uses long-tail rank labels from finding.",
    )
    parser.add_argument(
        "--medical-eval-split",
        "--medical_eval_split",
        dest="medical_eval_split",
        choices=["val", "test"],
        default="test",
        help="FoPro-KD ISIC balanced eval split.",
    )
    parser.add_argument(
        "--medical-image-size",
        "--medical_image_size",
        dest="medical_image_size",
        type=int,
        default=224,
        help="Medical image resize/crop size. FoPro-KD uses 224.",
    )
    parser.add_argument(
        "--medical-normalization",
        "--medical_normalization",
        dest="medical_normalization",
        choices=["imagenet", "half"],
        default="imagenet",
        help="Normalization for medical image datasets. Default follows ImageNet-pretrained practice.",
    )
    parser.add_argument("--base-classes", "--base_classes", dest="base_classes", type=int, default=50)
    parser.add_argument(
        "--incremental-steps",
        "--incremental_steps",
        dest="incremental_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--debug-num-classes",
        "--debug_num_classes",
        dest="debug_num_classes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-train-per-class",
        "--max_train_per_class",
        dest="max_train_per_class",
        type=int,
        default=None,
    )
    parser.add_argument("--preview-dataset", "--preview_dataset", action="store_true")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--base-epochs", "--base_epochs", dest="base_epochs", type=int, default=None)
    parser.add_argument("--inc-epochs", "--inc_epochs", dest="inc_epochs", type=int, default=None)
    parser.add_argument("--max-phases", "--max_phases", dest="max_phases", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["constant", "cosine", "multistep"], default="constant")
    parser.add_argument(
        "--milestones",
        default="60,80",
        help="Comma-separated epoch milestones for --scheduler multistep.",
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="LR decay factor for --scheduler multistep.")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--num-exemplars-per-class", "--num_exemplars_per_class", dest="num_exemplars_per_class", type=int, default=0)
    parser.add_argument(
        "--exemplar-selection",
        "--exemplar_selection",
        dest="exemplar_selection",
        choices=["deterministic_random"],
        default="deterministic_random",
    )
    parser.add_argument("--lambda-gpa", "--lambda_gpa", dest="lambda_gpa", type=float, default=0.12)
    parser.add_argument(
        "--gpa-anchor-start-phase",
        "--gpa_anchor_start_phase",
        dest="gpa_anchor_start_phase",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpa-init-bias",
        "--gpa_init_bias",
        dest="gpa_init_bias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gpa-freeze-old-head",
        "--gpa_freeze_old_head",
        dest="gpa_freeze_old_head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Zero old classifier row/bias gradients during GPA incremental phases.",
    )
    parser.add_argument(
        "--gpa-restore-old-head-after-step",
        "--gpa_restore_old_head_after_step",
        dest="gpa_restore_old_head_after_step",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restore old classifier row/bias snapshots after optimizer.step().",
    )
    parser.add_argument(
        "--gpa-anchor-reduction",
        "--gpa_anchor_reduction",
        dest="gpa_anchor_reduction",
        choices=["mean", "sum", "cosine"],
        default="mean",
        help="Anchor loss reduction. Default preserves the previous mean-MSE behavior.",
    )
    parser.add_argument(
        "--gpa-bias-mode",
        "--gpa_bias_mode",
        dest="gpa_bias_mode",
        choices=["paper", "zero", "old_mean", "none"],
        default="paper",
        help="Diagnostic-only GPA new-class bias init mode. Default keeps the current paper formula.",
    )
    parser.add_argument(
        "--diagnose-task-hist",
        "--diagnose_task_hist",
        dest="diagnose_task_hist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record predicted task histograms for all-seen and per-task eval loaders.",
    )
    parser.add_argument(
        "--base-only",
        "--base_only",
        dest="base_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train and evaluate only task_0 for base sanity checks.",
    )
    parser.add_argument(
        "--freeze-backbone-after-base",
        "--freeze_backbone_after_base",
        dest="freeze_backbone_after_base",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--freeze-bn-after-base",
        "--freeze_bn_after_base",
        dest="freeze_bn_after_base",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train-new-head-only-after-base",
        "--train_new_head_only_after_base",
        dest="train_new_head_only_after_base",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-ltconcm",
        "--use_ltconcm",
        dest="use_ltconcm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltconcm-calibrate-prototypes",
        "--ltconcm_calibrate_prototypes",
        dest="ltconcm_calibrate_prototypes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ltconcm-tail-anchor",
        "--ltconcm_tail_anchor",
        dest="ltconcm_tail_anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ltconcm-anchor-target",
        "--ltconcm_anchor_target",
        dest="ltconcm_anchor_target",
        choices=["gpa_raw", "calibrated", "tdsm"],
        default="calibrated",
    )
    parser.add_argument(
        "--ltconcm-anchor-gamma",
        "--ltconcm_anchor_gamma",
        dest="ltconcm_anchor_gamma",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ltconcm-anchor-max-weight",
        "--ltconcm_anchor_max_weight",
        dest="ltconcm_anchor_max_weight",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--ltconcm-memory-topk",
        "--ltconcm_memory_topk",
        dest="ltconcm_memory_topk",
        type=int,
        default=5,
    )
    parser.add_argument("--ltconcm-alpha-a", "--ltconcm_alpha_a", dest="ltconcm_alpha_a", type=float, default=2.0)
    parser.add_argument("--ltconcm-alpha-b", "--ltconcm_alpha_b", dest="ltconcm_alpha_b", type=float, default=1.0)
    parser.add_argument(
        "--ltconcm-alpha-min",
        "--ltconcm_alpha_min",
        dest="ltconcm_alpha_min",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--ltconcm-alpha-max",
        "--ltconcm_alpha_max",
        dest="ltconcm_alpha_max",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--ltconcm-use-tdsm",
        "--ltconcm_use_tdsm",
        dest="ltconcm_use_tdsm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltconcm-use-match-loss",
        "--ltconcm_use_match_loss",
        dest="ltconcm_use_match_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ltconcm-match-lambda",
        "--ltconcm_match_lambda",
        dest="ltconcm_match_lambda",
        type=float,
        default=0.1,
    )

    parser.add_argument("--data-root", "--data_root", dest="data_root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument(
        "--output-root",
        "--output_root",
        "--run-root",
        "--run_root",
        dest="output_root",
        default=os.environ.get("OUTPUT_ROOT", "runs"),
    )
    parser.add_argument("--output", "--output-dir", "--output_dir", dest="output", default=None)
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", default=os.environ.get("XDG_CACHE_HOME"))
    parser.add_argument("--ckpt-dir", "--ckpt_dir", dest="ckpt_dir", default=None)
    parser.add_argument(
        "--use-swanlab",
        "--use_swanlab",
        dest="use_swanlab",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--swanlab-project", "--swanlab_project", dest="swanlab_project", default="LongTailed-CL-TaConCM")
    parser.add_argument("--swanlab-run-name", "--swanlab_run_name", dest="swanlab_run_name", default=None)
    parser.add_argument(
        "--swanlab-mode",
        "--swanlab_mode",
        dest="swanlab_mode",
        choices=["online", "cloud", "offline", "local", "disabled"],
        default="online",
    )
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser


def apply_method_aliases(args: argparse.Namespace) -> argparse.Namespace:
    args.ablation_stage = args.method
    args.uses_gpa = args.method != "finetune"

    medical_class_counts = {
        "hyper_kvasir23": 23,
        "isic2019_lt": 8,
    }
    if args.dataset in medical_class_counts and args.base_classes == 50 and args.incremental_steps == 5:
        # FoPro-KD is a long-tailed recognition protocol, not a class-incremental
        # protocol. Keep CIFAR defaults unchanged, but make medical defaults usable
        # as one all-class base phase unless the caller explicitly provides a CIL split.
        args.base_classes = medical_class_counts[args.dataset]
        args.incremental_steps = 0

    if args.gpa_freeze_old_head is None:
        args.gpa_freeze_old_head = args.method == "finetune_gpa"
    if args.gpa_restore_old_head_after_step is None:
        args.gpa_restore_old_head_after_step = bool(args.gpa_freeze_old_head)

    if args.method == "finetune":
        args.use_ltconcm = False
        args.ltconcm_calibrate_prototypes = False
        args.ltconcm_tail_anchor = False
        args.ltconcm_anchor_target = "gpa_raw"
        args.ltconcm_use_tdsm = False
        args.ltconcm_use_match_loss = False
    elif args.method == "finetune_gpa":
        args.use_ltconcm = False
        args.ltconcm_calibrate_prototypes = False
        args.ltconcm_tail_anchor = False
        args.ltconcm_anchor_target = "gpa_raw"
        args.ltconcm_use_tdsm = False
        args.ltconcm_use_match_loss = False
    elif args.method == "taconcm_stage1_calib_init_only":
        args.use_ltconcm = True
        args.ltconcm_calibrate_prototypes = True
        args.ltconcm_anchor_target = "gpa_raw"
        args.ltconcm_tail_anchor = False
        args.ltconcm_use_tdsm = False
        args.ltconcm_use_match_loss = False
    elif args.method == "taconcm_stage2_calib_anchor":
        args.use_ltconcm = True
        args.ltconcm_calibrate_prototypes = True
        args.ltconcm_anchor_target = "calibrated"
        args.ltconcm_tail_anchor = False
        args.ltconcm_use_tdsm = False
        args.ltconcm_use_match_loss = False
    elif args.method == "taconcm_stage3_calib_tailanchor":
        args.use_ltconcm = True
        args.ltconcm_calibrate_prototypes = True
        args.ltconcm_anchor_target = "calibrated"
        args.ltconcm_tail_anchor = True
        args.ltconcm_use_tdsm = False
        args.ltconcm_use_match_loss = False
    elif args.method == "taconcm_stage4_full":
        args.use_ltconcm = True
        args.ltconcm_calibrate_prototypes = True
        args.ltconcm_anchor_target = "tdsm"
        args.ltconcm_tail_anchor = True
        args.ltconcm_use_tdsm = True
        args.ltconcm_use_match_loss = True
    elif args.method == "taconcm_gpa":
        args.use_ltconcm = True

    if args.ltconcm_anchor_target == "tdsm" and not args.ltconcm_use_tdsm:
        raise ValueError("--ltconcm-anchor-target tdsm requires --ltconcm-use-tdsm")
    if args.gpa_anchor_start_phase < 0:
        raise ValueError("--gpa-anchor-start-phase must be non-negative")
    if args.base_epochs is not None and args.base_epochs <= 0:
        raise ValueError("--base-epochs must be positive")
    if args.inc_epochs is not None and args.inc_epochs <= 0:
        raise ValueError("--inc-epochs must be positive")
    if args.max_phases is not None and args.max_phases <= 0:
        raise ValueError("--max-phases must be positive")
    if args.num_exemplars_per_class < 0:
        raise ValueError("--num-exemplars-per-class must be non-negative")
    return args


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = apply_method_aliases(args)

    if args.preview_dataset:
        preview_dataset(args)
        return

    if args.method in METHOD_CHOICES:
        run_finetune(args)
        return

    raise ValueError(f"Unsupported method: {args.method}")


if __name__ == "__main__":
    main()
