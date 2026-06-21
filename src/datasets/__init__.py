"""Dataset builders for the minimal LT-CIL pipeline."""

from __future__ import annotations

from pathlib import Path

from src.datasets.cifar100_lt import build_cifar100_lt_protocol
from src.datasets.medical_lt import build_hyperkvasir23_protocol, build_isic2019_lt_protocol

DATASET_CHOICES = ["cifar100_lt", "hyper_kvasir23", "isic2019_lt"]


def build_protocol(args):
    """Dispatch CLI args to a dataset protocol with the engine-compatible interface."""

    if args.dataset == "cifar100_lt":
        return build_cifar100_lt_protocol(
            data_root=args.data_root,
            rho=args.rho,
            order=args.order,
            base_classes=args.base_classes,
            incremental_steps=args.incremental_steps,
            seed=args.seed,
            debug_num_classes=args.debug_num_classes,
            max_train_per_class=args.max_train_per_class,
            count_assignment=args.count_assignment,
            class_order_source=args.class_order_source,
            class_order_file=args.class_order_file,
            cifar_normalization=args.cifar_normalization,
            download=args.download,
        )
    if args.dataset == "hyper_kvasir23":
        return build_hyperkvasir23_protocol(
            data_root=Path(args.data_root),
            order=args.order,
            base_classes=args.base_classes,
            incremental_steps=args.incremental_steps,
            seed=args.seed,
            fold=args.medical_fold,
            image_size=args.medical_image_size,
            normalization=args.medical_normalization,
            debug_num_classes=args.debug_num_classes,
            max_train_per_class=args.max_train_per_class,
            class_order_source=args.class_order_source,
            class_order_file=args.class_order_file,
        )
    if args.dataset == "isic2019_lt":
        return build_isic2019_lt_protocol(
            data_root=Path(args.data_root),
            order=args.order,
            base_classes=args.base_classes,
            incremental_steps=args.incremental_steps,
            seed=args.seed,
            fold=args.medical_fold,
            imb_factor=args.medical_imb_factor,
            split_root=args.medical_split_root,
            label_column=args.medical_label_column,
            eval_split=args.medical_eval_split,
            image_size=args.medical_image_size,
            normalization=args.medical_normalization,
            debug_num_classes=args.debug_num_classes,
            max_train_per_class=args.max_train_per_class,
            class_order_source=args.class_order_source,
            class_order_file=args.class_order_file,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")
