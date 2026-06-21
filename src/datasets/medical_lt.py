"""FoPro-KD style long-tailed medical image protocols."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FOPRO_SPLIT_ROOT = Path(__file__).resolve().parent / "fopro_splits" / "ham2019"

MEDICAL_NORMALIZATIONS = {
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "half": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
}


def _normalization_values(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if name not in MEDICAL_NORMALIZATIONS:
        raise ValueError("--medical-normalization must be imagenet or half")
    values = MEDICAL_NORMALIZATIONS[name]
    return values["mean"], values["std"]


def medical_train_transform(image_size: int = 224, normalization: str = "imagenet") -> transforms.Compose:
    mean, std = _normalization_values(normalization)
    resize_size = int(image_size) + 32 if int(image_size) >= 64 else int(image_size)
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(int(image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def medical_eval_transform(image_size: int = 224, normalization: str = "imagenet") -> transforms.Compose:
    mean, std = _normalization_values(normalization)
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    target: int


class MedicalImageSubset(Dataset):
    def __init__(self, records: Sequence[ImageRecord], indices: Sequence[int], transform: transforms.Compose):
        self.records = list(records)
        self.indices = [int(index) for index in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        record = self.records[self.indices[item]]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(record.target)


@dataclass
class MedicalLTProtocol:
    dataset_name: str
    protocol_source: str
    data_root: Path
    train_records: List[ImageRecord]
    test_records: List[ImageRecord]
    selected_classes: List[int]
    class_names: Dict[int, str]
    class_counts: Dict[int, int]
    class_order: List[int]
    tasks: List[List[int]]
    train_indices_by_class: Dict[int, List[int]]
    test_indices_by_class: Dict[int, List[int]]
    train_transform: transforms.Compose
    eval_transform: transforms.Compose
    seed: int
    rho: float | None
    order: str
    count_assignment: str
    class_order_source: str
    class_order_file: str | None
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    image_size: int
    max_train_per_class: int | None
    split_metadata: Dict[str, object]
    frequency_many_threshold: int = 100
    frequency_few_threshold: int = 20

    def train_dataset_for_classes(self, class_ids: Sequence[int]) -> MedicalImageSubset:
        indices: List[int] = []
        for class_id in class_ids:
            indices.extend(self.train_indices_by_class[int(class_id)])
        return MedicalImageSubset(self.train_records, indices, self.train_transform)

    def train_dataset_for_indices(self, indices: Sequence[int]) -> MedicalImageSubset:
        return MedicalImageSubset(self.train_records, indices, self.train_transform)

    def prototype_dataset_for_classes(self, class_ids: Sequence[int]) -> MedicalImageSubset:
        indices: List[int] = []
        for class_id in class_ids:
            indices.extend(self.train_indices_by_class[int(class_id)])
        return MedicalImageSubset(self.train_records, indices, self.eval_transform)

    def test_dataset_for_classes(self, class_ids: Sequence[int]) -> MedicalImageSubset:
        indices: List[int] = []
        for class_id in class_ids:
            indices.extend(self.test_indices_by_class[int(class_id)])
        return MedicalImageSubset(self.test_records, indices, self.eval_transform)

    def preview(self) -> Dict[str, object]:
        task_sizes = [len(task) for task in self.tasks]
        counts = [self.class_counts[class_id] for class_id in self.selected_classes]
        return {
            "dataset": self.dataset_name,
            "protocol_source": self.protocol_source,
            "data_root": str(self.data_root),
            "num_classes": len(self.selected_classes),
            "rho": self.rho,
            "order": self.order,
            "count_assignment": self.count_assignment,
            "class_order_source": self.class_order_source,
            "class_order_file": self.class_order_file,
            "seed": self.seed,
            "image_size": self.image_size,
            "max_count": max(counts) if counts else 0,
            "min_count": min(counts) if counts else 0,
            "base_classes": len(self.tasks[0]) if self.tasks else 0,
            "incremental_steps": len(self.tasks) - 1,
            "incremental_classes_per_step": task_sizes[1:],
            "task_sizes": task_sizes,
            "class_order": self.class_order,
            "class_names": {str(k): v for k, v in sorted(self.class_names.items())},
            "class_counts": {str(k): v for k, v in sorted(self.class_counts.items())},
            "train_samples": len(self.train_records),
            "test_samples": len(self.test_records),
            "split_metadata": self.split_metadata,
            "frequency_many_threshold": self.frequency_many_threshold,
            "frequency_few_threshold": self.frequency_few_threshold,
        }


def _load_class_order_file(path: str | Path) -> List[int]:
    text = Path(path).read_text(encoding="utf-8")
    return [int(item) for item in re.findall(r"-?\d+", text)]


def _resolve_class_order(
    selected_classes: Sequence[int],
    order: str,
    class_order_source: str,
    class_order_file: str | None,
    rng: np.random.Generator,
) -> List[int]:
    selected_classes = [int(class_id) for class_id in selected_classes]
    if class_order_source == "random_seed":
        if order == "ordered":
            return selected_classes.copy()
        if order == "shuffled":
            return rng.permutation(selected_classes).astype(int).tolist()
        raise ValueError("--order must be either ordered or shuffled")
    if class_order_source == "file":
        if not class_order_file:
            raise ValueError("--class-order-source file requires --class-order-file")
        class_order = _load_class_order_file(class_order_file)
    else:
        raise ValueError("medical datasets support --class-order-source random_seed or file")

    if len(class_order) != len(selected_classes):
        raise ValueError(
            f"class order length {len(class_order)} does not match selected class count {len(selected_classes)}"
        )
    if set(int(class_id) for class_id in class_order) != set(selected_classes):
        raise ValueError("class order must contain each selected class exactly once")
    return [int(class_id) for class_id in class_order]


def _resolve_tasks(class_order: Sequence[int], base_classes: int, incremental_steps: int) -> List[List[int]]:
    total_classes = len(class_order)
    if incremental_steps == 0:
        if base_classes != total_classes:
            raise ValueError("--incremental-steps 0 requires --base-classes to equal the selected class count")
        return [[int(class_id) for class_id in class_order]]
    if incremental_steps < 0:
        raise ValueError("--incremental-steps must be non-negative")
    if base_classes <= 0 or base_classes >= total_classes:
        raise ValueError("--base-classes must be positive and smaller than the selected class count")
    remaining_classes = total_classes - base_classes
    if remaining_classes % incremental_steps != 0:
        raise ValueError(
            "remaining classes must divide evenly by --incremental-steps "
            f"({remaining_classes} % {incremental_steps} != 0)"
        )
    step_size = remaining_classes // incremental_steps
    tasks = [[int(class_id) for class_id in class_order[:base_classes]]]
    offset = base_classes
    for _ in range(incremental_steps):
        tasks.append([int(class_id) for class_id in class_order[offset : offset + step_size]])
        offset += step_size
    return tasks


def _class_indices(records: Sequence[ImageRecord], classes: Sequence[int]) -> Dict[int, List[int]]:
    indices_by_class: Dict[int, List[int]] = {int(class_id): [] for class_id in classes}
    for index, record in enumerate(records):
        indices_by_class[int(record.target)].append(int(index))
    return indices_by_class


def _apply_max_train_per_class(
    records_by_class: Dict[int, List[Path]],
    max_train_per_class: int | None,
) -> Dict[int, List[Path]]:
    if max_train_per_class is None:
        return records_by_class
    limit = int(max_train_per_class)
    if limit <= 0:
        raise ValueError("--max-train-per-class must be positive when provided")
    return {class_id: paths[: min(limit, len(paths))] for class_id, paths in records_by_class.items()}


def _records_from_paths(records_by_class: Dict[int, List[Path]]) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for class_id in sorted(records_by_class):
        for path in records_by_class[class_id]:
            records.append(ImageRecord(path=Path(path), target=int(class_id)))
    return records


def _image_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("._")
    )


def _build_image_id_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in _image_files(root):
        index.setdefault(path.stem, path)
    return index


def _load_fopro_split_rows(path: Path, label_column: str) -> List[tuple[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"FoPro split file not found: {path}")
    rows: List[tuple[str, int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "image" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain an image column")
        if label_column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain label column {label_column!r}")
        for row in reader:
            rows.append((str(row["image"]), int(row[label_column])))
    return rows


def _fopro_suffix(imb_factor: float) -> str:
    known = {
        0.01: "0.01",
        0.005: "0.005",
        0.002: "0.002",
    }
    for value, suffix in known.items():
        if abs(float(imb_factor) - value) < 1e-12:
            return suffix
    raise ValueError("--medical-imb-factor must be one of 0.01, 0.005, or 0.002 for isic2019_lt")


def build_isic2019_lt_protocol(
    data_root: str | Path,
    order: str,
    base_classes: int,
    incremental_steps: int,
    seed: int,
    fold: int = 1,
    imb_factor: float = 0.01,
    split_root: str | Path | None = None,
    label_column: str = "finding",
    eval_split: str = "test",
    image_size: int = 224,
    normalization: str = "imagenet",
    debug_num_classes: int | None = None,
    max_train_per_class: int | None = None,
    class_order_source: str = "random_seed",
    class_order_file: str | None = None,
) -> MedicalLTProtocol:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"ISIC data root does not exist: {data_root}. "
            "Download ISIC_2019_Training_Input.zip or point --data-root to extracted ISIC images."
        )
    if eval_split not in {"val", "test"}:
        raise ValueError("--medical-eval-split must be val or test")
    fold = int(fold)
    if not (1 <= fold <= 5):
        raise ValueError("--medical-fold must be between 1 and 5 for FoPro ISIC splits")
    split_root = Path(split_root) if split_root is not None else FOPRO_SPLIT_ROOT
    suffix = _fopro_suffix(imb_factor)
    train_rows = _load_fopro_split_rows(split_root / f"train_skin{fold}_{suffix}.csv", label_column)
    test_rows = _load_fopro_split_rows(split_root / f"{eval_split}_skin{fold}_{suffix}.csv", label_column)

    image_index = _build_image_id_index(data_root)
    missing = sorted({image_id for image_id, _ in train_rows + test_rows if image_id not in image_index})
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} ISIC images under {data_root}; examples: {preview}. "
            "Expected extracted ISIC_2019_Training_Input images named like ISIC_0000000.jpg."
        )

    classes = sorted({label for _, label in train_rows + test_rows})
    if debug_num_classes is not None:
        classes = classes[: int(debug_num_classes)]
    selected = set(classes)
    train_by_class: Dict[int, List[Path]] = {class_id: [] for class_id in classes}
    test_by_class: Dict[int, List[Path]] = {class_id: [] for class_id in classes}
    for image_id, label in train_rows:
        if label in selected:
            train_by_class[int(label)].append(image_index[image_id])
    for image_id, label in test_rows:
        if label in selected:
            test_by_class[int(label)].append(image_index[image_id])
    train_by_class = _apply_max_train_per_class(train_by_class, max_train_per_class)

    rng = np.random.default_rng(seed)
    class_order = _resolve_class_order(classes, order, class_order_source, class_order_file, rng)
    tasks = _resolve_tasks(class_order, int(base_classes), int(incremental_steps))
    train_records = _records_from_paths(train_by_class)
    test_records = _records_from_paths(test_by_class)
    normalization_mean, normalization_std = _normalization_values(normalization)
    train_transform = medical_train_transform(image_size=image_size, normalization=normalization)
    eval_transform = medical_eval_transform(image_size=image_size, normalization=normalization)
    class_counts = {class_id: len(train_by_class[class_id]) for class_id in classes}
    return MedicalLTProtocol(
        dataset_name="isic2019_lt",
        protocol_source="FoPro-KD ISIC-19 split CSVs",
        data_root=data_root,
        train_records=train_records,
        test_records=test_records,
        selected_classes=[int(class_id) for class_id in classes],
        class_names={int(class_id): f"isic_class_{class_id}" for class_id in classes},
        class_counts=class_counts,
        class_order=class_order,
        tasks=tasks,
        train_indices_by_class=_class_indices(train_records, classes),
        test_indices_by_class=_class_indices(test_records, classes),
        train_transform=train_transform,
        eval_transform=eval_transform,
        seed=int(seed),
        rho=float(imb_factor),
        order=order,
        count_assignment="fopro_split_csv",
        class_order_source=class_order_source,
        class_order_file=class_order_file,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        image_size=int(image_size),
        max_train_per_class=max_train_per_class,
        split_metadata={
            "fold": fold,
            "imb_factor": float(imb_factor),
            "train_split": str(split_root / f"train_skin{fold}_{suffix}.csv"),
            "eval_split": str(split_root / f"{eval_split}_skin{fold}_{suffix}.csv"),
            "eval_split_name": eval_split,
            "label_column": label_column,
            "paper_validation_per_class": 50,
            "paper_test_per_class": 100,
        },
    )


def _discover_hyperkvasir_classes(data_root: Path) -> tuple[Dict[int, str], Dict[int, List[Path]]]:
    class_to_paths: Dict[str, List[Path]] = {}
    for path in _image_files(data_root):
        class_name = path.parent.name
        class_to_paths.setdefault(class_name, []).append(path)
    if not class_to_paths:
        raise FileNotFoundError(f"No image files found under HyperKvasir root: {data_root}")
    class_names_sorted = sorted(class_to_paths)
    class_names = {class_id: class_name for class_id, class_name in enumerate(class_names_sorted)}
    records_by_class = {
        class_id: sorted(class_to_paths[class_name])
        for class_id, class_name in class_names.items()
    }
    return class_names, records_by_class


def _hyperkvasir_stratified_fold(
    records_by_class: Dict[int, List[Path]],
    fold: int,
    seed: int,
    num_folds: int = 5,
) -> tuple[Dict[int, List[Path]], Dict[int, List[Path]]]:
    if not (1 <= int(fold) <= int(num_folds)):
        raise ValueError("--medical-fold must be between 1 and 5 for HyperKvasir stratified CV")
    train_by_class: Dict[int, List[Path]] = {}
    test_by_class: Dict[int, List[Path]] = {}
    fold_index = int(fold) - 1
    for class_id, paths in records_by_class.items():
        rng = np.random.default_rng(int(seed) + 7919 * int(class_id))
        shuffled = np.array(sorted(paths), dtype=object)
        rng.shuffle(shuffled)
        chunks = np.array_split(shuffled, int(num_folds))
        test_paths = [Path(path) for path in chunks[fold_index].tolist()]
        train_paths = [
            Path(path)
            for idx, chunk in enumerate(chunks)
            if idx != fold_index
            for path in chunk.tolist()
        ]
        train_by_class[int(class_id)] = sorted(train_paths)
        test_by_class[int(class_id)] = sorted(test_paths)
    return train_by_class, test_by_class


def build_hyperkvasir23_protocol(
    data_root: str | Path,
    order: str,
    base_classes: int,
    incremental_steps: int,
    seed: int,
    fold: int = 1,
    image_size: int = 224,
    normalization: str = "imagenet",
    debug_num_classes: int | None = None,
    max_train_per_class: int | None = None,
    class_order_source: str = "random_seed",
    class_order_file: str | None = None,
) -> MedicalLTProtocol:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"HyperKvasir data root does not exist: {data_root}")
    class_names, all_by_class = _discover_hyperkvasir_classes(data_root)
    classes = sorted(class_names)
    if debug_num_classes is not None:
        classes = classes[: int(debug_num_classes)]
        all_by_class = {class_id: all_by_class[class_id] for class_id in classes}
        class_names = {class_id: class_names[class_id] for class_id in classes}

    train_by_class, test_by_class = _hyperkvasir_stratified_fold(all_by_class, fold=fold, seed=seed)
    train_by_class = _apply_max_train_per_class(train_by_class, max_train_per_class)
    rng = np.random.default_rng(seed)
    class_order = _resolve_class_order(classes, order, class_order_source, class_order_file, rng)
    tasks = _resolve_tasks(class_order, int(base_classes), int(incremental_steps))
    train_records = _records_from_paths(train_by_class)
    test_records = _records_from_paths(test_by_class)
    normalization_mean, normalization_std = _normalization_values(normalization)
    train_transform = medical_train_transform(image_size=image_size, normalization=normalization)
    eval_transform = medical_eval_transform(image_size=image_size, normalization=normalization)
    class_counts = {class_id: len(train_by_class[class_id]) for class_id in classes}
    return MedicalLTProtocol(
        dataset_name="hyper_kvasir23",
        protocol_source="FoPro-KD HyperKvasir original long-tail + stratified 5-fold CV",
        data_root=data_root,
        train_records=train_records,
        test_records=test_records,
        selected_classes=[int(class_id) for class_id in classes],
        class_names={int(k): v for k, v in class_names.items()},
        class_counts=class_counts,
        class_order=class_order,
        tasks=tasks,
        train_indices_by_class=_class_indices(train_records, classes),
        test_indices_by_class=_class_indices(test_records, classes),
        train_transform=train_transform,
        eval_transform=eval_transform,
        seed=int(seed),
        rho=None,
        order=order,
        count_assignment="original_hyperkvasir_counts",
        class_order_source=class_order_source,
        class_order_file=class_order_file,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        image_size=int(image_size),
        max_train_per_class=max_train_per_class,
        split_metadata={
            "fold": int(fold),
            "num_folds": 5,
            "paper_protocol_note": "FoPro-KD follows BalMixUp stratified 5-fold CV because the official test set has 12 classes.",
            "paper_head_threshold": ">700",
            "paper_tail_threshold": "<70",
        },
        frequency_many_threshold=700,
        frequency_few_threshold=70,
    )

