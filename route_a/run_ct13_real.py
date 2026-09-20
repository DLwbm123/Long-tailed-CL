"""Real CT13-ISIC launcher and pre-val engineering gate.

The launcher owns the protocol boundary: it reads train only during the gate,
requires the locked encoders and normal Task1 parent states, and refuses to
fall back to a validator, another CLIP checkpoint, or a test/reserved split.
The external encoder factories are intentionally explicit because APART is a
read-only historical dependency and is not imported by this source release.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from route_a.run_ct13 import validate_config
from shared.frozen_dual_features import label_free_forward, validate_encoder_lock


ORDERS = {
    "1993": [4, 0, 3, 7, 5, 6, 2, 1],
    "1994": [3, 7, 4, 5, 1, 0, 6, 2],
    "1995": [1, 3, 0, 6, 4, 5, 2, 7],
}


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "mapped_label", "split", "relative_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("BLOCKED_MANIFEST_COLUMNS")
    if any(row["split"] != split for row in rows):
        raise ValueError("BLOCKED_MANIFEST_SPLIT")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("BLOCKED_MANIFEST_DUPLICATE_SAMPLE")
    return rows


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _factory(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("BLOCKED_FACTORY_SPEC")
    module, name = spec.split(":", 1)
    fn = getattr(importlib.import_module(module), name, None)
    if not callable(fn):
        raise ValueError("BLOCKED_FACTORY_CALLABLE")
    return fn


def audit_assets(config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit real train/parent assets without opening val, test, or reserved files."""
    data = config.get("data", {})
    parents = config.get("f1_parent_state", {})
    checked: dict[str, Any] = {"train_manifest": 0, "image_missing": [], "parents": {},
                               "clip_search_paths": list(config.get("clip_search_paths", []))}
    errors: list[str] = []
    try:
        validate_encoder_lock(config["clip_lock"])
        checked["clip_lock"] = "PASS"
    except (KeyError, ValueError) as exc:
        errors.append(str(exc))
        checked["clip_lock"] = "BLOCKED"
    train_path = Path(str(data.get("train_manifest", ""))).expanduser()
    val_path = Path(str(data.get("val_manifest", ""))).expanduser()
    image_root = Path(str(data.get("image_root", ""))).expanduser()
    if not train_path.is_file():
        errors.append("BLOCKED_TRAIN_MANIFEST:" + str(train_path))
    if not val_path.is_file():
        errors.append("BLOCKED_VAL_MANIFEST:" + str(val_path))
    if not image_root.is_dir():
        errors.append("BLOCKED_IMAGE_ROOT:" + str(image_root))
    train: list[dict[str, str]] = []
    if train_path.is_file():
        try:
            train = _rows(train_path, "train")
            checked["train_manifest"] = len(train)
        except ValueError as exc:
            errors.append(str(exc))
    if image_root.is_dir() and train:
        checked["image_missing"] = [
            row["relative_path"] for row in train if not (image_root / row["relative_path"]).is_file()
        ][:20]
        if checked["image_missing"]:
            errors.append("BLOCKED_TRAIN_IMAGE:" + str(len(checked["image_missing"])))
    for seed in (1993, 1994, 1995):
        path = Path(str(parents.get(str(seed), ""))).expanduser()
        meta = Path(str(config.get("f1_parent_metadata", {}).get(str(seed), ""))).expanduser()
        checked["parents"][str(seed)] = {"path": str(path), "exists": path.is_file(),
                                          "metadata": str(meta), "metadata_exists": meta.is_file()}
        if not path.is_file():
            errors.append("BLOCKED_F1_PARENT:" + str(seed))
        if not meta.is_file():
            errors.append("BLOCKED_F1_PARENT_LOCK:" + str(seed))
    names = config.get("class_names")
    if not isinstance(names, Mapping) or set(map(str, range(8))) != set(names):
        errors.append("BLOCKED_CLASS_NAME_MAP")
    return {
        "status": "PASS" if not errors else "BLOCKED",
        "errors": errors,
        "checked": checked,
        "val_manifest_checked_for_existence_only": val_path.is_file(),
        "test_access": 0,
        "reserved_access": 0,
    }


def _qualification(config: Mapping[str, Any], train: list[dict[str, str]], output: Path) -> dict[str, Any]:
    """Run one fixed train-only batch per requested seed; no val selection."""
    factories = config.get("factories", {})
    apart = _factory(str(factories.get("apart")))
    clip = _factory(str(factories.get("clip")))
    data = config["data"]
    image_root = Path(str(data["image_root"])).expanduser()
    # Factories receive the locked parent path and must return (preprocess, forward).
    rows_by_seed = {str(seed): sorted(train, key=lambda row: row["sample_id"])[:8]
                    for seed in config["seeds"]}
    receipts = []
    for seed in config["seeds"]:
        parent = Path(str(config["f1_parent_state"][str(seed)])).expanduser()
        a_pre, a_forward = apart(checkpoint=parent, lock=config["f1_parent_lock"])
        c_pre, c_forward = clip(lock=config["clip_lock"])
        images = [c_pre(image_root / row["relative_path"]) for row in rows_by_seed[str(seed)]]
        batch = c_pre.stack(images) if hasattr(c_pre, "stack") else images
        features = label_free_forward(batch, a_forward, c_forward)
        if features.a.shape[1] != 1536 or features.u.shape[1] != 512:
            raise ValueError("BLOCKED_CT13_FEATURE_DIM")
        receipts.append({"seed": seed, "n": len(rows_by_seed[str(seed)]), "feature_dim": [1536, 512, 2048]})
    _write(output / "QUALIFICATION_RECEIPTS.json", {"status": "PASS", "receipts": receipts,
                                                       "val_access": 0, "test_access": 0})
    return {"status": "PASS", "receipts": receipts}


def execute(config: Mapping[str, Any], output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    _write(output / "PROTOCOL_LOCK.json", {
        "base_commit": config.get("base_commit"),
        "implementation_commit": config.get("implementation_commit"),
        "route": "A_RASP", "dataset": "isic", "tasks": [1, 2, 3, 4],
        "seeds": [1993, 1994, 1995], "task_sizes": [2, 2, 2, 2],
        "readout": "A6", "new_training_epochs": 0, "optimizer_steps": 0,
        "test_access": 0, "reserved_access": 0,
    })
    try:
        validate_config(dict(config), require_assets=False)
    except ValueError as exc:
        _write(output / "ENGINEERING_GATE.json", {"status": "BLOCKED", "reason": str(exc),
                                                      "val_access": 0, "test_access": 0,
                                                      "elapsed_s": time.time() - start})
        _write(output / "BLOCKED.json", {"status": "BLOCKED_PROTOCOL", "errors": [str(exc)],
                                          "val_access": 0, "test_access": 0, "reserved_access": 0})
        return 1
    audit = audit_assets(config)
    _write(output / "ASSET_AUDIT.json", audit)
    if audit["status"] != "PASS":
        _write(output / "ENGINEERING_GATE.json", {"status": "BLOCKED", "reason": audit["errors"],
                                                      "val_access": 0, "test_access": 0,
                                                      "elapsed_s": time.time() - start})
        _write(output / "BLOCKED.json", {"status": "BLOCKED_ASSETS", "errors": audit["errors"],
                                          "checked": audit["checked"], "minimum_to_resume": [
                                              "Provide the exact locked laion400m_e32 weights and digest",
                                              "Provide the locked preprocess/tokenizer files and digests",
                                              "Keep the three F1 parent states readable",
                                          ], "val_access": 0, "test_access": 0, "reserved_access": 0})
        return 2
    try:
        train = _rows(Path(str(config["data"]["train_manifest"])).expanduser(), "train")
        qualification = _qualification(config, train, output)
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        _write(output / "ENGINEERING_GATE.json", {"status": "BLOCKED", "reason": str(exc),
                                                      "val_access": 0, "test_access": 0,
                                                      "elapsed_s": time.time() - start})
        _write(output / "BLOCKED.json", {"status": "BLOCKED_QUALIFICATION", "errors": [str(exc)],
                                          "minimum_to_resume": ["Supply read-only APART and CLIP factories"],
                                          "val_access": 0, "test_access": 0, "reserved_access": 0})
        return 3
    _write(output / "ENGINEERING_GATE.json", {"status": "PASS", "qualification": qualification,
                                                  "val_access": 0, "test_access": 0,
                                                  "elapsed_s": time.time() - start})
    raise RuntimeError("BLOCKED_CT13_MATRIX_ADAPTER: factories must expose the locked text encoder and moment-bank adapter")
