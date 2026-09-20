"""Real CT13-ISIC launcher and pre-val engineering gate.

The launcher owns the protocol boundary: it reads train only during the gate,
requires the locked encoders and normal Task1 parent states, and refuses to
fall back to a validator, another CLIP checkpoint, or a test/reserved split.
The external encoder factories are intentionally explicit because APART is a
read-only historical dependency and is not imported by this source release.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from route_a.run_ct13 import validate_config
from route_a.semantic_prior import (component_reliability, semantic_scale, text_prompts,
                                    unit_columns)
from route_a.spectral_prior_ridge import a0_to_a8, readout_scores
from shared.dual_moment_bank import ClassMoments, DualMomentBank
from shared.frozen_dual_features import (_unwrap_pre_logits, build_joint_feature,
                                         validate_encoder_lock)


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


def _bundle_parts(raw: Any, *, clip: bool = False) -> dict[str, Any]:
    """Normalize the explicit factory contract while keeping historical adapters small."""
    if isinstance(raw, Mapping):
        out = dict(raw)
    elif isinstance(raw, tuple):
        if len(raw) != 2:
            raise ValueError("BLOCKED_FACTORY_RETURN")
        out = {"preprocess": raw[0], "forward": raw[1]}
    else:
        raise ValueError("BLOCKED_FACTORY_RETURN")
    if not callable(out.get("preprocess")) or not callable(out.get("forward")):
        raise ValueError("BLOCKED_FACTORY_INTERFACE")
    if clip and (not callable(out.get("tokenizer")) or not callable(out.get("encode_text"))):
        raise ValueError("BLOCKED_CLIP_TEXT_INTERFACE")
    return out


def _stack(values: list[Any]) -> Any:
    if not values:
        raise ValueError("BLOCKED_EMPTY_BATCH")
    try:
        import torch
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values, dim=0)
    except ImportError:
        pass
    return np.stack([np.asarray(v) for v in values], axis=0)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _open_rgb(path: Path) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("BLOCKED_PILLOW") from exc
    with Image.open(path) as image:
        return image.convert("RGB")


def _extract_rows(rows: list[dict[str, str]], image_root: Path,
                  apart: Mapping[str, Any], clip: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the two transforms to the same RGB images, then release raw data."""
    images = [_open_rgb(image_root / row["relative_path"]) for row in rows]
    a_batch = _stack([apart["preprocess"](image) for image in images])
    c_batch = _stack([clip["preprocess"](image) for image in images])
    a_raw = apart["forward"](a_batch)
    c_raw = clip["forward"](c_batch)
    feature = build_joint_feature(_unwrap_pre_logits(a_raw, apart=True), _unwrap_pre_logits(c_raw))
    a, u, h = map(_to_numpy, (feature.a, feature.u, feature.h))
    del images, a_batch, c_batch, a_raw, c_raw, feature
    gc.collect()
    if a.shape[1:] != (1536,) or u.shape[1:] != (512,) or h.shape[1:] != (2048,):
        raise ValueError("BLOCKED_CT13_FEATURE_DIM")
    if not all(np.isfinite(x).all() for x in (a, u, h)):
        raise ValueError("BLOCKED_CT13_NONFINITE_FEATURE")
    return a, u, h


def _save_bank(bank: DualMomentBank, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    S, M, ids = bank.class_balanced()
    rows = [bank.classes[int(c)] for c in ids]
    np.savez_compressed(
        path, S=S, M=M, class_ids=ids,
        tasks=np.asarray([row.task for row in rows], dtype=np.int64),
        ns=np.asarray([row.n for row in rows], dtype=np.int64),
        component_counts=np.asarray([row.component_count for row in rows], dtype=np.int64),
        component_sum=np.stack([row.component_sum for row in rows]),
        component_second=np.stack([row.component_second for row in rows]),
    )


def _load_bank(path: Path) -> DualMomentBank:
    with np.load(path, allow_pickle=False) as z:
        ids, tasks, ns = z["class_ids"], z["tasks"], z["ns"]
        bank = DualMomentBank(dim_a=1536, dim_u=512, _S=np.asarray(z["S"], dtype=np.float64))
        sums, seconds, counts = z["component_sum"], z["component_second"], z["component_counts"]
        means = np.asarray(z["M"], dtype=np.float64)
        for j, cid in enumerate(ids.tolist()):
            bank.classes[int(cid)] = ClassMoments(
                int(cid), int(tasks[j]), int(ns[j]), means[:, j].copy(), int(counts[j]),
                sums[j].copy(), seconds[j].copy(),
            )
            bank.class_order.append(int(cid))
    return bank


def _save_readouts(readouts: Mapping[str, Any], path: Path, *, text_u: np.ndarray,
                   a_scale: float, gamma: np.ndarray, ncomp: np.ndarray) -> None:
    arrays = {key: value for key, value in readouts.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays)
    _write(path.with_suffix(".json"), {
        "readouts": sorted(readouts), "text_u": text_u.tolist(), "a_scale": float(a_scale),
        "gamma": np.asarray(gamma).tolist(), "ncomp": np.asarray(ncomp).tolist(),
    })


def _load_readouts(path: Path) -> dict[str, Any]:
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as z:
        out = {key: np.asarray(z[key]) for key in z.files}
    from route_a.spectral_prior_ridge import RASPReadout
    out["A3"] = RASPReadout(out["A2"], np.asarray(meta["text_u"]), 0.0, centered=False)
    out["A3s"] = RASPReadout(out["A2"], np.asarray(meta["text_u"]), float(meta["a_scale"]), centered=True)
    return out


def _text_bundle(clip: Mapping[str, Any], names: list[str]) -> np.ndarray:
    prompts = text_prompts("isic", names, 0) + text_prompts("isic", names, 1)
    tokens = clip["tokenizer"](prompts)
    raw = _to_numpy(clip["encode_text"](tokens))
    if raw.shape != (2 * len(names), 512):
        raise ValueError("BLOCKED_TEXT_FEATURE_DIM")
    return unit_columns(((raw[:len(names)] + raw[len(names):]) / 2.0).T)


def _stage_metrics(scores: np.ndarray, labels: np.ndarray, seen: list[int], current: list[int],
                   tail: set[int]) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    original_labels = np.asarray(labels, dtype=np.int64)
    keep = np.isin(labels, np.asarray(seen))
    labels = labels[keep]
    scores = scores[keep]
    if not len(labels):
        raise ValueError("BLOCKED_EMPTY_VAL_STAGE")
    pred = np.asarray(seen, dtype=np.int64)[np.argmax(scores, axis=1)]
    predictions = np.full(original_labels.shape, -1, dtype=np.int64)
    predictions[keep] = pred
    class_rows = []
    recalls = []
    for cid in seen:
        mask = labels == cid
        n = int(mask.sum())
        tp = int((pred[mask] == cid).sum()) if n else 0
        recall = tp / n if n else float("nan")
        recalls.append(recall)
        class_rows.append({"class_id": int(cid), "n": n, "recall": recall,
                           "zero_recall": bool(n and tp == 0)})
    def mean_for(ids: list[int] | set[int]) -> float:
        vals = [row["recall"] for row in class_rows if row["class_id"] in ids and np.isfinite(row["recall"])]
        return float(np.mean(vals)) if vals else float("nan")
    old = [c for c in seen if c not in current]
    old_ba, current_ba, tail_ba = mean_for(old), mean_for(current), mean_for(tail)
    hm = float(2 * old_ba * current_ba / (old_ba + current_ba)) if old_ba + current_ba else 0.0
    stage = {"ba": float(np.mean([x for x in recalls if np.isfinite(x)])),
             "accuracy": float(np.mean(pred == labels)),
             "macro_f1": float(np.mean([row["recall"] for row in class_rows if np.isfinite(row["recall"])])),
             "old_ba": old_ba, "current_ba": current_ba, "hm": hm, "tail_ba": tail_ba,
             "zero_recall": int(sum(row["zero_recall"] for row in class_rows))}
    return stage, class_rows, predictions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("BLOCKED_EMPTY_REPORT")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    semantic_raw = str(config.get("semantic_map_path", "")).strip()
    semantic_path = Path(semantic_raw).expanduser() if semantic_raw else Path("<required-semantic-map>")
    checked["semantic_map"] = str(semantic_path)
    if not semantic_path.is_file():
        errors.append("BLOCKED_SEMANTIC_MAP:" + str(semantic_path))
    return {
        "status": "PASS" if not errors else "BLOCKED",
        "errors": errors,
        "checked": checked,
        "val_manifest_checked_for_existence_only": val_path.is_file(),
        "test_access": 0,
        "reserved_access": 0,
    }


def verify_semantics(config: Mapping[str, Any], train: list[dict[str, str]], output: Path) -> dict[str, Any]:
    """Verify an independently sourced train-only mapped-label/name ledger."""
    path = Path(str(config.get("semantic_map_path", ""))).expanduser()
    if not path.is_file():
        raise ValueError("BLOCKED_SEMANTIC_MAP:" + str(path))
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("BLOCKED_SEMANTIC_MAP_READ") from exc
    mapping = evidence.get("mapping")
    if not isinstance(mapping, Mapping) or set(map(str, range(8))) != set(mapping):
        raise ValueError("BLOCKED_SEMANTIC_MAP_COVERAGE")
    names = {}
    diagnoses = set()
    for key, value in mapping.items():
        if not isinstance(value, Mapping):
            raise ValueError("BLOCKED_SEMANTIC_MAP_ENTRY")
        name, diagnosis = str(value.get("class_name", "")).strip(), str(value.get("diagnosis_code", "")).strip()
        if not name or not diagnosis or diagnosis in diagnoses:
            raise ValueError("BLOCKED_SEMANTIC_MAP_CONFLICT")
        names[int(key)] = name
        diagnoses.add(diagnosis)
    counts = {cid: 0 for cid in range(8)}
    for row in train:
        cid = int(row["mapped_label"])
        if cid not in names:
            raise ValueError("BLOCKED_SEMANTIC_MAP_TRAIN_LABEL")
        counts[cid] += 1
    if sum(counts.values()) != len(train) or any(counts[cid] == 0 for cid in counts):
        raise ValueError("BLOCKED_SEMANTIC_MAP_TRAIN_COVERAGE")
    result = {"status": "VERIFIED_BY_TRAIN_PROVENANCE", "mapping": mapping,
              "train_rows": len(train), "train_counts": counts,
              "source_sha256": _digest(path), "private_sample_ids_exported": False,
              "val_access": 0, "test_access": 0, "reserved_access": 0}
    _write(output / "SEMANTIC_MAP_VERIFICATION.json", result)
    return result


def _qualification(config: Mapping[str, Any], train: list[dict[str, str]], output: Path) -> dict[str, Any]:
    """Run one fixed train-only batch per requested seed; no val selection."""
    factories = config.get("factories", {})
    apart = _factory(str(factories.get("apart")))
    clip = _factory(str(factories.get("clip")))
    data = config["data"]
    image_root = Path(str(data["image_root"])).expanduser()
    # Factories receive the locked parent path and return separate APART/CLIP transforms.
    receipts = []
    for seed in config["seeds"]:
        classes = ORDERS[str(seed)][:2]
        rows_by_class = {
            cid: sorted((row for row in train if int(row["mapped_label"]) == cid),
                        key=lambda row: row["sample_id"])[:2]
            for cid in classes
        }
        if any(len(rows_by_class[cid]) < 2 for cid in classes):
            raise ValueError("BLOCKED_QUALIFICATION_TASK1_CLASS_SUPPORT")
        rows = [row for cid in classes for row in rows_by_class[cid]]
        parent = Path(str(config["f1_parent_state"][str(seed)])).expanduser()
        a_bundle = _bundle_parts(apart(checkpoint=parent, lock=config["f1_parent_lock"]), clip=False)
        c_bundle = _bundle_parts(clip(lock=config["clip_lock"]), clip=True)
        _, _, h = _extract_rows(rows, image_root, a_bundle, c_bundle)
        features = h
        if features.shape[1] != 2048 or not np.isfinite(features).all():
            raise ValueError("BLOCKED_CT13_FEATURE_DIM")
        receipts.append({"seed": seed, "classes": classes, "sample_ids": [row["sample_id"] for row in rows],
                         "n": len(rows), "feature_dim": [1536, 512, 2048],
                         "separate_preprocess": True, "labels_passed_to_forward": False})
    _write(output / "QUALIFICATION_RECEIPTS.json", {"status": "PASS", "receipts": receipts,
                                                       "val_access": 0, "test_access": 0})
    return {"status": "PASS", "receipts": receipts}


def _run_formal_matrix(config: Mapping[str, Any], train: list[dict[str, str]], output: Path,
                       apart_factory: Callable[..., Any], clip_factory: Callable[..., Any]) -> dict[str, Any]:
    """Fit all four frozen stages per seed, seal them, then perform one val pass."""
    data_root = Path(str(config["data"]["image_root"])).expanduser()
    class_names = {int(k): str(v) for k, v in config["class_names"].items()}
    stage_records: list[dict[str, Any]] = []
    stage_objects: list[dict[str, Any]] = []
    resource = {"train_image_reads": 0, "val_image_reads": 0, "historical_fit_rereads": 0,
                "within_run_old_fit_reads": 0, "test_access": 0, "reserved_access": 0}
    for seed in config["seeds"]:
        a_parent = Path(str(config["f1_parent_state"][str(seed)])).expanduser()
        apart = _bundle_parts(apart_factory(checkpoint=a_parent, lock=config["f1_parent_lock"]), clip=False)
        clip = _bundle_parts(clip_factory(lock=config["clip_lock"]), clip=True)
        bank = DualMomentBank(dim_a=1536, dim_u=512)
        seen: list[int] = []
        for task in range(1, 5):
            stage_dir = output / f"seed_{seed}" / f"task_{task}"
            bank_path = stage_dir / "BANK.npz"
            state_path = stage_dir / "STATE_W_LOCK.json"
            readout_path = stage_dir / "READOUTS.npz"
            if bank_path.is_file() and state_path.is_file() and readout_path.is_file():
                lock = json.loads(state_path.read_text(encoding="utf-8"))
                if int(lock.get("seed", -1)) != int(seed) or int(lock.get("task", -1)) != task:
                    raise ValueError("BLOCKED_STAGE_LOCK_MISMATCH")
                bank = _load_bank(bank_path)
                seen = [int(cid) for cid in lock.get("class_order", [])]
                if seen != bank.class_order:
                    raise ValueError("BLOCKED_STAGE_BANK_MISMATCH")
                readouts = _load_readouts(readout_path)
                current = ORDERS[str(seed)][(task - 1) * 2:task * 2]
                stage_objects.append({"seed": seed, "task": task, "seen": seen,
                                      "current": list(current), "readouts": readouts})
                continue
            current = ORDERS[str(seed)][(task - 1) * 2:task * 2]
            for cid in current:
                rows = sorted((row for row in train if int(row["mapped_label"]) == cid),
                              key=lambda row: row["sample_id"])
                if not rows:
                    raise ValueError("BLOCKED_EMPTY_TRAIN_CLASS:" + str(cid))
                _, _, h = _extract_rows(rows, data_root, apart, clip)
                bank.add_class(cid, h, task=task,
                               component_ids=[row.get("identity_component", "default") for row in rows])
                resource["train_image_reads"] += len(rows)
                del h
            seen.extend(current)
            stage_dir.mkdir(parents=True, exist_ok=True)
            bank_path = stage_dir / "BANK.npz"
            _save_bank(bank, bank_path)
            S, M, ids = bank.class_balanced()
            ncomp = np.asarray([bank.classes[int(cid)].component_count for cid in ids], dtype=np.float64)
            G_u, R_u = bank.u_stats()
            text_u = _text_bundle(clip, [class_names[cid] for cid in seen])
            a_scale = semantic_scale(text_u, G_u, R_u)
            comp_stats, comp_counts = bank.component_reliability_inputs()
            _, gamma = component_reliability(text_u, comp_stats, comp_counts,
                                              class_ids=seen, a_scale=a_scale)
            V = np.zeros_like(M)
            V[1536:] = np.sqrt(2.0) * a_scale * (text_u - text_u.mean(axis=1, keepdims=True))
            readouts = a0_to_a8(S, M, text_u, V=V, gamma=gamma, a_scale=a_scale, ncomp=ncomp)
            readout_path = stage_dir / "READOUTS.npz"
            _save_readouts(readouts, readout_path, text_u=text_u, a_scale=a_scale,
                           gamma=gamma, ncomp=ncomp)
            lock = {"seed": seed, "task": task, "class_order": seen, "readouts": list(readouts),
                    "a_scale": float(a_scale), "gamma": gamma.tolist(), "ncomp": ncomp.tolist(),
                    "train_image_reads": resource["train_image_reads"], "test_access": 0,
                    "reserved_access": 0}
            _write(stage_dir / "STATE_W_LOCK.json", lock)
            stage_objects.append({"seed": seed, "task": task, "seen": list(seen),
                                  "current": list(current), "readouts": readouts})
        del apart, clip, bank
        gc.collect()

    # No validation rows are opened until every stage/readout lock exists.
    val_rows = _rows(Path(str(config["data"]["val_manifest"])).expanduser(), "val")
    tail = {6, 7}
    for seed in config["seeds"]:
        a_parent = Path(str(config["f1_parent_state"][str(seed)])).expanduser()
        apart = _bundle_parts(apart_factory(checkpoint=a_parent, lock=config["f1_parent_lock"]), clip=False)
        clip = _bundle_parts(clip_factory(lock=config["clip_lock"]), clip=True)
        _, _, h_val = _extract_rows(val_rows, data_root, apart, clip)
        resource["val_image_reads"] += len(val_rows)
        labels = np.asarray([int(row["mapped_label"]) for row in val_rows], dtype=np.int64)
        prediction_rows: list[np.ndarray] = []
        for item in [x for x in stage_objects if x["seed"] == seed]:
            seen, current = item["seen"], item["current"]
            for name in ("A0", "A1", "A2", "A3", "A3s", "A4", "A5", "A6", "A7", "A8"):
                scores = readout_scores(item["readouts"], name, h_val)
                stage, classes, predictions = _stage_metrics(scores, labels, seen, current, tail)
                prediction_rows.append(predictions)
                stage_records.append({"seed": seed, "task": item["task"], "readout": name,
                                      "seen_classes": len(seen), **stage})
                for row in classes:
                    row.update({"seed": seed, "task": item["task"], "readout": name})
                stage_records[-1]["_class_rows"] = classes
        if len(prediction_rows) != 40:
            raise ValueError("BLOCKED_PREDICTION_ROW_COUNT")
        resource.setdefault("prediction_rows", []).extend(prediction_rows)
        del apart, clip, h_val
        gc.collect()
    class_records = [row for stage in stage_records for row in stage.pop("_class_rows", [])]
    public_stage = [{k: v for k, v in row.items() if k != "class_rows"} for row in stage_records]
    if len(public_stage) != 120 or len(class_records) != 600:
        raise ValueError("BLOCKED_REPORT_ROW_COUNT")
    _write_csv(output / "stage_metrics.csv", public_stage)
    _write_csv(output / "class_metrics.csv", class_records)
    predictions = np.stack(resource.pop("prediction_rows"), axis=0)
    np.savez_compressed(output / "VAL_PREDICTIONS.npz", labels=labels,
                        predictions=predictions,
                        stage_count=np.asarray([predictions.shape[0]]))
    _write(output / "PREDICTIONS_LOCK.json", {"status": "PASS", "stage_rows": len(public_stage),
                                               "class_rows": len(class_records),
                                               "val_image_reads": resource["val_image_reads"],
                                               "test_access": 0, "reserved_access": 0})
    _write(output / "RESOURCE_LEDGER.json", resource)
    return {"stage_rows": len(public_stage), "class_rows": len(class_records), "resource": resource}


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
                                              "Provide the independent train-only semantic map and source digest",
                                              "Keep the three F1 parent states readable",
                                          ], "val_access": 0, "test_access": 0, "reserved_access": 0})
        return 2
    try:
        train = _rows(Path(str(config["data"]["train_manifest"])).expanduser(), "train")
        semantic = verify_semantics(config, train, output)
        runtime = dict(config)
        runtime["class_names"] = {str(k): value["class_name"] for k, value in semantic["mapping"].items()}
        qualification = _qualification(runtime, train, output)
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
    try:
        factories = runtime["factories"]
        result = _run_formal_matrix(runtime, train, output, _factory(str(factories["apart"])),
                                    _factory(str(factories["clip"])))
        baseline = config.get("f1_a0_reference")
        if not baseline:
            decision = "BLOCKED_BASELINE_REPRODUCTION"
            baseline_status = "BLOCKED_NO_REFERENCE_PREDICTIONS"
        else:
            decision = "PENDING_UTILITY_GATE"
            baseline_status = "LOCKED_REFERENCE_PROVIDED"
        report = {
            "status": decision, "dataset": "isic", "primary_readout": "A6",
            "stage_rows": result["stage_rows"], "class_rows": result["class_rows"],
            "baseline_reproduction": baseline_status, "test_access": 0,
            "reserved_access": 0, "new_training_epochs": 0, "optimizer_steps": 0,
            "resource": result["resource"],
        }
        _write(output / "REPORT.json", report)
        (output / "FINAL_REPORT_ZH.md").write_text(
            "# CT13-ISIC\n\n" + f"状态：`{decision}`。\n\n"
            f"阶段行 {result['stage_rows']}，逐类行 {result['class_rows']}；"
            "A6 为主方法，新增神经训练和 optimizer step 均为 0。\n\n"
            f"A0 历史 F1 复现：`{baseline_status}`。\n\n"
            "本次 val 在全部统计/文本/读出锁定后统一访问；test/reserved 访问为 0。\n",
            encoding="utf-8",
        )
        if decision != "PENDING_UTILITY_GATE":
            return 4
        _write(output / "COMPLETE.json", {"status": "COMPLETE", **report})
        return 0
    except (ImportError, KeyError, TypeError, ValueError, OSError) as exc:
        _write(output / "ENGINEERING_GATE.json", {"status": "FAILED_AFTER_GATE", "reason": str(exc),
                                                      "val_access": 0, "test_access": 0,
                                                      "elapsed_s": time.time() - start})
        _write(output / "BLOCKED.json", {"status": "BLOCKED_MATRIX", "errors": [str(exc)],
                                          "val_access": 0, "test_access": 0, "reserved_access": 0})
        return 5
