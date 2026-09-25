"""Frozen NB2-VLM-R1 matrix. Configuration is supplied through NB2_CONFIG."""
from __future__ import annotations

import csv
import fcntl
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ct13_runtime.factories import build_apart, build_clip
from route_a.run_ct13_real import (
    _extract_rows, _load_bank, _load_readouts, _save_bank, _save_readouts,
    _stage_metrics, _write_csv,
)
from route_a.semantic_prior import component_reliability, semantic_scale, text_prompts, unit_columns
from route_a.spectral_prior_ridge import a0_to_a8, readout_scores, ridge
from shared.dual_moment_bank import DualMomentBank

METHODS = ("P", "A0", "A1", "A2", "A3", "A3s", "A4", "A5", "A6", "A7", "A8")
SIZES = {"ISIC": (2, 2, 2, 2), "HK": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3)}


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("BLOCKED_DUPLICATE_JSON_KEY:" + key)
        out[key] = value
    return out


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _manifest(path: Path, split: str, root: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["split"] != split for row in rows):
        raise ValueError("BLOCKED_MANIFEST_SPLIT")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("BLOCKED_MANIFEST_IDS")
    base = root.resolve()
    for row in rows:
        rel = Path(row["relative_path"])
        if rel.is_absolute() or ".." in rel.parts or not (base / rel).resolve().is_relative_to(base):
            raise ValueError("BLOCKED_MANIFEST_PATH")
        if not row.get("identity_component"):
            raise ValueError("BLOCKED_COMPONENT_ID")
        if not (base / rel).is_file():
            raise ValueError("BLOCKED_IMAGE_MISSING")
    return sorted(rows, key=lambda row: row["sample_id"])


def _label(row: dict[str, str]) -> int:
    value = int(row["original_label"])
    if "mapped_label" in row and int(row["mapped_label"]) != value:
        raise ValueError("BLOCKED_LABEL_MAPPING")
    return value


def _names(dataset: str, cfg: dict[str, Any], train: list[dict[str, str]]) -> dict[int, str]:
    if dataset == "ISIC":
        evidence = _read(Path(cfg["isic_semantics"]))
        mapping = {int(k): v["class_name"] for k, v in evidence["mapping"].items()}
        if evidence["matched_train_rows"] != len(train) or evidence["train_manifest_sha256"] != _sha(Path(cfg["datasets"][dataset]["train_manifest"])):
            raise ValueError("BLOCKED_ISIC_SEMANTICS_COUNT")
    else:
        with Path(cfg["hk_namespace"]).open(newline="", encoding="utf-8") as stream:
            mapping = {int(row["official_name_sorted_id"]): row["official_class_name"]
                       for row in csv.DictReader(stream)}
        for row in train:
            if mapping[_label(row)] != row["official_class_name"]:
                raise ValueError("BLOCKED_HK_SEMANTICS_ROW")
    if set(mapping) != set(map(_label, train)) or any(not name for name in mapping.values()):
        raise ValueError("BLOCKED_SEMANTICS_COVERAGE")
    return mapping


def _task_order(lock: dict[str, Any], dataset: str, seed: int) -> list[int]:
    info = lock[dataset]
    order = list(map(int, info["orders"][str(seed)]))
    if tuple(info["task_sizes"]) != SIZES[dataset] or len(order) != sum(SIZES[dataset]) or len(set(order)) != len(order):
        raise ValueError("BLOCKED_TASK_LOCK")
    return order


def _tail(lock: dict[str, Any], cfg: dict[str, Any], dataset: str) -> set[int]:
    # Read the historical frequency groups, then check them against locked fit counts.
    counts = {int(k): int(v) for k, v in lock[dataset]["train_counts"].items()}
    n = 2 if dataset == "ISIC" else 8
    key = "tail_rank2" if dataset == "ISIC" else "tail_rank8"
    ids = list(map(int, _read(Path(cfg["datasets"][dataset]["frequency_lock"]))["frequency_groups"][key]))
    if len(ids) != n or set(ids) != set(sorted(counts, key=lambda c: (counts[c], c))[:n]):
        raise ValueError("BLOCKED_FREQUENCY_GROUP_LOCK")
    return set(ids)


def _parent_lock(cfg: dict[str, Any], task: dict[str, Any], dataset: str, seed: int) -> tuple[Path, dict[str, Any]]:
    root = Path(cfg["asset_root"])
    parent = root / "parents_task1" / f"{dataset}_{seed}_U_t01.pt"
    meta = _read(Path(str(parent) + ".json"))
    weight = Path(cfg["augreg_weight"])
    lock = {
        "dataset": dataset, "seed": seed, "parent_sha256": meta["sha256"],
        "order": _task_order(task, dataset, seed),
        "manifest_sha256": task[dataset]["manifest_sha256"],
        "source_root": cfg["ct1_source_root"],
        "original_weight_path": cfg["original_augreg_path"],
        "weight_path": str(weight),
    }
    return parent, lock


def _clip_lock(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["asset_root"]) / "weights"
    files = _read(root / "CLIP_FILES_LOCK.json")["files"]
    return {
        "weights_path": str(root / "open_clip_model.safetensors"),
        "weights_sha256": files["open_clip_model.safetensors"]["sha256"],
        "preprocess_path": str(root / "open_clip_config.json"),
        "preprocess_sha256": files["open_clip_config.json"]["sha256"],
        "tokenizer_path": str(root / "tokenizer.json"),
        "tokenizer_sha256": files["tokenizer.json"]["sha256"],
    }


def _text(clip: dict[str, Any], dataset: str, names: list[str]) -> np.ndarray:
    domain = "isic" if dataset == "ISIC" else "hyperkvasir"
    prompts = text_prompts(domain, names, 0) + text_prompts(domain, names, 1)
    raw = clip["encode_text"](clip["tokenizer"](prompts)).numpy().astype(np.float64)
    if raw.shape != (2 * len(names), 512):
        raise ValueError("BLOCKED_TEXT_DIM")
    return unit_columns(((raw[:len(names)] + raw[len(names):]) / 2).T)


def _p_empty() -> dict[str, Any]:
    return {"S": np.zeros((768, 768), np.float64), "means": [], "ids": []}


def _p_add(bank: dict[str, Any], cid: int, x: np.ndarray) -> None:
    if cid in bank["ids"] or x.ndim != 2 or x.shape[1] != 768 or not np.isfinite(x).all():
        raise ValueError("BLOCKED_P_BANK")
    bank["S"] += x.T @ x / len(x)
    bank["means"].append(x.mean(axis=0))
    bank["ids"].append(cid)


def _p_save(bank: dict[str, Any], path: Path) -> None:
    np.savez_compressed(path, S=bank["S"], M=np.column_stack(bank["means"]),
                        ids=np.asarray(bank["ids"], dtype=np.int64))


def _p_load(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {"S": z["S"].copy(), "means": [z["M"][:, j].copy() for j in range(z["M"].shape[1])],
                "ids": z["ids"].astype(int).tolist()}


def _append_access(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _seal(stage: Path, bank: DualMomentBank, pbank: dict[str, Any],
          readouts: dict[str, Any], p_weight: np.ndarray, text: np.ndarray,
          a_scale: float, gamma: np.ndarray, ncomp: np.ndarray,
          lock: dict[str, Any]) -> None:
    if stage.exists():
        raise FileExistsError("BLOCKED_STAGE_ALREADY_SEALED")
    temp = stage.with_name(stage.name + f".part.{os.getpid()}")
    temp.mkdir(parents=True, exist_ok=False)
    try:
        _save_bank(bank, temp / "BANK.npz")
        _p_save(pbank, temp / "P_BANK.npz")
        _save_readouts(readouts, temp / "READOUTS.npz", text_u=text, a_scale=a_scale,
                       gamma=gamma, ncomp=ncomp)
        np.save(temp / "P_W.npy", p_weight)
        files = ("BANK.npz", "P_BANK.npz", "READOUTS.npz", "READOUTS.json", "P_W.npy")
        lock["file_sha256"] = {name: _sha(temp / name) for name in files}
        _write(temp / "STATE_W_LOCK.json", lock)
        temp.rename(stage)
    except BaseException:
        # Preserve a failed transaction for diagnosis; a later run never treats it as committed.
        raise


def _resume_stage(stage: Path, source_hash: str, protocol_hash: str,
                  order: list[int]) -> tuple[DualMomentBank, dict[str, Any]]:
    lock = _read(stage / "STATE_W_LOCK.json")
    if (lock["source_sha256"] != source_hash or lock["protocol_sha256"] != protocol_hash
            or lock["class_order"] != order):
        raise ValueError("BLOCKED_STAGE_LOCK")
    if any(_sha(stage / name) != digest for name, digest in lock["file_sha256"].items()):
        raise ValueError("BLOCKED_STAGE_DIGEST")
    bank, pbank = _load_bank(stage / "BANK.npz"), _p_load(stage / "P_BANK.npz")
    if bank.class_order != order or pbank["ids"] != order:
        raise ValueError("BLOCKED_STAGE_ORDER")
    _load_readouts(stage / "READOUTS.npz")
    return bank, pbank


def _guard() -> set[str]:
    allowed: set[str] = set()
    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        if path.name in ("test.csv", "reserved.csv", "test.npz", "reserved.npz"):
            raise ValueError("BLOCKED_TEST_RESERVED_ACCESS")
        if path.suffix.lower() in (".jpg", ".jpeg", ".png") and str(path.resolve()) not in allowed:
            raise ValueError("BLOCKED_OLD_FUTURE_OR_TEST_IMAGE")
    sys.addaudithook(audit)
    return allowed


def _allow(allowed: set[str], root: Path, rows: list[dict[str, str]]) -> None:
    allowed.clear()
    allowed.update(str((root / row["relative_path"]).resolve()) for row in rows)


def _heartbeat(out: Path, phase: str, dataset: str = "", seed: int = 0, task: int = 0) -> None:
    _write(out / "HEARTBEAT.json", {"time": time.time(), "phase": phase, "dataset": dataset,
                                    "seed": seed, "task": task, "pid": os.getpid()})


def _gate(cfg: dict[str, Any], out: Path, task: dict[str, Any],
          train: dict[str, list[dict[str, str]]], allowed: set[str]) -> None:
    clip = build_clip(lock=_clip_lock(cfg))
    receipts = []
    for dataset in SIZES:
        image_root = Path(cfg["datasets"][dataset]["image_root"])
        for seed in (1993, 1994, 1995):
            order = _task_order(task, dataset, seed)
            rows = [row for cid in order[:2]
                    for row in [r for r in train[dataset] if _label(r) == cid][:2]]
            if len(rows) != 4:
                raise ValueError("BLOCKED_GATE_TASK1_SUPPORT")
            parent, lock = _parent_lock(cfg, task, dataset, seed)
            apart = build_apart(checkpoint=parent, lock=lock)
            _allow(allowed, image_root, rows)
            a, u, h, p = _extract_rows(rows, image_root, apart, clip, batch_size=48, with_p=True)
            if apart["state_hash"]() != apart["restore"]["network_sha256"]:
                raise ValueError("BLOCKED_GATE_BUFFER_DRIFT")
            receipts.append({"dataset": dataset, "seed": seed, "parent_sha256": lock["parent_sha256"],
                             "rows": len(rows), "shape": [list(x.shape) for x in (a, u, h, p)],
                             "network_sha256": apart["restore"]["network_sha256"]})
            _heartbeat(out, "gate", dataset, seed, 1)
            del apart
            gc.collect()
            torch.cuda.empty_cache()
    if clip["state_hash"]() != clip["initial_state_hash"]:
        raise ValueError("BLOCKED_CLIP_STATE_DRIFT")
    _write(out / "ENGINEERING_REPORT.json", {"status": "PASS", "receipts": receipts,
                                             "source_sha256": _sha(out / "SOURCE_LOCK.json"),
                                             "protocol_sha256": _sha(out / "PROTOCOL_LOCK.json"),
                                             "val_access": 0, "test_access": 0, "reserved_access": 0,
                                             "new_training_epochs": 0, "optimizer_steps": 0})


def _fit(cfg: dict[str, Any], out: Path, task: dict[str, Any],
         train: dict[str, list[dict[str, str]]], names: dict[str, dict[int, str]],
         allowed: set[str], source_hash: str, protocol_hash: str) -> None:
    clip = build_clip(lock=_clip_lock(cfg))
    journal = out / "ACCESS_LEDGER.jsonl"
    sealed_count = 0
    for dataset in SIZES:
        image_root = Path(cfg["datasets"][dataset]["image_root"])
        by_class = {cid: [row for row in train[dataset] if _label(row) == cid] for cid in names[dataset]}
        for seed in (1993, 1994, 1995):
            order = _task_order(task, dataset, seed)
            parent, lock = _parent_lock(cfg, task, dataset, seed)
            apart = build_apart(checkpoint=parent, lock=lock)
            bank, pbank = DualMomentBank(), _p_empty()
            position = 0
            for stage_index, size in enumerate(SIZES[dataset], 1):
                current = order[position:position + size]
                position += size
                seen = order[:position]
                stage = out / "stages" / dataset / str(seed) / f"task_{stage_index:02d}"
                if stage.exists():
                    bank, pbank = _resume_stage(stage, source_hash, protocol_hash, seen)
                    sealed_count += 1
                    _heartbeat(out, "fit_resumed", dataset, seed, stage_index)
                    continue
                for cid in current:
                    rows = by_class[cid]
                    if not rows:
                        raise ValueError("BLOCKED_EMPTY_TRAIN_CLASS")
                    _allow(allowed, image_root, rows)
                    def batch_done(n: int, *, cid: int = cid) -> None:
                        _append_access(journal, {"phase": "fit", "dataset": dataset, "seed": seed,
                                                 "task": stage_index, "class_id": cid,
                                                 "image_reads": n, "apart_forwards": 1,
                                                 "clip_forwards": 1, "time": time.time()})
                    _, _, h, p = _extract_rows(rows, image_root, apart, clip, batch_size=48,
                                               with_p=True, on_batch=batch_done)
                    bank.add_class(cid, h, task=stage_index,
                                   component_ids=[row["identity_component"] for row in rows])
                    _p_add(pbank, cid, p)
                    del h, p
                S, M, ids = bank.class_balanced()
                if ids.tolist() != seen or pbank["ids"] != seen:
                    raise ValueError("BLOCKED_FUTURE_OR_OLD_CLASS")
                ncomp = np.asarray([bank.classes[c].component_count for c in seen], np.float64)
                text = _text(clip, dataset, [names[dataset][c] for c in seen])
                Gu, Ru = bank.u_stats()
                scale = semantic_scale(text, Gu, Ru)
                comp_stats, comp_counts = bank.component_reliability_inputs()
                _, gamma = component_reliability(text, comp_stats, comp_counts,
                                                  class_ids=seen, a_scale=scale)
                V = np.zeros_like(M)
                V[1536:] = np.sqrt(2) * scale * (text - text.mean(axis=1, keepdims=True))
                readouts = a0_to_a8(S, M, text, V=V, gamma=gamma, a_scale=scale, ncomp=ncomp)
                k = len(seen)
                p_weight = ridge(pbank["S"] / k, np.column_stack(pbank["means"]) / k)
                if apart["state_hash"]() != apart["restore"]["network_sha256"]:
                    raise ValueError("BLOCKED_NEURAL_STATE_DRIFT")
                stage_lock = {"dataset": dataset, "seed": seed, "task": stage_index,
                              "class_order": list(seen), "current_classes": list(current),
                              "parent_sha256": lock["parent_sha256"],
                              "network_sha256": apart["restore"]["network_sha256"],
                              "source_sha256": source_hash, "protocol_sha256": protocol_hash,
                              "n_components": ncomp.tolist(), "gamma": gamma.tolist(),
                              "semantic_scale": scale, "new_training_epochs": 0,
                              "optimizer_steps": 0, "test_access": 0, "reserved_access": 0,
                              "rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
                _seal(stage, bank, pbank, readouts, p_weight, text, scale, gamma, ncomp, stage_lock)
                _resume_stage(stage, source_hash, protocol_hash, seen)
                sealed_count += 1
                _heartbeat(out, "fit_sealed", dataset, seed, stage_index)
                print("SEALED", dataset, seed, stage_index, flush=True)
            del apart, bank, pbank
            gc.collect()
            torch.cuda.empty_cache()
    if sealed_count != 45:
        raise ValueError("BLOCKED_STAGE_COVERAGE")
    if clip["state_hash"]() != clip["initial_state_hash"]:
        raise ValueError("BLOCKED_CLIP_STATE_DRIFT")
    _write(out / "FIT_LOCK.json", {"status": "ALL_STAGES_SEALED", "stage_count": sealed_count,
                                   "val_access": 0, "test_access": 0, "reserved_access": 0})


def _evaluate(cfg: dict[str, Any], out: Path, task: dict[str, Any],
              allowed: set[str], source_hash: str, protocol_hash: str) -> None:
    if _read(out / "FIT_LOCK.json")["status"] != "ALL_STAGES_SEALED":
        raise ValueError("BLOCKED_PREVAL_FIT")
    clip = build_clip(lock=_clip_lock(cfg))
    stages, classes, index = [], [], []
    journal = out / "ACCESS_LEDGER.jsonl"
    for dataset in SIZES:
        image_root = Path(cfg["datasets"][dataset]["image_root"])
        val = _manifest(Path(cfg["datasets"][dataset]["val_manifest"]), "val", image_root)
        labels = np.asarray([_label(row) for row in val], np.int64)
        components = np.asarray([row["identity_component"] for row in val])
        predictions = []
        for seed in (1993, 1994, 1995):
            order = _task_order(task, dataset, seed)
            parent, lock = _parent_lock(cfg, task, dataset, seed)
            apart = build_apart(checkpoint=parent, lock=lock)
            _allow(allowed, image_root, val)
            def batch_done(n: int) -> None:
                _append_access(journal, {"phase": "val", "dataset": dataset, "seed": seed,
                                         "image_reads": n, "apart_forwards": 1,
                                         "clip_forwards": 1, "time": time.time()})
            _, _, h_val, p_val = _extract_rows(val, image_root, apart, clip, batch_size=48,
                                               with_p=True, on_batch=batch_done)
            position = 0
            for stage_index, size in enumerate(SIZES[dataset], 1):
                position += size
                seen = order[:position]
                current = seen[-size:]
                stage = out / "stages" / dataset / str(seed) / f"task_{stage_index:02d}"
                _resume_stage(stage, source_hash, protocol_hash, seen)
                readouts = _load_readouts(stage / "READOUTS.npz")
                p_weight = np.load(stage / "P_W.npy", allow_pickle=False)
                stage_lock = _read(stage / "STATE_W_LOCK.json")
                for method in METHODS:
                    scores = p_val @ p_weight if method == "P" else readout_scores(readouts, method, h_val)
                    metric, class_rows, pred = _stage_metrics(scores, labels, seen, current,
                                                              _tail(task, cfg, dataset))
                    valid = np.isin(labels, seen)
                    old = np.isin(labels, seen[:-size])
                    new = np.isin(labels, current)
                    metric["old_to_current"] = int(np.sum(valid & old & np.isin(pred, current)))
                    metric["current_to_old"] = int(np.sum(valid & new & np.isin(pred, seen[:-size])))
                    stages.append({"dataset": dataset, "seed": seed, "task": stage_index,
                                   "method": method, "seen_classes": len(seen), **metric})
                    for row in class_rows:
                        row.update({"dataset": dataset, "seed": seed, "task": stage_index,
                                    "method": method, "n_components_fit":
                                        stage_lock["n_components"][seen.index(row["class_id"])]})
                    classes.extend(class_rows)
                    predictions.append(pred.astype(np.int16))
                    index.append({"dataset": dataset, "seed": seed, "task": stage_index, "method": method})
                _heartbeat(out, "val", dataset, seed, stage_index)
            if apart["state_hash"]() != apart["restore"]["network_sha256"]:
                raise ValueError("BLOCKED_NEURAL_STATE_DRIFT")
            del apart, h_val, p_val
            gc.collect()
            torch.cuda.empty_cache()
        private = out / "private"
        private.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(private / f"{dataset}_VAL_PREDICTIONS.npz",
                            labels=labels, components=components, predictions=np.stack(predictions),
                            index=np.asarray([json.dumps(item) for item in index if item["dataset"] == dataset]))
    if clip["state_hash"]() != clip["initial_state_hash"]:
        raise ValueError("BLOCKED_CLIP_STATE_DRIFT")
    keys = [(r["dataset"], r["seed"], r["task"], r["method"]) for r in stages]
    class_keys = [(*keys_for_class(r), r["class_id"]) for r in classes]
    if (len(stages) != 495 or len(classes) != 5049 or len(set(keys)) != 495
            or len(set(class_keys)) != 5049):
        raise ValueError("BLOCKED_MATRIX_COVERAGE")
    _write_csv(out / "stage_metrics.csv", stages)
    _write_csv(out / "class_metrics.csv", classes)
    _write(out / "PREDICTIONS_LOCK.json", {"status": "PASS", "stage_rows": 495,
                                          "class_rows": 5049, "test_access": 0,
                                          "reserved_access": 0})


def keys_for_class(row: dict[str, Any]) -> tuple[Any, ...]:
    return row["dataset"], row["seed"], row["task"], row["method"]


def _access_summary(out: Path) -> dict[str, Any]:
    phases = {"fit": {"image_reads": 0, "apart_forwards": 0, "clip_forwards": 0},
              "val": {"image_reads": 0, "apart_forwards": 0, "clip_forwards": 0}}
    with (out / "ACCESS_LEDGER.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            for key in phases[event["phase"]]:
                phases[event["phase"]][key] += int(event[key])
    result = {"phases": phases, "test_image_reads": 0, "test_model_forwards": 0,
              "reserved_image_reads": 0, "reserved_model_forwards": 0,
              "new_training_epochs": 0, "optimizer_steps": 0}
    _write(out / "ACCESS_LEDGER.json", result)
    return result


def main() -> int:
    started = time.monotonic()
    os.umask(0o077)
    cfg = _read(Path(os.environ["NB2_CONFIG"]))
    out = Path(cfg["run_root"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".process.lock").open("w") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        task = _read(Path(cfg["task_lock"]))
        migration = _read(Path(cfg["image_verify_summary"]))
        if any(migration[d]["files"] != (19013 if d == "ISIC" else 8519)
               or not migration[d]["sha256_verified"] for d in SIZES):
            raise ValueError("BLOCKED_MIGRATION_IMAGES")
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)
        allowed = _guard()
        train = {}
        names = {}
        for dataset in SIZES:
            spec = cfg["datasets"][dataset]
            root = Path(spec["image_root"])
            train[dataset] = _manifest(Path(spec["train_manifest"]), "train", root)
            if _sha(Path(spec["train_manifest"])) != task[dataset]["manifest_sha256"]["train"]:
                raise ValueError("BLOCKED_TRAIN_MANIFEST_SHA256")
            names[dataset] = _names(dataset, cfg, train[dataset])
            if {c: sum(_label(row) == c for row in train[dataset]) for c in names[dataset]} != {
                int(k): v for k, v in task[dataset]["train_counts"].items()
            }:
                raise ValueError("BLOCKED_TRAIN_COUNTS")
        protocol = {"name": "NB2-VLM-R1", "tasks": {k: list(v) for k, v in SIZES.items()},
                    "seeds": [1993, 1994, 1995], "methods": list(METHODS), "primary": "A6",
                    "lambda": .001, "batch_size": 48, "new_training_epochs": 0,
                    "optimizer_steps": 0, "test_access": 0, "reserved_access": 0}
        source = {"task_lock_sha256": _sha(Path(cfg["task_lock"])),
                  "augreg_sha256": _sha(Path(cfg["augreg_weight"])),
                  "clip_files": _read(Path(cfg["asset_root"]) / "weights/CLIP_FILES_LOCK.json"),
                  "image_verify_summary_sha256": _sha(Path(cfg["image_verify_summary"])),
                  "parent_sha256": {f"{ds}_{seed}": _parent_lock(cfg, task, ds, seed)[1]["parent_sha256"]
                                    for ds in SIZES for seed in (1993, 1994, 1995)},
                  "code_sha256": {name: _sha(Path(__file__).resolve().parents[1] / name)
                                  for name in ("tools/run_nb2_vlm_r1.py", "tools/finalize_nb2_vlm_r1.py",
                                               "ct13_runtime/factories.py",
                                               "route_a/run_ct13_real.py", "route_a/spectral_prior_ridge.py",
                                               "route_a/semantic_prior.py", "shared/dual_moment_bank.py",
                                               "shared/frozen_dual_features.py",
                                               "third_party/APART/utils/inc_net.py",
                                               "third_party/APART/utils/medical_v2.py",
                                               "third_party/APART/backbone/vision_transformer_adapter_pool_a.py")},
                  "frequency_lock_sha256": {ds: _sha(Path(cfg["datasets"][ds]["frequency_lock"]))
                                            for ds in SIZES}}
        if source["augreg_sha256"] != "c401d219603ac3e20b6373c7b198c78d3a733f80b755d148bda3bc320ae69800":
            raise ValueError("BLOCKED_AUGREG_SHA256")
        _write(out / "PROTOCOL_LOCK.json", protocol)
        _write(out / "SOURCE_LOCK.json", source)
        source_hash, protocol_hash = _sha(out / "SOURCE_LOCK.json"), _sha(out / "PROTOCOL_LOCK.json")
        try:
            phase = cfg["phase"]
            if phase == "gate":
                _gate(cfg, out, task, train, allowed)
            elif phase == "formal":
                gate = _read(out / "ENGINEERING_REPORT.json")
                if (gate["status"] != "PASS" or gate.get("source_sha256") != source_hash
                        or gate.get("protocol_sha256") != protocol_hash):
                    raise ValueError("BLOCKED_ENGINEERING_GATE")
                _fit(cfg, out, task, train, names, allowed, source_hash, protocol_hash)
                _evaluate(cfg, out, task, allowed, source_hash, protocol_hash)
                access = _access_summary(out)
                _write(out / "RESOURCE_LEDGER.json", {
                    "elapsed_seconds": time.monotonic() - started,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "single_gpu_scientific_worker": True, "loader_workers": 0,
                    "new_training_epochs": 0, "optimizer_steps": 0,
                    "fit_image_reads": access["phases"]["fit"]["image_reads"],
                    "val_image_reads": access["phases"]["val"]["image_reads"]})
                from tools.finalize_nb2_vlm_r1 import finalize
                finalize(out)
                _write(out / "MATRIX_COMPLETE.json", {"status": "COMPLETE",
                                                       "stage_rows": 495, "class_rows": 5049,
                                                       "new_training_epochs": 0,
                                                       "optimizer_steps": 0})
            else:
                raise ValueError("BLOCKED_PHASE")
        except BaseException as exc:
            _write(out / "BLOCKED.json", {"status": "BLOCKED", "phase": cfg.get("phase"),
                                          "reason": repr(exc), "time": time.time()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
