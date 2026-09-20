#!/usr/bin/env python3
"""Read-only P0 integrity audit of the frozen ISIC split; never trains or predicts."""
import argparse
import csv
import hashlib
import io
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def audit(data_root, split_root, output, workers=4):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dataset_audit.json").exists():
        raise FileExistsError("Refusing to overwrite a previous dataset audit")
    images = defaultdict(list)
    for path in data_root.rglob("*"):
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            images[path.stem].append(path)
    rows, counts, split_hashes, failures = [], {}, {}, []
    semantic_codes = defaultdict(set)
    for split in ("train", "val", "test"):
        path = split_root / f"{split}_skin1_0.01.csv"
        split_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        seen = set()
        counts[split] = Counter()
        for record in records:
            sid, label = record["image"], int(record["finding"])
            if sid in seen:
                failures.append({"reason": "duplicate_id_within_split", "split": split, "sample_id": sid})
            seen.add(sid)
            semantic_codes[label].add(record.get("finding_name", ""))
            candidates = images.get(sid, [])
            if len(candidates) != 1:
                failures.append({"reason": "missing_or_ambiguous_path", "sample_id": sid, "matches": len(candidates)})
            counts[split][label] += 1
            rows.append({"sample_id": sid, "original_label": label, "mapped_label": label,
                         "split": split, "relative_path": str(candidates[0].relative_to(data_root)) if len(candidates) == 1 else "",
                         "lesion_or_patient_id_if_available": "", "file_hash": ""})
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["sample_id"]].append(row)
    for sid, group in by_id.items():
        if len({r["split"] for r in group}) > 1:
            failures.append({"reason": "cross_split_id", "sample_id": sid})
        if len({r["original_label"] for r in group}) > 1:
            failures.append({"reason": "conflicting_label", "sample_id": sid})
    if set(counts["train"]) != set(range(8)):
        failures.append({"reason": "training_classes_not_0_to_7"})
    for split, n in (("val", 50), ("test", 100)):
        if counts[split] != Counter({i: n for i in range(8)}):
            failures.append({"reason": "unexpected_evaluation_counts", "split": split})
    if any(len(v) != 1 for v in semantic_codes.values()):
        failures.append({"reason": "inconsistent_finding_name_code"})

    def inspect(row):
        if not row["relative_path"]:
            return row, 0, None
        path = data_root / row["relative_path"]
        try:
            data = path.read_bytes()
            row["file_hash"] = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                image.convert("RGB").load()
            return row, len(data), None
        except Exception as exc:
            return row, 0, {"reason": "read_or_decode_error", "sample_id": row["sample_id"], "error": str(exc)}

    total_bytes, hashes, checked = 0, defaultdict(list), []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (row, size, error) in enumerate(executor.map(inspect, rows), 1):
            checked.append(row)
            total_bytes += size
            if error:
                failures.append(error)
            if row["file_hash"]:
                hashes[row["file_hash"]].append(row)
            if index % 2000 == 0:
                print(json.dumps({"decoded": index, "total": len(rows), "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
    duplicate_content_groups = [group for group in hashes.values() if len(group) > 1]
    for group in duplicate_content_groups:
        if len({r["split"] for r in group}) > 1:
            failures.append({"reason": "cross_split_identical_content", "samples": [{"sample_id": r["sample_id"], "split": r["split"]} for r in group]})
        if len({r["original_label"] for r in group}) > 1:
            failures.append({"reason": "conflicting_content_labels", "samples": [{"sample_id": r["sample_id"], "original_label": r["original_label"]} for r in group]})
    with (output / "image_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(checked)
    with (output / "class_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_label", "n_train", "n_val", "n_test"])
        for label in range(8):
            writer.writerow([label] + [counts[s][label] for s in ("train", "val", "test")])
    metadata_candidates = [str(p) for folder in (data_root, data_root.parent) for p in folder.iterdir()
                           if p.is_file() and p.suffix.lower() in {".csv", ".json", ".tsv"} and not p.name.startswith("._")]
    result = {
        "status": "BLOCKED_DATA_PROTOCOL" if failures else "IMAGE_LEVEL_AUDIT_PASS",
        "split_id": 1, "split_suffix": "0.01", "split_sha256": split_hashes,
        "counts": {s: dict(sorted(c.items())) for s, c in counts.items()},
        "samples_audited": len(rows), "decoded_and_read_bytes": total_bytes,
        "nmax_nmin_ratio": max(counts["train"].values()) / min(counts["train"].values()),
        "classes_above_embedding_limit_500": [k for k,v in counts["train"].items() if v > 500],
        "duplicate_content_groups": len(duplicate_content_groups),
        "failure_counts": dict(Counter(x["reason"] for x in failures)),
        "failures_private_file": "dataset_failures.json",
        "available_metadata_candidates": metadata_candidates,
        "group_id_coverage": 0, "group_isolation": "UNVERIFIED",
        "semantic_map": "SEMANTIC_MAP_UNVERIFIED",
        "numeric_finding_name_codes": {k: sorted(v) for k,v in semantic_codes.items()},
        "mapped_label_definition": "canonical numerical label; repeat-specific head mappings are separate",
        "development_exposure": "UNKNOWN", "model_test_predictions": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "dataset_failures.json").write_text(json.dumps(failures, indent=2))
    (output / "dataset_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    audit(args.data_root, args.split_root, args.output, args.workers)
