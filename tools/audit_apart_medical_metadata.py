#!/usr/bin/env python3
"""Audit exact ISIC IDs against available metadata, retaining group IDs privately."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit_metadata(split_root, metadata, output):
    selected = {}
    for split in ("train", "val", "test"):
        with (split_root / f"{split}_skin1_0.01.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                selected[row["image"]] = {"split": split, "label": int(row["finding"])}
    matched, duplicates, conflicts = {}, [], []
    with metadata.open(newline="") as handle:
        reader = csv.DictReader(handle)
        count = 0
        for row in reader:
            count += 1
            sid = row["isic_id"]
            if sid not in selected:
                continue
            value = {k: row.get(k, "").strip() for k in ("patient_id", "lesion_id", "diagnosis")}
            if sid in matched:
                duplicates.append(sid)
                if matched[sid] != value:
                    conflicts.append(sid)
            matched[sid] = value
    coverage, overlap, details, pairs = {}, {}, {}, {}
    for field in ("lesion_id", "patient_id"):
        groups = defaultdict(list)
        for sid, record in matched.items():
            if record[field].lower() not in {"", "nan", "none", "null", "unknown"}:
                groups[record[field]].append({"sample_id": sid, "split": selected[sid]["split"]})
        details[field] = {key: rows for key, rows in groups.items() if len({r["split"] for r in rows}) > 1}
        overlap[field] = len(details[field])
        coverage[field] = {"known_images": sum(len(v) for v in groups.values()), "total_images": len(selected)}
        pairs[field] = {
            f"{a}__{b}": sum({a,b}.issubset({r["split"] for r in rows}) for rows in groups.values())
            for a,b in (("train", "val"), ("train", "test"), ("val", "test"))
        }
    blocked = any(overlap.values()) or bool(conflicts)
    result = {
        "status": "BLOCKED_DATA_PROTOCOL" if blocked else "NO_OBSERVED_GROUP_OVERLAP",
        "source_path": str(metadata), "source_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "metadata_rows_scanned": count, "matched_images": len(matched), "selected_images": len(selected),
        "duplicate_metadata_ids": len(duplicates), "conflicting_metadata_ids": len(conflicts),
        "coverage": coverage, "cross_split_overlapping_groups": overlap, "pairwise_overlapping_groups": pairs,
        "group_isolation_fully_verified": False,
        "semantic_labels_by_finding": {label: dict(Counter(v["diagnosis"] for sid,v in matched.items() if selected[sid]["label"] == label)) for label in range(8)},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "group_metadata_audit.json").write_text(json.dumps(result, indent=2))
    (output / "group_overlap_details_private.json").write_text(json.dumps(details, indent=2))
    (output / "matched_group_metadata_private.json").write_text(json.dumps(matched, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "semantic_labels_by_finding"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_metadata(args.split_root, args.metadata, args.output)
