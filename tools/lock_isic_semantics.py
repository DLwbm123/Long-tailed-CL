"""Bind the local training labels to the official ISIC 2019 training diagnoses.

Read the official CSV from stdin; retain evidence only for IDs in the locked
train manifest. No validation or held-out manifest is opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys


NAMES = {
    "MEL": "melanoma",
    "NV": "melanocytic nevus",
    "BCC": "basal cell carcinoma",
    "AK": "actinic keratosis",
    "BKL": "benign keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular lesion",
    "SCC": "squamous cell carcinoma",
}
SOURCE = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv"


def lock(train_path: Path, source: bytes) -> dict:
    with train_path.open(newline="", encoding="utf-8") as stream:
        train = list(csv.DictReader(stream))
    ids = {row["sample_id"]: int(row["original_label"]) for row in train}
    if len(ids) != len(train) or set(ids.values()) != set(range(8)):
        raise ValueError("BLOCKED_TRAIN_LABELS")
    seen = set()
    labels: dict[int, set[str]] = {n: set() for n in range(8)}
    counts: dict[int, int] = {n: 0 for n in range(8)}
    reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig")))
    if not set(NAMES).issubset(reader.fieldnames or ()) or "image" not in (reader.fieldnames or ()):
        raise ValueError("BLOCKED_OFFICIAL_COLUMNS")
    for row in reader:
        image = row["image"]
        if image not in ids:
            continue
        if image in seen:
            raise ValueError("BLOCKED_OFFICIAL_DUPLICATE")
        positive = [code for code in NAMES if row[code] == "1.0" or row[code] == "1"]
        if len(positive) != 1:
            raise ValueError("BLOCKED_OFFICIAL_ONE_HOT")
        label = ids[image]
        labels[label].add(positive[0])
        counts[label] += 1
        seen.add(image)
    if seen != set(ids) or any(len(codes) != 1 for codes in labels.values()):
        raise ValueError("BLOCKED_SEMANTIC_MAP_COVERAGE_OR_CONFLICT")
    mapping = {str(label): {"diagnosis_code": next(iter(codes)),
                            "class_name": NAMES[next(iter(codes))]}
               for label, codes in labels.items()}
    if len({v["diagnosis_code"] for v in mapping.values()}) != 8:
        raise ValueError("BLOCKED_SEMANTIC_MAP_CONFLICT")
    return {"mapping": mapping, "source_url": SOURCE,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "train_manifest_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "matched_train_rows": len(seen), "counts": counts,
            "name_source": "https://challenge.isic-archive.com/landing/2019/"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = lock(args.train, sys.stdin.buffer.read())
    os.umask(0o077)
    tmp = args.out.with_suffix(args.out.suffix + ".partial")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.out)
    print(json.dumps({"status": "VERIFIED_TRAIN_ONLY", "matched_train_rows": result["matched_train_rows"],
                      "mapping": result["mapping"]}, sort_keys=True))


if __name__ == "__main__":
    main()
