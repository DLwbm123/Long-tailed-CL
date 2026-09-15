"""Small synthetic integrity checks; these never use the medical test images."""
import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_apart_medical_transfer import audit
from audit_apart_medical_metadata import audit_metadata


class AuditTests(unittest.TestCase):
    def test_content_overlap_is_not_hidden_by_distinct_ids(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            images, splits = root / "images", root / "splits"
            images.mkdir()
            splits.mkdir()
            sid = 0
            for split, n in (("train", 1), ("val", 50), ("test", 100)):
                with (splits / f"{split}_skin1_0.01.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["image", "finding_name", "finding"])
                    for label in range(8):
                        for _ in range(n):
                            name = f"sample{sid}"
                            writer.writerow([name, str(label), label])
                            Image.new("RGB", (2, 2), (sid % 256, sid // 256, 123)).save(images / f"{name}.png")
                            sid += 1
            clean = audit(images, splits, root / "clean", workers=1)
            self.assertEqual(clean["status"], "IMAGE_LEVEL_AUDIT_PASS")
            self.assertEqual(clean["samples_audited"], 1208)
            (images / "sample408.png").write_bytes((images / "sample1.png").read_bytes())
            leaking = audit(images, splits, root / "leaking", workers=1)
            self.assertEqual(leaking["failure_counts"], {"cross_split_identical_content": 1, "conflicting_content_labels": 1})
            self.assertEqual(leaking["status"], "BLOCKED_DATA_PROTOCOL")

    def test_missing_group_ids_are_unknown_and_known_overlap_blocks(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for i, split in enumerate(("train", "val", "test")):
                (root / f"{split}_skin1_0.01.csv").write_text(f"image,finding\nsample{i},0\n")
            meta = root / "metadata.csv"
            meta.write_text("isic_id,lesion_id,patient_id,diagnosis\nsample0,group1,,\nsample1,group1,,\nsample2,,,\n")
            result = audit_metadata(root, meta, root / "out")
            self.assertEqual(result["cross_split_overlapping_groups"], {"lesion_id": 1, "patient_id": 0})
            self.assertEqual(result["coverage"]["patient_id"]["known_images"], 0)
            self.assertEqual(result["status"], "BLOCKED_DATA_PROTOCOL")


if __name__ == "__main__":
    unittest.main()
