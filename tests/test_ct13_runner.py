import json
from pathlib import Path

import numpy as np
from PIL import Image

from route_a.run_ct13_real import _extract_rows, execute


def test_real_launcher_records_missing_locked_assets_without_val_access(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "train.csv").write_text("sample_id,mapped_label,split,relative_path\na,0,train,a.jpg\n")
    (root / "val.csv").write_text("sample_id,mapped_label,split,relative_path\nb,0,val,b.jpg\n")
    cfg = {
        "route": "A_RASP", "dataset": "isic", "tasks": [1, 2, 3, 4],
        "seeds": [1993, 1994, 1995], "lambda": 0.001, "holdout_access": "zero",
        "data": {"image_root": str(root), "train_manifest": str(root / "train.csv"),
                 "val_manifest": str(root / "val.csv")},
        "f1_parent_state": {str(s): str(root / f"{s}.pt") for s in (1993, 1994, 1995)},
        "clip_lock": {"model_name": "ViT-B-16", "pretrained": "laion400m_e32",
                      "weights_path": str(root / "missing.pt"), "preprocess_path": str(root / "prep"),
                      "tokenizer_path": str(root / "tok"), "weights_sha256": "0" * 64,
                      "preprocess_sha256": "1" * 64, "tokenizer_sha256": "2" * 64},
        "output_dir": str(tmp_path / "out"),
    }
    assert execute(cfg, Path(cfg["output_dir"])) == 2
    gate = json.loads((tmp_path / "out" / "ENGINEERING_GATE.json").read_text())
    blocked = json.loads((tmp_path / "out" / "BLOCKED.json").read_text())
    assert gate["status"] == "BLOCKED" and gate["val_access"] == 0
    assert blocked["test_access"] == blocked["reserved_access"] == 0


def test_real_extractor_uses_separate_preprocess_and_explicit_apart_pair(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_root / "a.jpg")
    rows = [{"sample_id": "a", "mapped_label": "0", "split": "train", "relative_path": "a.jpg"}]
    apart_seen, clip_seen = [], []

    def apart_pre(image):
        apart_seen.append(image.getpixel((0, 0)))
        return np.ones(2)

    def clip_pre(image):
        clip_seen.append(image.getpixel((0, 0)))
        return np.full(2, 2.0)

    apart = {
        "preprocess": apart_pre,
        "forward": lambda batch: {"pre_logits": np.ones((len(batch), 768)),
                                   "pre_logits_few": np.ones((len(batch), 768))},
    }
    clip = {"preprocess": clip_pre,
            "forward": lambda batch: {"pre_logits": np.ones((len(batch), 512))}}
    a, u, h = _extract_rows(rows, image_root, apart, clip)
    assert len(apart_seen) == len(clip_seen) == 1
    assert a.shape == (1, 1536) and u.shape == (1, 512) and h.shape == (1, 2048)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0)
