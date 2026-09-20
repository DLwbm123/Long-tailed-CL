import json
from pathlib import Path

from route_a.run_ct13_real import execute


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
