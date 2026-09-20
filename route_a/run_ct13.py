"""CT13-RASP protocol validator (no training or test access)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from shared.frozen_dual_features import validate_encoder_lock
except ModuleNotFoundError:  # direct ``python route_a/run_ct13.py`` from repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from shared.frozen_dual_features import validate_encoder_lock


REQUIRED = ("route", "dataset", "tasks", "seeds", "clip_lock", "holdout_access")


def validate_config(config: dict, *, require_assets: bool = False) -> dict:
    missing = [k for k in REQUIRED if k not in config]
    if missing:
        raise ValueError("BLOCKED_CONFIG:" + ",".join(missing))
    if config["route"] != "A_RASP" or config["holdout_access"] != "zero":
        raise ValueError("BLOCKED_PROTOCOL_SCOPE")
    if config["dataset"] not in {"isic", "hyperkvasir"}:
        raise ValueError("BLOCKED_DATASET")
    if config["tasks"] != [1, 2, 3, 4] or config["seeds"] != [1993, 1994, 1995]:
        raise ValueError("BLOCKED_TASK_OR_SEED_MATRIX")
    if float(config.get("lambda", 0)) != 0.001:
        raise ValueError("BLOCKED_LAMBDA")
    if require_assets:
        validate_encoder_lock(config["clip_lock"])
    return {"status": "ASSET_VERIFIED" if require_assets else "CONFIG_VALID", "training_started": False,
            "test_access": 0, "holdout_access": 0,
            "route": config["route"], "dataset": config["dataset"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate only")
    parser.add_argument("--execute", action="store_true", help="run the real train-only gate and CT13 launcher")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.execute:
        from route_a.run_ct13_real import execute
        raise SystemExit(execute(config, Path(config.get("output_dir", "ct13_output"))))
    result = validate_config(config, require_assets=not args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
