"""CT14-ACTM protocol validator (training is gated and not implicit)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from shared.frozen_dual_features import validate_encoder_lock
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from shared.frozen_dual_features import validate_encoder_lock


def validate_config(config: dict, *, require_assets: bool = False) -> dict:
    required = ("route", "dataset", "variants", "seeds", "support_query", "holdout_access", "clip_lock")
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError("BLOCKED_CONFIG:" + ",".join(missing))
    if config["route"] != "B_ACTM" or config["holdout_access"] != "zero":
        raise ValueError("BLOCKED_PROTOCOL_SCOPE")
    if config["dataset"] != "isic" or set(config["variants"]) != {"B00", "B01", "B10", "B11"}:
        raise ValueError("BLOCKED_FINITE_MATRIX")
    if require_assets:
        validate_encoder_lock(config["clip_lock"])
    return {"status": "QUALIFIED_DRY_RUN", "training_started": False,
            "test_access": 0, "holdout_access": 0,
            "route": config["route"], "dataset": config["dataset"],
            "trajectories": len(config["variants"]) * len(config["seeds"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate only; training is never implicit")
    args = parser.parse_args()
    result = validate_config(json.loads(Path(args.config).read_text()), require_assets=not args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
