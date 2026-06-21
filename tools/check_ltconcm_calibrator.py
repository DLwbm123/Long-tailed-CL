#!/usr/bin/env python3
"""Small standalone checks for TaConCM prototype calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.methods.ltconcm import PrototypeMemory, ReliabilityAwarePrototypeCalibrator


def main() -> None:
    torch.manual_seed(0)
    class_ids = [10, 11, 12, 13]
    proto_raw = torch.randn(len(class_ids), 8)
    counts = torch.tensor([500, 100, 20, 5], dtype=torch.float32)
    proto_var = torch.rand(len(class_ids), 8) * 0.1

    memory = PrototypeMemory()
    memory_ids = [0, 1, 2]
    memory_raw = F.normalize(torch.randn(len(memory_ids), 8), dim=1)
    memory_counts = torch.tensor([500, 400, 350], dtype=torch.float32)
    memory_unc = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32)
    memory.update(
        class_ids=memory_ids,
        proto_raw=memory_raw,
        proto_calib=memory_raw,
        counts=memory_counts,
        uncertainty=memory_unc,
        session_id=0,
    )

    calibrator = ReliabilityAwarePrototypeCalibrator()
    result = calibrator.calibrate(
        proto_raw=proto_raw,
        class_counts=counts,
        proto_var=proto_var,
        prototype_memory=memory,
        class_ids=class_ids,
    )

    assert result.proto_calib.shape == proto_raw.shape
    assert result.alpha.shape == counts.shape
    assert torch.isfinite(result.proto_calib).all()
    assert torch.isfinite(result.alpha).all()
    assert torch.all(result.alpha >= 0.15)
    assert torch.all(result.alpha <= 0.95)
    assert torch.allclose(torch.linalg.vector_norm(result.proto_calib, dim=1), torch.ones(len(class_ids)), atol=1e-5)
    assert result.memory_used.float().mean().item() > 0.0

    empty_result = calibrator.calibrate(
        proto_raw=proto_raw,
        class_counts=counts,
        proto_var=None,
        prototype_memory=PrototypeMemory(),
        class_ids=class_ids,
    )
    assert torch.isfinite(empty_result.proto_calib).all()
    assert not empty_result.memory_used.any()
    assert torch.allclose(torch.linalg.vector_norm(empty_result.proto_calib, dim=1), torch.ones(len(class_ids)), atol=1e-5)
    print("ltconcm calibrator checks passed")


if __name__ == "__main__":
    main()
