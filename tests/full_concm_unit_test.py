#!/usr/bin/env python3
"""Focused tests for the full-session medical ConCM implementation."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from src.methods.concm_dsm import DSMProjector, compute_dynamic_structure
from src.methods.full_concm import (
    MPCNetwork,
    PrototypeRecord,
    class_balanced_mean,
    fixed_nc_geometry,
    resample_repository,
    train_projector_session,
    validate_continuity,
)
from tools.run_hyperkvasir_full_concm import restore_checkpoint, save_checkpoint


def record(class_id: int, session: int, dim: int = 8) -> PrototypeRecord:
    generator = torch.Generator().manual_seed(100 + class_id)
    mean = torch.randn(dim, generator=generator)
    covariance = torch.rand(dim, generator=generator) * 0.1 + 0.01
    return PrototypeRecord(class_id, session, mean, covariance, mean.clone(), 10 + class_id, "mid", session > 0)


class FullConCMUnitTest(unittest.TestCase):
    def test_centered_svd_etf_residual(self) -> None:
        torch.manual_seed(1)
        _, stats = compute_dynamic_structure(torch.randn(7, 16, dtype=torch.float64))
        self.assertLess(float(stats["etf_residual"]), 1e-10)

    def test_global_label_geometry_mapping(self) -> None:
        seen = [8, 3, 21]
        mapping = {class_id: head for head, class_id in enumerate(seen)}
        geometry = fixed_nc_geometry(5, 12)[:3]
        projected = geometry.clone()
        predictions = (projected @ geometry.T).argmax(dim=1).tolist()
        self.assertEqual(predictions, [mapping[class_id] for class_id in seen])

    def test_gaussian_resampling_is_epoch_deterministic(self) -> None:
        records = [record(4, 0), record(9, 1)]
        first, labels_first, stats_first = resample_repository(records, [9], seed=3, epoch=2)
        second, labels_second, stats_second = resample_repository(records, [9], seed=3, epoch=2)
        changed, _, stats_changed = resample_repository(records, [9], seed=3, epoch=3)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(labels_first, labels_second))
        self.assertEqual([row["checksum"] for row in stats_first], [row["checksum"] for row in stats_second])
        self.assertFalse(torch.equal(first, changed))
        self.assertNotEqual([row["checksum"] for row in stats_first], [row["checksum"] for row in stats_changed])

    def test_class_balanced_loss_ignores_sample_count(self) -> None:
        losses = torch.tensor([1.0] * 100 + [3.0] * 2)
        labels = torch.tensor([0] * 100 + [1] * 2)
        self.assertAlmostEqual(float(class_balanced_mean(losses, labels, 2)), 2.0)

    def test_checkpoint_continuity_contract(self) -> None:
        validate_continuity([8, 3], [8, 3, 21], {8: 0, 3: 1}, {8: 0, 3: 1, 21: 2})
        with self.assertRaisesRegex(RuntimeError, "order changed"):
            validate_continuity([8, 3], [3, 8, 21], {8: 0, 3: 1}, {3: 0, 8: 1, 21: 2})

    def test_checkpoint_sha_continuity(self) -> None:
        records = [record(0, 0), record(1, 0), record(2, 0)]
        projector = DSMProjector(input_dim=8, hidden_dim=16, output_dim=16)
        mpc = MPCNetwork(feature_dim=8, semantic_dim=6, hidden_dim=10)
        geometry, _ = compute_dynamic_structure(projector(torch.stack([item.mean for item in records])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session_0.pt"
            digest = save_checkpoint(
                path,
                0,
                "full_dynamic",
                projector,
                mpc,
                records,
                geometry,
                {0: 0, 1: 1, 2: 2},
                {0: "a", 1: "b", 2: "c"},
                None,
                Path("base.pt"),
                "base-sha",
                {},
            )
            restored = restore_checkpoint(path, digest, torch.device("cpu"))
            self.assertEqual(restored[0]["session"], 0)
            with path.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                restore_checkpoint(path, digest, torch.device("cpu"))

    def test_mpc_forward_backward_shapes(self) -> None:
        torch.manual_seed(2)
        network = MPCNetwork(feature_dim=8, semantic_dim=6, hidden_dim=10)
        completed, attention = network(torch.randn(4, 8), torch.randn(4, 6), torch.randn(7, 8), torch.randn(7, 6))
        self.assertEqual(tuple(completed.shape), (4, 8))
        self.assertEqual(tuple(attention.shape), (4, 7))
        F.mse_loss(completed, torch.randn(4, 8)).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in network.parameters()))

    def test_five_session_synthetic_smoke(self) -> None:
        torch.manual_seed(5)
        records = [record(0, 0), record(1, 0), record(2, 0)]
        projector = DSMProjector(input_dim=8, hidden_dim=16, output_dim=16)
        fixed = fixed_nc_geometry(total_classes=8, output_dim=16)
        previous_seen = [item.class_id for item in records]
        previous_mapping = {value: index for index, value in enumerate(previous_seen)}
        for session in range(1, 6):
            records.append(record(session + 2, session))
            current_seen = [item.class_id for item in records]
            current_mapping = {value: index for index, value in enumerate(current_seen)}
            validate_continuity(previous_seen, current_seen, previous_mapping, current_mapping)
            geometry, trace, augmentation = train_projector_session(
                projector,
                records,
                [session + 2],
                fixed,
                "nc_anchored",
                epochs=1,
                lr=0.005,
                seed=7,
                session=session,
                device=torch.device("cpu"),
            )
            self.assertEqual(tuple(geometry.shape), (3 + session, 16))
            self.assertTrue(torch.isfinite(geometry).all())
            self.assertEqual(len(trace), 1)
            self.assertTrue(augmentation)
            previous_seen = current_seen
            previous_mapping = current_mapping
        self.assertEqual(len(records), 8)


if __name__ == "__main__":
    unittest.main()
