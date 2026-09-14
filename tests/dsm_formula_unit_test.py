#!/usr/bin/env python3
"""Focused numerical tests for the ConCM-style DSM construction."""

from __future__ import annotations

import unittest

import torch

from src.methods.concm_dsm import compute_dynamic_structure


class DynamicStructureFormulaTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260710)
        self.prototypes = torch.randn(11, 32, dtype=torch.float64)

    def test_full_rank_input_has_machine_precision_etf_residual(self) -> None:
        _, stats = compute_dynamic_structure(self.prototypes)
        self.assertLess(float(stats["etf_residual"]), 1e-10)
        self.assertEqual(int(stats["numerical_rank"]), 10)

    def test_class_permutation_preserves_row_mapping(self) -> None:
        geometry, _ = compute_dynamic_structure(self.prototypes)
        permutation = torch.tensor([5, 1, 8, 0, 10, 3, 7, 2, 9, 6, 4])
        permuted_geometry, _ = compute_dynamic_structure(self.prototypes[permutation])
        self.assertTrue(torch.allclose(permuted_geometry, geometry[permutation], atol=1e-10, rtol=1e-10))

    def test_repeated_construction_is_deterministic(self) -> None:
        first, first_stats = compute_dynamic_structure(self.prototypes)
        second, second_stats = compute_dynamic_structure(self.prototypes.clone())
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_stats["singular_values"], second_stats["singular_values"])

    def test_projection_dimension_must_exceed_class_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "dg > number of classes N"):
            compute_dynamic_structure(torch.randn(8, 8))
        with self.assertRaisesRegex(ValueError, "dg > number of classes N"):
            compute_dynamic_structure(torch.randn(8, 7))


if __name__ == "__main__":
    unittest.main()
