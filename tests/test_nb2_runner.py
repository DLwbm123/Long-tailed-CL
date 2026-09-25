import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from route_a.run_ct13_real import _extract_rows
from route_a.spectral_prior_ridge import RASPReadout, a0_to_a8, readout_scores, ridge, spectral_prior_ridge
from shared.dual_moment_bank import DualMomentBank
from tools.run_nb2_vlm_r1 import _allow, _guard, _pairs, _p_add, _p_empty, _seal, _resume_stage


class RunnerTest(unittest.TestCase):
    def test_duplicate_json_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "BLOCKED_DUPLICATE_JSON_KEY"):
            json.loads('{"tasks": [2], "tasks": [4]}', object_pairs_hook=_pairs)

    def test_p_branch_ridge_matches_class_balanced_direct_fit(self):
        bank = _p_empty()
        x0 = np.vstack((np.eye(768)[0], np.eye(768)[0]))
        x1 = np.vstack((np.eye(768)[1], np.eye(768)[1]))
        _p_add(bank, 9, x0)
        _p_add(bank, 4, x1)
        weight = ridge(bank["S"] / 2, np.column_stack(bank["means"]) / 2)
        self.assertEqual(bank["ids"], [9, 4])
        np.testing.assert_allclose(x0 @ weight, np.array([[1 / 1.002, 0]] * 2), rtol=1e-12)

    def test_zero_prior_strength_matches_plain_ridge(self):
        S = np.diag([3., 2., 1.])
        M = np.array([[.3, .1], [.1, .2], [.2, .3]])
        V = np.ones_like(M)
        got, diag = spectral_prior_ridge(S, M, V, 0.)
        expected = ridge(S / 2, M / 2)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
        self.assertLessEqual(diag["max_relative_residual"], 1e-8)

    def test_a0_a1_scaling_and_zero_gamma_on_joint_stream(self):
        a = np.zeros((4, 1536)); u = np.zeros((4, 512))
        a[np.arange(4), [0, 0, 1, 1]] = 1
        u[np.arange(4), [0, 1, 0, 1]] = 1
        h = np.concatenate((a, u), axis=1) / np.sqrt(2)
        bank = DualMomentBank()
        bank.add_class(9, h[:2], task=1, component_ids=["g0", "g1"])
        bank.add_class(4, h[2:], task=1, component_ids=["g2", "g3"])
        S, M, ids = bank.class_balanced()
        self.assertEqual(ids.tolist(), [9, 4])
        text = np.eye(512)[:, :2]
        outputs = a0_to_a8(S, M, text, V=np.zeros_like(M), gamma=0.,
                           a_scale=0., ncomp=np.ones(2))
        ga = (a[:2].T @ a[:2] / 2 + a[2:].T @ a[2:] / 2) / 2
        ra = np.column_stack((a[:2].mean(0), a[2:].mean(0))) / 2
        gu = (u[:2].T @ u[:2] / 2 + u[2:].T @ u[2:] / 2) / 2
        ru = np.column_stack((u[:2].mean(0), u[2:].mean(0))) / 2
        np.testing.assert_allclose(readout_scores(outputs, "A0", h), a @ ridge(ga, ra))
        np.testing.assert_allclose(readout_scores(outputs, "A1", h), u @ ridge(gu, ru))
        np.testing.assert_allclose(outputs["A6"], outputs["A2"], atol=1e-10)

    def test_sealed_stage_restores_and_rejects_drift(self):
        bank = DualMomentBank()
        x0 = np.zeros((2, 2048)); x0[:, 0] = 1
        x1 = np.zeros((2, 2048)); x1[:, 1536] = 1
        bank.add_class(9, x0, task=1, component_ids=["a", "b"])
        bank.add_class(4, x1, task=1, component_ids=["c", "d"])
        pbank = _p_empty()
        _p_add(pbank, 9, np.ones((2, 768)) / np.sqrt(768))
        _p_add(pbank, 4, -np.ones((2, 768)) / np.sqrt(768))
        text = np.eye(512)[:, :2]
        readouts = {name: np.zeros((2048, 2)) for name in
                    ("A0", "A1", "A2", "A4", "A5", "A6", "A7", "A8")}
        readouts["A3"] = RASPReadout(readouts["A2"], text, 1.)
        readouts["A3s"] = RASPReadout(readouts["A2"], text, 1., centered=True)
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / "task_01"
            lock = {"source_sha256": "source", "protocol_sha256": "protocol", "class_order": [9, 4]}
            _seal(stage, bank, pbank, readouts, np.zeros((768, 2)), text,
                  1., np.zeros(2), np.ones(2), lock)
            restored, prior = _resume_stage(stage, "source", "protocol", [9, 4])
            self.assertEqual(restored.class_order, [9, 4])
            self.assertEqual(prior["ids"], [9, 4])
            with self.assertRaisesRegex(ValueError, "BLOCKED_STAGE_LOCK"):
                _resume_stage(stage, "source", "protocol", [4, 9])
            with self.assertRaises(FileExistsError):
                _seal(stage, bank, pbank, readouts, np.zeros((768, 2)), text,
                      1., np.zeros(2), np.ones(2), lock)

    def test_stream_batch_equivalence_and_future_image_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for i, color in enumerate((20, 80, 160)):
                name = f"{i}.png"
                Image.new("RGB", (2, 2), (color, 0, 0)).save(root / name)
                rows.append({"relative_path": name})
            def preprocess(image):
                return np.asarray(image, dtype=np.float32)[0, 0]
            def apart_forward(batch):
                main = np.zeros((len(batch), 768), np.float32)
                few = np.zeros_like(main)
                main[:, 0] = batch[:, 0] + 1
                few[:, 1] = batch[:, 0] + 2
                return {"pre_logits": main, "pre_logits_few": few}
            def clip_forward(batch):
                out = np.zeros((len(batch), 512), np.float32)
                out[:, 0] = batch[:, 0] + 3
                out[:, 1] = 1
                return out
            apart = {"preprocess": preprocess, "forward": apart_forward}
            clip = {"preprocess": preprocess, "forward": clip_forward}
            allowed = _guard()
            _allow(allowed, root, rows[:2])
            with self.assertRaisesRegex(ValueError, "BLOCKED_OLD_FUTURE_OR_TEST_IMAGE"):
                (root / "2.png").open("rb")
            with self.assertRaisesRegex(ValueError, "BLOCKED_TEST_RESERVED_ACCESS"):
                (root / "test.csv").open("w")
            _allow(allowed, root, rows)
            one = _extract_rows(rows, root, apart, clip, batch_size=1)
            many = _extract_rows(rows, root, apart, clip, batch_size=3)
            for left, right in zip(one, many):
                np.testing.assert_allclose(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
