import numpy as np
import pytest

from shared.dual_moment_bank import DualMomentBank
from shared.frozen_dual_features import build_joint_feature
from route_a.semantic_prior import build_prior, semantic_scale
from route_a.spectral_prior_ridge import ridge, spectral_prior_ridge
from route_b.affine_moment_transport import transport_old_bank
from route_b.anchor_drift import fit_anchor_drift


def toy(seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(18, 1536)), rng.normal(size=(18, 512))


def test_joint_feature_and_complete_cross_moment_roundtrip():
    a, u = toy()
    f = build_joint_feature(a, u)
    assert f.h.shape == (18, 2048)
    bank = DualMomentBank()
    bank.add_class(0, f.h[:9], task=1, component_ids=[str(i % 3) for i in range(9)])
    bank.add_class(1, f.h[9:], task=1, component_ids=[str(i % 3) for i in range(9)])
    S, M, ids = bank.class_balanced()
    assert ids.tolist() == [0, 1]
    with np.errstate(all="ignore"):
        expected_cross = (f.h[:9].T @ f.h[:9] / 9 + f.h[9:].T @ f.h[9:] / 9)[:1536, 1536:]
    np.testing.assert_allclose(S[:1536, 1536:], expected_cross)
    assert bank.state_dict()["dim_a"] == 1536


def test_rasp_spectral_solution_and_prior():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(20, 8)); y = np.repeat(np.arange(4), 5)
    S = sum((X[y == c].T @ X[y == c] / 5 for c in range(4)), np.zeros((8, 8)))
    M = np.column_stack([X[y == c].mean(0) for c in range(4)])
    V = np.zeros_like(M); W, audit = spectral_prior_ridge(S, M, V, 0.0)
    np.testing.assert_allclose(W, ridge(S / 4, M / 4), atol=1e-10)
    assert audit["max_residual"] < 1e-8
    text = np.eye(4)
    assert build_prior(text, 1.0, dim_a=4).shape == (8, 4)
    assert semantic_scale(text, np.eye(4), text) >= 0


def test_transport_keeps_cross_block_and_identity():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(12, 6)); y = np.repeat([0, 1], 6)
    bank = DualMomentBank(dim_a=4, dim_u=2)
    bank.add_class(0, X[:6], task=1)
    bank.add_class(1, X[6:], task=1)
    S, M, _ = bank.class_balanced()
    barS, barM = S / 2, M / 2
    outS, outM = transport_old_bank(barS, barM, np.zeros((2, 4)), np.zeros(4), dim_a=4, dim_u=2)
    np.testing.assert_allclose(outS, barS)
    np.testing.assert_allclose(outM, barM)
    B = rng.normal(size=(2, 4)) * .01; b = rng.normal(size=4) * .01
    gotS, gotM = transport_old_bank(barS, barM, B, b, dim_a=4, dim_u=2)
    H = np.eye(7); H[:4, 4:6] = B.T; H[:4, -1] = b
    Z = np.column_stack([X[y == c].mean(0) for c in [0, 1]])
    augS = np.zeros((7, 7)); augS[:-1, :-1] = barS; augS[:-1, -1] = augS[-1, :-1] = barM.mean(1); augS[-1, -1] = 1
    np.testing.assert_allclose(gotS, (H @ augS @ H.T)[:-1, :-1])
    np.testing.assert_allclose(gotM, (H @ np.vstack((barM, np.ones((1, 2)))))[:-1])


def test_anchor_fit_and_protocol_rejection():
    rng = np.random.default_rng(10)
    v = rng.normal(size=(30, 3)); B = rng.normal(size=(3, 5)) * .01; b = rng.normal(size=5) * .01
    d = v @ B + b
    gotB, gotb, audit = fit_anchor_drift(v, d)
    assert audit["B_norm"] <= .25 + 1e-12 and audit["b_norm"] <= .05 + 1e-12
    with pytest.raises(ValueError):
        build_joint_feature(np.zeros((2, 4)), np.ones((3, 2)))


def test_torch_anchor_keeps_student_gradient():
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(11)
    teacher = torch.randn((12, 3), generator=g)
    student = (teacher[:, :2] * .02 + torch.randn((12, 2), generator=g) * .01).requires_grad_(True)
    from route_b.anchor_drift import torch_anchor_drift
    B, b = torch_anchor_drift(teacher, student)
    loss = (B.square().sum() + b.square().sum())
    grad = torch.autograd.grad(loss, student)[0]
    assert torch.isfinite(grad).all() and float(torch.linalg.vector_norm(grad)) > 0
