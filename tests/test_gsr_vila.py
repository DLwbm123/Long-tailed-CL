import numpy as np
import pytest

from shared.dual_moment_bank import DualMomentBank
from shared.frozen_dual_features import build_joint_feature
from route_a.semantic_prior import build_prior, semantic_scale
from route_a.spectral_prior_ridge import ridge, spectral_prior_ridge
from route_b.affine_moment_transport import transport_old_bank
from route_b.anchor_drift import fit_anchor_drift


@pytest.fixture(autouse=True)
def _restore_torch_default_dtype():
    import torch
    before = torch.get_default_dtype()
    yield
    torch.set_default_dtype(before)


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
    barS, barM = bank.homogeneous()
    outS, outM = transport_old_bank(barS, barM, np.zeros((2, 4)), np.zeros(4), dim_a=4, dim_u=2)
    np.testing.assert_allclose(outS, barS[:-1, :-1] / 2)
    np.testing.assert_allclose(outM, barM[:-1] / 2)
    B = rng.normal(size=(2, 4)) * .01; b = rng.normal(size=4) * .01
    gotS, gotM = transport_old_bank(barS, barM, B, b, dim_a=4, dim_u=2)
    H = np.eye(7); H[:4, 4:6] = B.T; H[:4, -1] = b
    Z = np.column_stack([X[y == c].mean(0) for c in [0, 1]])
    augS = barS
    expectedS = (H @ augS @ H.T)[:-1, :-1] / 2
    expectedM = (H @ barM)[:-1] / 2
    np.testing.assert_allclose(gotS, expectedS)
    np.testing.assert_allclose(gotM, expectedM)


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


def test_synthetic_actm_variants_step_and_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from route_b.torch_actm import actm_episode_loss
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(12)
    adapter = torch.nn.Linear(5, 3, bias=False)
    teacher_adapter = torch.nn.Linear(5, 3, bias=False)
    teacher_adapter.load_state_dict(adapter.state_dict())
    for p in teacher_adapter.parameters():
        p.requires_grad_(False)
    inp = torch.randn((12, 5), generator=g)
    anchor = torch.randn((12, 3), generator=g)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    W = torch.randn((6, 4), generator=g) * .1
    old_mean = torch.randn((6, 2), generator=g) * .1
    old_cov = torch.eye(6) * .2
    opt = torch.optim.SGD(adapter.parameters(), lr=.01)
    before_teacher = [p.detach().clone() for p in teacher_adapter.parameters()]
    for variant in ("B00", "B01", "B10", "B11"):
        opt.zero_grad(set_to_none=True)
        total, parts = actm_episode_loss(adapter(inp), teacher_adapter(inp), anchor, labels, W, old_mean, old_cov, variant=variant)
        total.backward()
        assert torch.isfinite(total) and all(torch.isfinite(p.grad).all() for p in adapter.parameters())
        opt.step()
        assert parts["b"].requires_grad
        if variant in ("B10", "B11"):
            assert parts["B"].requires_grad
    assert all(torch.equal(a, b) for a, b in zip(before_teacher, teacher_adapter.parameters()))
    state = {"adapter": adapter.state_dict(), "optimizer": opt.state_dict(), "step": 4}
    path = tmp_path / "synthetic_actm.pt"
    torch.save(state, path)
    restored = torch.nn.Linear(5, 3, bias=False)
    restored_opt = torch.optim.SGD(restored.parameters(), lr=.01)
    loaded = torch.load(path, weights_only=True)
    restored.load_state_dict(loaded["adapter"]); restored_opt.load_state_dict(loaded["optimizer"])
    assert loaded["step"] == 4
    for a, b in zip(adapter.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
