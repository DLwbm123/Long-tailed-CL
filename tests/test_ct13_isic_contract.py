import numpy as np

from route_a.spectral_prior_ridge import a0_to_a8, readout_scores


def test_a0_a1_scores_equal_original_branch_for_all_protocol_K():
    rng = np.random.default_rng(57002)
    for k in (2, 4, 6, 8):
        h = rng.normal(size=(k * 3, 2048))
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        S = np.eye(2048)
        M = rng.normal(scale=1e-4, size=(2048, k))
        text = rng.normal(size=(512, k))
        out = a0_to_a8(S, M, text, V=np.zeros_like(M), gamma=0.0,
                       ncomp=np.ones(k))
        a = np.sqrt(2.0) * h[:, :1536]
        u = np.sqrt(2.0) * h[:, 1536:]
        np.testing.assert_allclose(readout_scores(out, "A0", h), np.einsum("nd,dk->nk", a, out["A0"]), atol=1e-10)
        np.testing.assert_allclose(readout_scores(out, "A1", h), np.einsum("nd,dk->nk", u, out["A1"]), atol=1e-10)


def test_a3s_centering_is_text_only_and_zero_scale_reverts_to_a2():
    rng = np.random.default_rng(4)
    h = rng.normal(size=(7, 2048)); h /= np.linalg.norm(h, axis=1, keepdims=True)
    S = np.eye(2048); M = rng.normal(scale=1e-4, size=(2048, 4)); text = rng.normal(size=(512, 4))
    out = a0_to_a8(S, M, text, V=np.zeros_like(M), gamma=0.0,
                   a_scale=0.0, ncomp=np.ones(4))
    np.testing.assert_allclose(readout_scores(out, "A3s", h), readout_scores(out, "A2", h))
    assert out["A3s"].centered is True and out["A3"].centered is False
    tc = out["A3s"].text_u - out["A3s"].text_u.mean(axis=1, keepdims=True)
    assert np.allclose(tc.mean(axis=1), 0.0)
