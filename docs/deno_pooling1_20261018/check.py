"""Independent determinant oracle and invariance checks, no medical data."""
import torch
from deno import kernel, determinant_evidence


torch.manual_seed(74002)
torch.set_default_dtype(torch.float64)
x = torch.randn(3, 6, 5)
e = torch.randn(3, 6, 4)
k = kernel(x, .7)
a = determinant_evidence(k, e)
identity = torch.eye(6).expand(3, -1, -1)
assert torch.allclose(determinant_evidence(identity, e), e.sum(1), atol=1e-10)
assert torch.allclose(determinant_evidence(k, -e), -a, atol=1e-10)
perm = torch.tensor([2, 0, 4, 1, 5, 3])
assert torch.allclose(determinant_evidence(k[:, perm][:, :, perm], e[:, perm]), a, atol=1e-10)
# Exact whole-bag uniform replication, including kernel re-normalization.
rep_x = x.repeat_interleave(3, 1)
rep_e = e.repeat_interleave(3, 1)
assert torch.allclose(determinant_evidence(kernel(rep_x, .7), rep_e), a, atol=1e-10)
# slogdet is an independent dense oracle rather than Cholesky logdet.
expected = torch.zeros_like(a)
for i in range(3):
    for c in range(4):
        for sign in [1, -1]:
            d = torch.diag((sign * e[i, :, c]/4).exp())
            determinant_sign, logdet = torch.linalg.slogdet(torch.eye(6) + d @ k[i] @ d)
            assert determinant_sign == 1
            expected[i, c] += sign * 2 * logdet
assert torch.allclose(a, expected, atol=1e-10)
# Equal class scores use first argmax; identical instances are well defined.
ones = kernel(torch.ones_like(x), .7)
assert torch.isfinite(determinant_evidence(ones, e)).all()
print('PASS: slogdet oracle; identity additive reduction; oddness; permutation; uniform replication; identical-token kernel')
