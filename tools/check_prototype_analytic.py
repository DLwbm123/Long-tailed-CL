"""Independent direct-example oracle for ridge and per-class moment transport."""
import numpy as np
from prototype_analytic import empty, append, ridge, move, transport

rng = np.random.default_rng(13)
x = rng.normal(size=(13, 5)); y = np.array([0] * 4 + [1] * 9)
bank = append(empty(5), x, y, [0, 1])
weights = 1 / (2 * np.bincount(y)[y])
expected = np.linalg.solve(x.T @ (weights[:, None] * x) + .001 * np.eye(5),
                           x.T @ (weights[:, None] * np.eye(2)[y]))
assert np.allclose(ridge(bank)[0], expected)
shift = rng.normal(size=(2, 5))
moved = move(bank, shift)
direct = append(empty(5), x + shift[y], y, [0, 1])
assert np.allclose(moved['S'], direct['S']) and np.allclose(moved['mu'], direct['mu'])
assert np.allclose(ridge(moved)[0], ridge(direct)[0])
# Uniform drift must be identical for global and local interpolation.
z = rng.normal(size=(11, 5)); labels = np.array([2] * 3 + [3] * 8); delta = rng.normal(size=5)
a, _ = transport(bank, z, z + delta, labels, [2, 3], False)
b, _ = transport(bank, z, z + delta, labels, [2, 3], True)
assert np.allclose(a['S'], b['S']) and np.allclose(a['mu'], b['mu'])
# Zero drift and a new task preserve exact joint class-balanced fitting.
joint = append(transport(bank, z, z, labels, [2, 3], True)[0], z, labels, [2, 3])
all_data = append(empty(5), np.r_[x, z], np.r_[y, labels], [0, 1, 2, 3])
assert np.allclose(ridge(joint)[0], ridge(all_data)[0])
print('PASS: weighted ridge, class-specific raw moments, uniform/zero drift, joint equivalence')
