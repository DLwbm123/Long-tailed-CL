"""Direct-example moment oracle, graph limits, and reliability checks."""
import numpy as np
from prototype_analytic import empty as single_empty, append as single_append, ridge
from prototype_graph import empty, append, move, partition, reliability, transport

rng = np.random.default_rng(62)
x = rng.normal(size=(80, 6)); x[40:] += 3.
y = np.r_[np.zeros(64, dtype=int), np.ones(16, dtype=int)]
bank = append(empty(6), x, y, [0, 1])
assert len(bank['components']) == 5
assert np.allclose(ridge(bank)[0], ridge(single_append(single_empty(6), x, y, [0, 1]))[0])
shifts = rng.normal(size=(len(bank['components']), 6)); moved = move(bank, shifts)
direct_x = x.copy(); offset = 0
for c in [0, 1]:
 rows = np.flatnonzero(y == c)
 for ids in partition(x[rows]):
  direct_x[rows[ids]] += shifts[offset]; offset += 1
expected = single_append(single_empty(6), direct_x, y, [0, 1])
assert np.allclose(moved['S'], expected['S']) and np.allclose(moved['mu'], expected['mu'])
assert np.allclose(ridge(moved)[0], ridge(expected)[0])
assert len(partition(np.ones((3, 6)))) == 1
assert len(partition(np.ones((80, 6)))) == 1
z = rng.normal(size=(64, 6)); label = np.repeat([2, 3], 32); d = np.ones(6)*.2
uniform, audit = transport(bank, z, z+d, label, [2, 3])
expected = move(bank, np.tile(d, (len(bank['components']), 1)))
assert np.allclose(uniform['S'], expected['S']) and audit['graph_residual'] < 1e-7
zero, _ = transport(bank, z, z, label, [2, 3]); assert np.allclose(zero['S'], bank['S'])
noise = rng.normal(size=(40, 6));noise -= noise.mean(0)
assert reliability(np.tile(d, (40, 1))) > reliability(np.tile(d, (40, 1))+noise)
assert reliability(d[None, :]) == 0
# Current-class statistics remain exact irrespective of mixture counts.
joined = append(zero, z, label, [2, 3])
full = single_append(single_empty(6), np.r_[x, z], np.r_[y, label], [0, 1, 2, 3])
assert np.allclose(ridge(joined)[0], ridge(full)[0])
print('PASS: mixture moments, graph zero/uniform limits, support/noise reliability, ridge equivalence')
# Heterogeneous reliable anchors must produce heterogeneous old-node drift.
a = rng.normal(scale=.1, size=(32, 6)); b = rng.normal(scale=.1, size=(32, 6))+3
old = append(empty(6), np.r_[a, b], np.repeat([0, 1], 32), [0, 1])
current = np.r_[a+.01, b+.01]; directions = np.r_[np.tile(d, (32, 1)), np.tile(-d, (32, 1))]
propagated, _ = transport(old, current, current+directions, np.repeat([2, 3], 32), [2, 3])
assert np.dot(propagated['mu'][0]-old['mu'][0], d) > 0
assert np.dot(propagated['mu'][1]-old['mu'][1], d) < 0
print('PASS: graph propagates different reliable local drift directions')
