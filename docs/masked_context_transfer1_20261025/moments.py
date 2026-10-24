"""Equal class, equal patient and equal within-patient image normal systems."""
from collections import Counter
import numpy as np
import torch


def add_current(rows, features, classes, gram, means, record):
    for label in classes:
        ix = [i for i,r in enumerate(rows) if r['label'] == label]
        counts = Counter(rows[i]['group'] for i in ix)
        weight = torch.tensor([1/(len(counts)*counts[rows[i]['group']]) for i in ix], dtype=torch.float64)
        x = features[ix].double()
        current = x.T @ (weight[:,None]*x)
        mean = (weight[:,None]*x).sum(0)
        # Independently assign equal patient mass, then use a NumPy normal-system product.
        patient_positions = {}
        for j,i in enumerate(ix):patient_positions.setdefault(rows[i]['group'], []).append(j)
        reference_weights = np.empty(len(ix), dtype=np.float64)
        for positions in patient_positions.values():reference_weights[positions] = 1/(len(patient_positions)*len(positions))
        reference_x = x.numpy()
        independent = reference_x.T @ (reference_weights[:,None]*reference_x)
        independent_mean = np.sum(reference_weights[:,None]*reference_x, axis=0)
        error = max(float(abs(independent-current.numpy()).max()), float(abs(independent_mean-mean.numpy()).max()))
        record(dict(check='current_patient_class_moments', label=label, images=len(ix), patients=len(counts), error=error, PASS=error<=1e-9))
        assert error <= 1e-9
        gram += current;means.append(mean)


def solve(gram, means, ridge):
    c = len(means)
    matrix = gram/c + ridge*torch.eye(len(gram), dtype=torch.float64)
    rhs = torch.stack(means,1)/c
    head = torch.linalg.solve(matrix, rhs)
    return head, matrix, rhs
