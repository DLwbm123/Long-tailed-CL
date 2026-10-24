"""Independent pure-standard-library objective and analytic output derivative."""
import math


def objective_and_grad(first, second):
    # Canonical reference format: batch x raster-position x dimension, including P=1.
    n, p, d = len(first), len(first[0]), len(first[0][0])
    assert n > 1 and p > 0 and d > 1
    values = (first, second)
    grad = [[[[0. for _ in range(d)] for _ in range(p)] for _ in range(n)] for _ in range(2)]
    loss = 0.
    for i in range(n):
        for k in range(p):
            for j in range(d):
                delta = first[i][k][j]-second[i][k][j]
                loss += 25*delta*delta/(n*p*d)
                g = 50*delta/(n*p*d)
                grad[0][i][k][j] += g
                grad[1][i][k][j] -= g
    for view, value in enumerate(values):
        for k in range(p):
            mu = [sum(value[i][k][j] for i in range(n))/n for j in range(d)]
            centered = [[value[i][k][j]-mu[j] for j in range(d)] for i in range(n)]
            cov = [[sum(centered[i][j]*centered[i][l] for i in range(n))/(n-1) for l in range(d)] for j in range(d)]
            for j in range(d):
                std = math.sqrt(cov[j][j]+1e-4)
                loss += 12.5*max(0., 1-std)/(p*d)
                if std < 1:
                    for i in range(n):grad[view][i][k][j] -= 12.5*centered[i][j]/(p*d*(n-1)*std)
                loss += sum(cov[j][l]**2 for l in range(d) if j != l)/(p*d)
                for i in range(n):
                    grad[view][i][k][j] += 4*sum(centered[i][l]*cov[l][j] for l in range(d) if l != j)/(p*d*(n-1))
    return loss, grad
