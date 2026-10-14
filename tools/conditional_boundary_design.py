"""CPU design probe only: observed current-label errors, not inferred old errors."""
import math

import torch

import core1_competition as native

VERSION = 'conditional_boundary_graph_CPU_design_v1'


def finite(*values):
    if not values or any(v.dtype != torch.float64 or v.device != values[0].device or not torch.isfinite(v).all() for v in values):
        raise ValueError('Finite same-device float64 tensors required')


@torch.no_grad()
def evidence(x, labels, reference, geometry, current_labels):
    """Class-balanced actual OFF errors; old-old edges remain unidentifiable."""
    finite(x, reference, geometry)
    d, k = reference.shape
    current = set(current_labels)
    if (x.ndim != 2 or x.shape[1] != d or labels.dtype != torch.long or labels.shape != (len(x),)
            or labels.device != x.device or not len(x) or len(current) != len(current_labels)
            or not current or any(type(c) is not int or not 0 <= c < k for c in current)
            or set(labels.tolist()) != current or geometry.shape != (d,d)
            or not torch.allclose(geometry, geometry.T, atol=1e-12, rtol=1e-10)):
        raise ValueError('Only represented current fit labels and symmetric frozen geometry allowed')
    factor = torch.linalg.cholesky(geometry)
    L = reference.new_zeros(k,k); T = torch.zeros_like(reference); rows = []
    scores = x@reference; winner = scores.argmax(1)
    for c in sorted(current):
        count = int((labels == c).sum())
        for j in range(k):
            selected = (labels == c)&(winner == j)
            n = int(selected.sum())
            if j == c or not n:
                continue
            center = x[selected].mean(0)
            direction = torch.cholesky_solve(center[:,None], factor)[:,0]
            leverage = float(center@direction)
            if leverage <= 1e-12:
                rows.append(dict(label=c,rival=j,n=n,identified=False))
                continue
            v = reference.new_zeros(k); v[c]=1; v[j]=-1
            old = reference@v
            # Reuse the native unit pair margin; no val-selected margin/step.
            target = old+(1-float(center@old))/leverage*direction
            mass = n/count/len(current)
            L += mass*torch.outer(v,v); T += mass*torch.outer(target,v)
            rows.append(dict(label=c,rival=j,n=n,identified=True,mass=mass,
                             source_margin=float(center@old),target_margin=float(center@target)))
    return dict(version=VERSION,L=L,T=T,rows=rows,current_labels=sorted(current),
                old_old_error_identified=False,independent_calibration=False)


def validate_graph(graph, like, k):
    if graph.get('version') != VERSION:
        raise ValueError('Design checkpoint version mismatch')
    L,T=graph['L'],graph['T']; finite(like,L,T)
    if (L.shape != (k,k) or T.shape != like.shape or L.requires_grad or T.requires_grad
            or not torch.allclose(L,L.T,atol=1e-12,rtol=1e-10)
            or float(torch.linalg.eigvalsh(L).min()) < -1e-10
            or not torch.allclose(L.sum(1),L.new_zeros(k),atol=1e-10,rtol=0)
            or not torch.allclose(T.sum(1),T.new_zeros(len(T)),atol=1e-10,rtol=0)):
        raise ValueError('Detached PSD class-contrast graph required')


def solve(q,mu,mass,pairs,R,geometry,graph,strength,previous=None,proximal=0.):
    """Native quadratic plus a frozen PSD edge term; not a medical worker."""
    finite(q,mu,mass,pairs,R,geometry)
    k,d=mu.shape
    if (q.shape!=(k,d,d) or R.shape!=(d,d) or geometry.shape!=(d,d)
            or pairs.shape!=(k,k) or mass.shape!=(k,) or (mass<0).any()
            or (pairs<0).any() or pairs.diag().any()
            or not torch.allclose(pairs.sum(1),pairs.new_ones(k),atol=1e-10,rtol=1e-10)
            or not math.isfinite(strength) or strength<0
            or not math.isfinite(proximal) or proximal<0):
        raise ValueError('Invalid native dimensions, masses, competition or strength')
    for a in (R,geometry,*q):
        if not torch.allclose(a,a.T,atol=1e-12,rtol=1e-10):
            raise ValueError('Symmetric geometry required')
    torch.linalg.cholesky(geometry)
    validate_graph(graph,mu.T,k)
    baseline,audit=native.solve(q,mu,mass,pairs,regularizer=R,previous=previous,proximal=proximal)
    if strength==0 or not graph['L'].any():
        return baseline,audit
    G=q.sum(0)+.001*R+proximal*torch.eye(d,dtype=q.dtype,device=q.device)
    lap,targets=native.laplacians(pairs)
    rhs=mu.T+.5*mu.T@targets+strength*geometry@graph['T']
    if proximal:
        rhs+=proximal*previous
    def operator(w):
        return G@w+.5*torch.einsum('cdk,ckl->dl',q@w,lap)+strength*geometry@w@graph['L']
    factor=torch.linalg.cholesky(G)
    return native.pcg(operator,rhs,lambda z:torch.cholesky_solve(z,factor),baseline)


def penalty(w,geometry,graph,strength):
    return strength*(.5*(w*(geometry@w@graph['L'])).sum()-(w*(geometry@graph['T'])).sum())


def checkpoint(graph):
    return dict(graph,L=graph['L'].detach().cpu(),T=graph['T'].detach().cpu())


def restore(saved,like):
    graph=dict(saved,L=saved['L'].to(like),T=saved['T'].to(like))
    validate_graph(graph,like,like.shape[1])
    return graph
