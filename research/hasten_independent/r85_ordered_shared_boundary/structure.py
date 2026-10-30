"""Explicit label-neighbor prior, constructed without query outcomes."""
import torch
ORDER=['ulcerative colitis grade 0 1','ulcerative colitis grade 1',
       'ulcerative colitis grade 1 2','ulcerative colitis grade 2',
       'ulcerative colitis grade 2 3','ulcerative colitis grade 3']

def matrices(names):
    nodes=sorted([i for i,s in enumerate(names) if s.startswith('ulcerative colitis')],key=lambda c:ORDER.index(names[c]))
    n=len(names);k=len(nodes);assert k>=3
    adjacency=torch.zeros(n,n,dtype=torch.double)
    for a,b in zip(nodes,nodes[1:]):adjacency[a,b]=adjacency[b,a]=1
    D=torch.diag(adjacency.sum(1));ordered=D-adjacency
    complete=torch.zeros_like(adjacency)
    for a in nodes:
        for b in nodes:
            if a!=b:complete[a,b]=2/k
    mean=torch.diag(complete.sum(1))-complete
    assert abs(float(adjacency.sum()/2)-(k-1))<1e-12
    assert abs(float(complete.sum()/2)-(k-1))<1e-12
    assert torch.equal(ordered.diag(),D.diag())
    for L in [ordered,D,mean]:
        assert torch.equal(L,L.T) and float(torch.linalg.eigvalsh(L).min())>=-1e-10
    return dict(ordered=ordered,degree_only=D,family_mean=mean),nodes
