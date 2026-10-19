import torch


def training_rms(gram,head,classes):
    h=torch.eye(classes,dtype=head.dtype)-torch.ones(classes,classes,dtype=head.dtype)/classes
    centered=head@h
    value=float(torch.trace(centered.T@gram@centered))/(classes*classes)
    assert value>0
    return value**.5


def overlap(a,b):
    union=a|b
    return dict(A_count=len(a),J_count=len(b),shared=len(a&b),A_only=len(a-b),J_only=len(b-a),union=len(union),Jaccard=len(a&b)/len(union) if union else None)
