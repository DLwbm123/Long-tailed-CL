"""Support-conditioned visual decision interface; implementation, no experiment launched."""
import torch
import torch.nn.functional as F

def prototypes(embeddings,labels,classes):
    """Equal empirical class means; callers supply only opposite-fold support."""
    counts=torch.bincount(labels,minlength=classes)
    assert bool((counts>0).all())
    means=torch.zeros(classes,embeddings.shape[1],device=embeddings.device,dtype=embeddings.dtype)
    means.index_add_(0,labels,embeddings)
    return F.normalize(means/counts[:,None],dim=1),counts

def query_risk(query,labels,support_prototypes,global_class_counts,total_rows):
    """Same all-seen decision for query training and inference; no text or fusion."""
    logits=20*F.normalize(query,dim=1)@support_prototypes.T
    classes=len(global_class_counts)
    weights=total_rows/(classes*global_class_counts[labels].to(query.dtype))
    return weights*F.cross_entropy(logits,labels,reduction='none'),logits
