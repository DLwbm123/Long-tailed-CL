"""Prior-consistent support likelihood; count prior enters TRAIN only."""
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
    """Natural TRAIN posterior; return uniform-prior logits for inference."""
    logits=20*query@support_prototypes.T
    assert bool((global_class_counts>0).all()) and int(global_class_counts.sum())==total_rows
    train_logits=logits+global_class_counts.to(logits.dtype).log()[None,:]
    return F.cross_entropy(train_logits,labels,reduction='none'),logits
