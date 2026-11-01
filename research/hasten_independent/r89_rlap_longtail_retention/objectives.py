"""Matched class risk and previous-stage functional retention; no task oracle."""
import torch
import torch.nn.functional as F

def classification(scores, difficulty, labels, training_counts, macro):
    n=len(training_counts)
    weight=training_counts.sum()/(n*training_counts[labels]) if macro else torch.ones_like(labels,dtype=scores.dtype)
    return (weight*(1+difficulty.gather(1,labels[:,None]).squeeze(1))*F.cross_entropy(scores,labels,reduction='none')).mean()

def retention(scores, teacher_old_scores, labels, training_counts, old, total_training_rows):
    mask=labels<old
    if old==0 or not mask.any():return scores.sum()*0
    target=teacher_old_scores[mask].detach().softmax(1)
    log_target=teacher_old_scores[mask].detach().log_softmax(1)
    log_current=scores[mask].log_softmax(1)[:,:old]
    # Zero probability assigned to all new classes: old-to-new stealing is penalized.
    divergence=(target*(log_target-log_current)).sum(1)
    w=total_training_rows/(old*training_counts[labels[mask]])
    # Mean over the full microbatch estimates the class-mean OLD replay objective.
    return (w*divergence).sum()/len(labels)
