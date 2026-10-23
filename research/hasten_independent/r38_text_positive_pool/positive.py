"""Text-anchor probability mass for all same-class images in the batch."""
import torch
def pooled_text_ce(logits,labels):
 assert logits.ndim==2 and logits.shape==(len(labels),len(labels)) and len(labels)>0
 mask=labels[:,None].eq(labels[None,:]);assert mask.diagonal().all()
 return (torch.logsumexp(logits,dim=1)-torch.logsumexp(logits.masked_fill(~mask,-torch.inf),dim=1)).mean()
