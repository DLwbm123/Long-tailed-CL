"""Cross-patient visual-word self-classification and matched crop restoration."""
import copy,random
import torch
from torch import nn

ARMS=['NATIVE_RIDGE','FULL_IMAGE_SUPERVISED','WORD_RESTORATION','VISUAL_WORD_CLASS_RESTORATION']

def patchify(raw):
    return raw.reshape(len(raw),3,14,16,14,16).permute(0,2,4,1,3,5).reshape(len(raw),196,768)

def unpatchify(patch):
    return patch.reshape(len(patch),14,14,3,16,16).permute(0,3,1,4,2,5).reshape(len(patch),3,224,224)

def descriptor(z):
    v=torch.cat((z[:,0],z[:,1:].mean(1)),1);v=v/v.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return torch.cat((v,torch.ones_like(v[:,:1])),1)/(2**.5)

def shuffle_receipt(seed):
    rng=random.Random(seed);permutations=[]
    for _ in range(196):
        indices=list(range(256));rng.shuffle(indices);permutations.append(indices)
    return dict(seed=seed,permutations=permutations)

def corrupt(raw,receipt):
    x=patchify(raw).reshape(len(raw),196,3,256);ix=torch.tensor(receipt['permutations'],device=raw.device)[None,:,None,:].expand(len(raw),196,3,256)
    return unpatchify(x.gather(3,ix).reshape(len(raw),196,768))

def crop(raw,top,left):
    assert 0<=top<=112 and 0<=left<=112
    return torch.nn.functional.interpolate(raw[:,:,top:top+112,left:left+112],size=(224,224),mode='bilinear',align_corners=False)

class Model(nn.Module):
    def __init__(self,backbone,initial_head):
        super().__init__();self.backbone=copy.deepcopy(backbone).requires_grad_(True)
        self.decoder=nn.Linear(768,768).double();self.word_head=nn.Linear(1536,100).double();self.disease_head=nn.Parameter(initial_head.clone().double())
    def forward(self,raw):return self.backbone.forward_features((raw-.5)/.5)
    def objective_from_tokens(self,z,target,labels,arm):
        x=descriptor(z)[:,:1536]*(2**.5)
        if arm=='FULL_IMAGE_SUPERVISED':
            head=self.disease_head/self.disease_head.norm(dim=1,keepdim=True).clamp_min(1e-12);logits=25*x@head.T
            return (torch.logsumexp(logits,1)-logits.gather(1,labels[:,None]).squeeze(1)).mean()
        loss=(self.decoder(z[:,1:])-patchify(target)).square().mean()
        if arm=='VISUAL_WORD_CLASS_RESTORATION':
            logits=self.word_head(x);loss=loss+.01*(torch.logsumexp(logits,1)-logits.gather(1,labels[:,None]).squeeze(1)).mean()
        else:assert arm=='WORD_RESTORATION'
        return loss
    def objective(self,inputs,target,labels,arm):return self.objective_from_tokens(self(inputs),target,labels,arm)
