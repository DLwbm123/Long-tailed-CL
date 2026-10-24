"""Full native ViT masked-context prediction followed by supervised consolidation."""
import copy,random
import torch
from torch import nn

ARMS=['NATIVE_RIDGE','SUPERVISED_ONLY','VISIBLE_RECON_THEN_SUPERVISED','MASKED_CONTEXT_THEN_SUPERVISED']

def mask_receipt(seed):
    cells=random.Random(seed).sample(range(49),30)
    tokens=sorted((2*(c//7)+dy)*14+2*(c%7)+dx for c in cells for dy in [0,1] for dx in [0,1])
    assert len(tokens)==len(set(tokens))==120
    return dict(seed=seed,cells=cells,tokens=tokens)

def patchify(raw):
    assert raw.ndim==4 and raw.shape[1:]==(3,224,224)
    return raw.reshape(len(raw),3,14,16,14,16).permute(0,2,4,1,3,5).reshape(len(raw),196,768)

def descriptor(tokens):
    z=torch.cat((tokens[:,0],tokens[:,1:].mean(1)),dim=1)
    z=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return torch.cat((z,torch.ones_like(z[:,:1])),dim=1)/(2**.5)

class Model(nn.Module):
    def __init__(self,backbone,initial_head):
        super().__init__();self.backbone=copy.deepcopy(backbone).requires_grad_(True)
        self.decoder=nn.Linear(768,768);self.decoder.double()
        self.head=nn.Parameter(initial_head.clone().double())
    def forward(self,raw,receipt=None,masked=False):
        b=self.backbone;x=b.patch_embed((raw-.5)/.5)
        if masked:
            x=x.clone();x[:,receipt['tokens']]=0
        x=b.norm_pre(b.patch_drop(b._pos_embed(x)))
        return b.norm(b.blocks(x))
    def objective(self,raw,labels,receipt,arm,phase):
        assert arm in ARMS[1:] and phase in ['A','B']
        masked=phase=='A' and arm=='MASKED_CONTEXT_THEN_SUPERVISED'
        tokens=self(raw,receipt,masked)
        if phase=='A' and arm!='SUPERVISED_ONLY':
            target=patchify(raw)[:,receipt['tokens']]
            pred=self.decoder(tokens[:,1:][:,receipt['tokens']])
            return (pred-target).abs().mean()
        feature=descriptor(tokens)[:,:1536]*(2**.5)
        head=self.head/self.head.norm(dim=1,keepdim=True).clamp_min(1e-12)
        logits=25*feature@head.T
        return (torch.logsumexp(logits,dim=1)-logits.gather(1,labels[:,None]).squeeze(1)).mean()
