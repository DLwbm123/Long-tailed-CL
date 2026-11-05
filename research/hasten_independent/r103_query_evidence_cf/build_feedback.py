"""Bank-specific detached routing tables; no labels in inference features."""
import torch
from router import features
from common import save
from fusion import scores

@torch.no_grad()
def table(z,P,models,nc,birth,stage,labels):
    ss=scores(z,P,models);phi=features(ss,z.cuda(),nc.cuda(),birth.cuda(),stage);correct=ss.argmax(-1)==labels.cuda()[:,None]
    return {'scores':ss.detach(),'features':phi.detach(),'correct':correct.detach()}

def persist(path,tab,ids,bank):
    save(path,{'ids':ids,'source_bank':bank,**{k:v.cpu() for k,v in tab.items()}})
