"""Two independent supervised fusion heads; exact full-fit gradient per update."""
import torch,time
from torch import nn
import torch.nn.functional as F
from common import save,write,guard

class Fusion(nn.Module):
    def __init__(self):
        super().__init__();self.V=nn.Linear(1024,16,bias=False);self.U=nn.Linear(16,512,bias=False);nn.init.zeros_(self.U.weight)
    def forward(self,z,v,P):return 20*F.normalize(z+self.U(F.gelu(self.V(torch.cat([z,v],-1)))),dim=-1)@P.T

def loss_part(model,z,y,P,nc,total,class_counts):
    score=model(z[:,0],z[:,1],P);teacher=20*z[:,0]@P.T
    ce=F.cross_entropy(score+nc.log(),y,reduction='sum')/total
    lp=score.double().log_softmax(-1);lt=teacher.double().log_softmax(-1)
    kl=(lt.exp()*(lt-lp)).sum(-1)
    return ce+.1*(kl/class_counts[y].double()).sum()/len(nc)

def full_step(model,opt,z,y,P,nc,clip=True):
    opt.zero_grad(set_to_none=True);N=len(y)
    for lo in range(0,N,64):loss_part(model,z[lo:lo+64],y[lo:lo+64],P,nc,N,nc).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.) if clip else torch.tensor(0.)
    opt.step();return float(norm)

def train(c,folder,features,y,P,seed,counts):
    models=[];records=[]
    nc=torch.bincount(y,minlength=len(P)).float();assert (nc>0).all()
    for action in [1,2]:
        torch.manual_seed(seed);m=Fusion().cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01)
        z=features[:,[0,action]].cuda();target=y.cuda();p=P.cuda();cc=nc.cuda()
        with torch.no_grad():
            a=m(z[:,0],z[:,1],p);t=20*z[:,0]@p.T
            assert (a-t).abs().max()<=1e-4 and torch.equal(a.argmax(1),t.argmax(1))
        save(folder/f'fusion{action}_step0.private.pt',{'model':m.state_dict(),'optimizer':opt.state_dict()})
        wall=time.time();cpu=time.process_time();norms=[]
        for step in range(1,513):
            guard(c);norms.append(full_step(m,opt,z,target,p,cc));counts['fusion_optimizer']+=1;counts['fusion_fit_rows']+=len(y)
            if step in [256,512]:save(folder/f'fusion{action}_step{step}.private.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),'step':step})
            if step%32==0:write(folder/'FUSION_ACTIVE.json',{'action':action,'step':step,'counts':counts})
        # Saved optimizer restore checked without another optimization update.
        pack=torch.load(folder/f'fusion{action}_step512.private.pt',weights_only=False);other=Fusion().cuda();other.load_state_dict(pack['model']);oo=torch.optim.AdamW(other.parameters(),lr=1e-4,weight_decay=.01);oo.load_state_dict(pack['optimizer'])
        with torch.no_grad():assert torch.equal(m(z[:2,0],z[:2,1],p),other(z[:2,0],z[:2,1],p))
        assert all(int(v['step'])==512 for v in oo.state.values());del other,oo
        records.append({'action':action,'updates':512,'fit_rows':len(y),'micro_backwards':512*((len(y)+63)//64),'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'grad_clip_fraction':sum(x>1 for x in norms)/512,'strict_restore':True})
        m.eval().requires_grad_(False);models.append(m);del opt,z,target
    write(folder/'FUSION_RECORDS.json',records);return models

@torch.no_grad()
def scores(features,P,models):
    z=features.cuda();P=P.cuda();out=[20*z[:,0]@P.T]
    for a,m in enumerate(models,1):out.append(m(z[:,0],z[:,a],P))
    return torch.stack(out,1).detach()
