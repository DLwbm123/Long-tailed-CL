"""Outer-fold-only LoRA lineage; deterministic inner singleton fallback."""
import time,math
from pathlib import Path
import torch
import torch.nn.functional as F
from teacher_base import Prefixes,new_tail,encode,COUNTS
from teacher_vision import delta_state,load_delta
from teacher_sft_decision import query_risk
from common import write,save,guard

class Store(Prefixes):
    @torch.no_grad()
    def arrive(self,classes):
        for cl in classes:
            assert cl not in self.cache
            pack=torch.load(Path(self.c['reuse_root'])/'generic_prefix'/f'p{cl}.private.pt',map_location='cpu',weights_only=False)
            assert pack['ids']==[i for i,v in enumerate(self.y.tolist()) if v==cl]
            assert pack['tokens'].shape==(len(pack['ids']),197,768) and pack['tokens'].dtype==torch.float32
            self.cache[cl]=pack
            for j,i in enumerate(pack['ids']):self.position[i]=(cl,j)

def inner(ids,rows,y):
    groups={};mapping={}
    for i in ids:groups.setdefault((int(y[i]),rows[i]['identity_component']),[]).append(i)
    for cl in sorted(set(int(y[i]) for i in ids)):
        sizes=[0,0]
        for (c,g),ix in sorted([(key,v) for key,v in groups.items() if key[0]==cl],key=lambda z:(-len(z[1]),z[0][1])):
            fold=0 if sizes[0]<=sizes[1] else 1;sizes[fold]+=len(ix)
            for i in ix:mapping[i]=fold
    return mapping

def stage(c,rows,y,store,model,params,fit_ids,s,folder,counts):
    folder.mkdir(exist_ok=True);C=sum(map(len,c['stages'][:s+1]));ids=sorted(i for i in fit_ids if y[i]<C);mapping=inner(ids,rows,y);ii=[[i for i in ids if mapping[i]==f] for f in [0,1]]
    nc=torch.bincount(y[ids],minlength=C).cuda();assert (nc>0).all();N=len(ids);fallback=[]
    def support(f):
        v=ii[1-f].copy();missing=[cl for cl in range(C) if not any(int(y[i])==cl for i in v)]
        for cl in missing:v.extend(i for i in ids if int(y[i])==cl)
        assert set(v)<=set(ids);return sorted(v),missing
    ss=[support(f) for f in [0,1]]
    for f,(_,missing) in enumerate(ss):
        for cl in missing:fallback.append({'stage':s,'query_inner_fold':f,'class':cl,'fit_class_count':int(nc[cl]),'query_rows':sum(int(y[i])==cl for i in ii[f])})
    write(folder/'INNER_SUPPORT_FALLBACK.json',fallback)
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10)
    pg=torch.Generator(device='cuda').manual_seed(c['seed']+10000*s);curves=[];steps=0;wall=time.time();cpu=time.process_time()
    frozen={k:p.detach().cpu().clone() for k,p in model.named_parameters() if not p.requires_grad}
    @torch.no_grad()
    def observe(epoch):
        cc=torch.zeros(C,device='cuda',dtype=torch.long)
        for f in [0,1]:
            proto,_=encode(model,store,ss[f][0],C,'observation_tail_rows')
            for lo in range(0,len(ii[f]),64):
                ix=ii[f][lo:lo+64];tx,_,yy=store.batch(ix);pred=(20*model(tx)@proto.T).argmax(1);cc+=torch.bincount(yy[pred==yy],minlength=C);counts['fold_observation_rows']+=len(ix)
        curves.append({'epoch':epoch,'correct':cc.tolist(),'counts':nc.tolist(),'optimizer_steps':steps,'feedback_rows_used':0});write(folder/'TEACHER_CURVES.json',curves)
    observe(0)
    for epoch in range(10):
        for f in [0,1]:
            proto,_=encode(model,store,ss[f][0],C,'training_support_rows');counts['fold_support_rows']+=len(ss[f][0]);assert not proto.requires_grad
            shuffled=[ii[f][j] for j in torch.randperm(len(ii[f]),device='cuda',generator=pg).cpu().tolist()]
            for lo in range(0,len(shuffled),256):
                guard(c);ix=shuffled[lo:lo+256];opt.zero_grad(set_to_none=True)
                for ml in range(0,len(ix),64):
                    jj=ix[ml:ml+64];tok,base,yy=store.batch(jj);z=model(tok);risk,_=query_risk(z,yy,proto,nc,N);loss=(risk+.01*(z-base).square().sum(-1)).sum()/len(ix)
                    assert torch.isfinite(loss);loss.backward();counts['fold_query_rows']+=len(jj);counts['fold_micro_backwards']+=1
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);opt.step();steps+=1;counts['fold_teacher_optimizer']+=1
                write(Path(c['output'])/'ACTIVE_COUNTS.private.json',{'phase':'FOLD_TEACHER','bank':folder.parent.name,'stage':s,'epoch':epoch,'counts':counts})
        sched.step();observe(epoch+1)
    assert steps==10*sum(math.ceil(len(v)/256) for v in ii)
    assert all(torch.equal(p.cpu(),frozen[k]) for k,p in model.named_parameters() if not p.requires_grad)
    P,den=encode(model,store,ids,C,'teacher_prototype_rows');counts['fold_prototype_rows']+=len(ids)
    head={'stage':s,'seen':C,'delta':delta_state(model),'prototypes':P.cpu(),'train_counts':den.cpu(),'fit_ids':ids,'new_stream':c['stream_name'],'outer_fold_only':True}
    save(folder/'teacher.private.pt',head);save(folder/'teacher_optimizer.private.pt',{'optimizer':opt.state_dict(),'scheduler':sched.state_dict()})
    other,_=new_tail(store.bundle,c['seed']);load_delta(other,head['delta']);tok,_,_=store.batch(ids[:2])
    with torch.no_grad():assert torch.equal(model(tok),other(tok))
    del other
    write(folder/'TEACHER_RECORD.json',{'stage':s,'fit_rows':N,'fit_counts':nc.tolist(),'inner_fold_rows':list(map(len,ii)),'fallback':fallback,'updates':steps,'micro_backwards':10*sum(math.ceil(len(v)/64) for v in ii),'strict_restore':True,'frozen_base_exact':True,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'outer_feedback_used':0})
    return head
