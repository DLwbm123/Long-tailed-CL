"""Fresh R67-recipe teachers from generic frozen prefixes, new stream only."""
import math,time,copy,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import guard,write,save
from teacher_vision import NativeTail,adapt_native_tail,delta_state,load_delta
from teacher_sft_decision import query_risk
from folds import assign_folds
from encoder import make_base

COUNTS={'prefix_image_rows':0,'generic_tail_forward_rows':0,'teacher_optimizer':0,'teacher_micro_backwards':0,'training_query_rows':0,'training_support_rows':0,'observation_tail_rows':0,'final_teacher_feature_rows':0,'teacher_prototype_rows':0,'VAL_TEST':0}

def bundle_from_visual(v):
    state={}
    for k,b in enumerate(v.trunk.blocks[10:]):state.update({f'blocks.{k}.{n}':p.detach().cpu().clone() for n,p in b.state_dict().items()})
    state.update({'norm.'+n:p.detach().cpu().clone() for n,p in v.trunk.norm.state_dict().items()})
    assert list(v.head._modules)==['drop','proj'] and v.head.drop.p==0 and isinstance(v.head.proj,torch.nn.Linear) and v.head.proj.bias is None,'UNSUPPORTED_VISUAL_HEAD'
    state['proj.weight']=v.head.proj.weight.detach().cpu().clone()
    return {'state':state,'meta':{'norm_eps':v.trunk.norm.eps,'block_norm_eps':[(b.norm1.eps,b.norm2.eps) for b in v.trunk.blocks[10:]],'global_pool':v.trunk.global_pool,'fc_norm_identity':isinstance(v.trunk.fc_norm,torch.nn.Identity)}}

class Prefixes:
    def __init__(self,c,rows,y):
        self.c=c;self.rows=rows;self.y=y;self.root=Path(c['output']);self.cache={};self.position={}
        base,self.tf,_=make_base(c,'cuda');self.visual=base.visual;self.bundle=bundle_from_visual(self.visual);del base
        self.folder=self.root/'generic_prefix';self.folder.mkdir(exist_ok=True)
    def pixels(self,ids):
        batch=[]
        for i in ids:
            with Image.open(Path(self.c['image_root'])/self.rows[i]['relative_path']) as im:batch.append(self.tf(im.convert('RGB')))
        return torch.stack(batch).cuda()
    @torch.no_grad()
    def prefix(self,ids):
        v=self.visual;t=v.trunk;x=t.patch_embed(self.pixels(ids));x=t._pos_embed(x);x=t.patch_drop(x);x=t.norm_pre(x)
        for block in t.blocks[:10]:x=block(x)
        tokens=x
        for block in t.blocks[10:]:x=block(x)
        raw=F.normalize(v.head(t.forward_head(t.norm(x))),dim=-1)
        return tokens,raw
    @torch.no_grad()
    def arrive(self,classes):
        for cl in classes:
            assert cl not in self.cache,'NO_PREFIX_OVERWRITE'
            ids=[i for i,v in enumerate(self.y.tolist()) if v==cl];tokens=[];base=[]
            for lo in range(0,len(ids),16):
                guard(self.c);ix=ids[lo:lo+16];z,b=self.prefix(ix);tokens.append(z.cpu());base.append(b.cpu());COUNTS['prefix_image_rows']+=len(ix);COUNTS['generic_tail_forward_rows']+=len(ix)
            pack={'ids':ids,'tokens':torch.cat(tokens),'base':torch.cat(base)};assert pack['tokens'].dtype==torch.float32 and pack['tokens'].shape==(len(ids),197,768)
            save(self.folder/f'p{cl}.private.pt',pack);self.cache[cl]=pack
            for j,i in enumerate(ids):self.position[i]=(cl,j)
    def batch(self,ids):
        z=[];b=[]
        for i in ids:
            cl,j=self.position[i];z.append(self.cache[cl]['tokens'][j]);b.append(self.cache[cl]['base'][j])
        return torch.stack(z).cuda(),torch.stack(b).cuda(),self.y[ids].cuda()

def new_tail(bundle,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    with torch.random.fork_rng(devices=[0]):m=NativeTail(bundle).cuda()
    params=adapt_native_tail(m);m.eval();return m,params

@torch.no_grad()
def encode(m,store,ids,C,counter_key):
    sums=torch.zeros(C,512,device='cuda',dtype=torch.double);den=torch.zeros(C,device='cuda',dtype=torch.long)
    for lo in range(0,len(ids),64):
        guard(store.c);ix=ids[lo:lo+64];tokens,_,y=store.batch(ix);z=m(tokens);assert torch.isfinite(z).all()
        sums.index_add_(0,y,z.double());den+=torch.bincount(y,minlength=C);COUNTS[counter_key]+=len(ix)
    assert (den>0).all();return F.normalize((sums/den[:,None]).float(),dim=1).detach(),den

def run(c,rows,y,store):
    r=Path(c['output']);heads=r/'strong_heads';heads.mkdir(exist_ok=True);m,params=new_tail(store.bundle,c['seed']);records=[];oldfold={}
    frozen={k:p.detach().cpu().clone() for k,p in m.named_parameters() if not p.requires_grad}
    for s,classes in enumerate(c['stages']):
        wall=time.time();cpu=time.process_time();guard(c);write(r/'STATUS.json',{'phase':'TEACHER_ARRIVAL','stage':s,'counts':COUNTS});store.arrive(classes)
        prefix_wall=time.time()-wall;prefix_cpu=time.process_time()-cpu;fitwall=time.time();fitcpu=time.process_time()
        C=sum(map(len,c['stages'][:s+1]));ids=sorted(store.position);N=len(ids);nc=torch.tensor(c['train_counts'][:C],device='cuda')
        assert set(y[ids].tolist())==set(range(C)) and torch.equal(torch.bincount(y[ids].cuda(),minlength=C),nc)
        folds=assign_folds(y[ids].tolist(),[rows[i]['identity_component'] for i in ids]);mapping=dict(zip(ids,folds));assert all(mapping[i]==v for i,v in oldfold.items());oldfold=mapping
        ii=[[i for i in ids if mapping[i]==b] for b in [0,1]];assert all((torch.bincount(y[v],minlength=C)>0).all() for v in ii)
        opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10);pg=torch.Generator(device='cuda').manual_seed(c['seed']+10000*s);curves=[];steps=0;micros=0;queries=0;supports=0
        def observe(epoch,obj):
            cc=torch.zeros(C,device='cuda',dtype=torch.long);den=cc.clone()
            with torch.no_grad():
                for b in [0,1]:
                    proto,_=encode(m,store,ii[1-b],C,'observation_tail_rows')
                    for lo in range(0,len(ii[b]),64):
                        ix=ii[b][lo:lo+64];tokens,_,yy=store.batch(ix);pr=(20*m(tokens)@proto.T).argmax(1);cc+=torch.bincount(yy[pr==yy],minlength=C);den+=torch.bincount(yy,minlength=C);COUNTS['observation_tail_rows']+=len(ix)
            assert torch.equal(den,nc);rec=cc.double()/den;tail=nc<250
            curves.append({'epoch':epoch,'objective':obj,'optimizer_steps':steps,'micro_backwards':micros,'TRAIN_counts':den.tolist(),'TRAIN_crossfit_correct':cc.tolist(),'TRAIN_macro':float(rec.mean()*100),'TRAIN_tail':float(rec[tail].mean()*100) if tail.any() else None,'lr':opt.param_groups[0]['lr'],'independent_validation':False})
            write(r/f'TEACHER_CURVES_s{s}.json',curves)
        observe(0,None)
        for epoch in range(10):
            total=0.
            for b in [0,1]:
                opt.zero_grad(set_to_none=True);proto,sc=encode(m,store,ii[1-b],C,'training_support_rows');assert not proto.requires_grad
                supports+=len(ii[1-b]);shuffled=[ii[b][j] for j in torch.randperm(len(ii[b]),device='cuda',generator=pg).cpu().tolist()]
                for lo in range(0,len(shuffled),256):
                    guard(c);batch=shuffled[lo:lo+256];opt.zero_grad(set_to_none=True)
                    for ml in range(0,len(batch),64):
                        ix=batch[ml:ml+64];tokens,base,yy=store.batch(ix);z=m(tokens);risk,_=query_risk(z,yy,proto,nc,N)
                        loss=(risk+.01*(z-base).square().sum(1)).sum()/len(batch);assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())*len(batch);micros+=1;queries+=len(ix);COUNTS['teacher_micro_backwards']+=1;COUNTS['training_query_rows']+=len(ix)
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);opt.step();steps+=1;COUNTS['teacher_optimizer']+=1
                    write(r/'ACTIVE_TEACHER.private.json',{'stage':s,'epoch':epoch,'query_fold':b,'optimizer_steps_stage':steps,'counts':COUNTS})
            sched.step();observe(epoch+1,total/N)
        assert steps==10*sum(math.ceil(len(v)/256) for v in ii) and micros==10*sum(math.ceil(len(v)/64) for v in ii) and queries==supports==10*N
        assert all(torch.equal(p.detach().cpu(),frozen[k]) for k,p in m.named_parameters() if not p.requires_grad)
        proto,den=encode(m,store,ids,C,'teacher_prototype_rows')
        save(heads/f'head{s}.private.pt',{'stage':s,'seen':C,'delta':delta_state(m),'prototypes':proto.cpu(),'train_counts':den.cpu(),'seed':c['seed'],'new_stream':c['stream_name'],'future_classes_seen':False})
        fit_wall=time.time()-fitwall;fit_cpu=time.process_time()-fitcpu;featurewall=time.time();featurecpu=time.process_time()
        z=torch.zeros(len(rows),512);filled=torch.zeros(len(rows),dtype=torch.bool)
        with torch.no_grad():
            for lo in range(0,N,64):
                guard(c);ix=ids[lo:lo+64];tx,_,_=store.batch(ix);z[ix]=m(tx).cpu();filled[ix]=True;COUNTS['final_teacher_feature_rows']+=len(ix)
        test,_,_=store.batch(ids[:2]);other,_=new_tail(store.bundle,c['seed']);load_delta(other,delta_state(m))
        with torch.no_grad():assert torch.equal(m(test),other(test))
        del other
        save(r/f'teacher_s{s}.private.pt',{'stage':s,'ids':ids,'z':z,'labels':y,'filled':filled,'raw_counts':den.cpu(),'new_stream':c['stream_name'],'no_legacy_HK_features':True})
        records.append({'stage':s,'TRAIN_rows':N,'seen_classes':C,'fold_rows':[len(v) for v in ii],'optimizer_steps':steps,'micro_backwards':micros,'epochs':10,'query_rows':queries,'support_rows':supports,'strict_restore_exact':True,'base_tail_frozen_exact':True,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'generic_prefix_CPU_seconds':prefix_cpu,'generic_prefix_wall_seconds':prefix_wall,'teacher_fit_observation_CPU_seconds':fit_cpu,'teacher_fit_observation_wall_seconds':fit_wall,'new_feature_and_restore_CPU_seconds':time.process_time()-featurecpu,'new_feature_and_restore_wall_seconds':time.time()-featurewall,'VAL_TEST':0})
        write(r/'TEACHER_RECORDS.json',records);del opt,sched,z
    write(r/'TEACHERS_SEALED.json',{'stages':len(records),'optimizer_steps':COUNTS['teacher_optimizer'],'records':records,'counts':COUNTS,'no_legacy_HK_supervision':True,'VAL_TEST':0})
    del m,params,frozen
