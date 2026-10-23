"""Single sealed six-stage fit; mandatory admission before bulk and no val before all fits."""
import copy,json,os,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import Upper,Growing
from data import read_rows,Images
from foundation import build_backbone,prefix,save,write
from training import ARMS,state,optimizer,counter,update,forward,cached,selected,weights,affine,restore
from checks import math_check,qualify


def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=Path(cfg['output']);p.mkdir(exist_ok=False)
    cost=dict(qualification_encodes=0,science_train_encodes=0,validation_encodes=0,arms={a:counter() for a in ARMS[1:]},old_cache_upper_rows={a:0 for a in ARMS[1:]},native_upper_rows=0,native_solves=0,qualification_discarded_updates=0)
    def status(phase,**more):
        cost['elapsed_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(p/'COSTS.json',cost);write(p/'status.private.json',dict(phase=phase,updated=time.time(),**more))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    def stop(sig,frame):status('INTERRUPTED');raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);status('mathematical_qualification');math_check(cfg,p)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val');assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val}
    for i,r in enumerate(train):r['id']=i
    b=build_backbone(cfg,'cuda');original=state(b);template=Upper(b).cpu().float().requires_grad_(False);save(p/'initial_upper.private.pt',state(template))
    def extract(rows,field,proof=False):
        tokens=torch.empty((len(rows),197,768),dtype=torch.float16);wanted=set(selected(rows,2,cfg['seed'])) if proof else set();proofs={}
        embeddings=torch.empty_like(tokens) if field=='validation_encodes' else None
        for images,ix in DataLoader(Images(rows,cfg),batch_size=cfg['encoder_batch'],num_workers=0,shuffle=False):
            with torch.no_grad():
                with torch.autocast('cuda',dtype=torch.float16):emb=prefix(b,images.cuda()).half()
                lower=b.blocks[:10](emb.float());raw=lower.half().cpu()
            assert torch.isfinite(raw).all() and raw.shape==(len(ix),197,768);tokens[ix]=raw
            if embeddings is not None:embeddings[ix]=emb.cpu()
            for j,i in enumerate(ix.tolist()):
                if i in wanted:proofs[i]=dict(embedding=emb[j].cpu(),lower=lower[j].cpu(),id=rows[i]['id'],label=rows[i]['label'])
            cost[field]+=len(ix);status(field,encoded=int(ix[-1])+1,total=len(rows))
        return tokens,embeddings,proofs
    def descriptors(tokens):
        u=copy.deepcopy(template).cuda();z=forward(u,tokens,cfg['microbatch']).double();del u;cost['native_upper_rows']+=len(tokens);return z
    def means(rows,z,classes):
        out=[]
        for c in classes:
            ii=[i for i,r in enumerate(rows) if r['label']==c];w,_=weights([rows[i] for i in ii]);out.append((torch.from_numpy(w)[:,None]*z[ii]).sum(0))
        return torch.stack(out).float()
    qr=[]
    for c in cfg['order'][:10]:
        rr=[r for r in train if r['label']==c];qr.extend(rr[i] for i in selected(rr,2,cfg['seed']))
    qt,_,qp=extract(qr,'qualification_encodes',True);qz=descriptors(qt);qm=means(qr,qz,cfg['order'][:10]);status('real_GPU_qualification');qualification=qualify(template,qt,qm,cfg,p)
    cost['qualification_discarded_updates']=qualification['discarded_updates']
    # Full12-chain independent real qualification, on saved actual embedding/lower interface.
    from verify import validate_chain
    qualification['full_chain']=validate_chain(b,template,qp.values(),cfg)
    forecast=cfg['feasibility_factor']*(cfg['base_updates']+5*cfg['later_updates'])*qualification['paired_step_seconds']+cfg['feasibility_reserve_seconds'];remaining=cfg['hard_deadline_epoch']-time.time()
    qualification.update(forecast_seconds=forecast,original_clock_remaining_seconds=remaining,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved());write(p/'QUALIFICATION.json',qualification)
    assert forecast<remaining-60 and torch.cuda.max_memory_reserved()<cfg['gpu_reserved_budget_bytes']
    del qt,qp,qz,qm,qr;torch.cuda.empty_cache()
    models={a:Growing(a=='FULL_SCORE_FROZEN') for a in ARMS[1:]};memory=torch.empty((0,197,768),dtype=torch.float16);memory_rows=[];gram=torch.zeros(1537,1537,dtype=torch.float64);mus=[];seals=[]
    for task in range(6):
        seen=cfg['order'][:10+2*task];new=seen[:10] if task==0 else seen[-2:];rows=[r for r in train if r['label'] in new]
        tokens,_,proofs=extract(rows,'science_train_encodes',True);z=descriptors(tokens);mm=means(rows,z,new);arrived=torch.zeros_like(gram)
        for c in new:
            ii=[i for i,r in enumerate(rows) if r['label']==c];rr=[rows[i] for i in ii];w,n=weights(rr);phi=affine(z[ii]);gg=phi.T@(torch.from_numpy(w)[:,None]*phi);mu=(torch.from_numpy(w)[:,None]*phi).sum(0)
            err=float(abs(phi.numpy().T@(w[:,None]*phi.numpy())-gg.numpy()).max());assert err<1e-9
            gram+=gg;arrived+=gg;mus.append(mu)
            pick=selected(rr,cfg['memory_per_class'],cfg['seed']);save(p/f'memory_{c}.private.pt',dict(tokens=tokens[[ii[i] for i in pick]],ids=[rr[i]['id'] for i in pick]))
            save(p/f'proof_{c}.private.pt',dict(label=c,images=len(rr),patients=n,gram_error=err,proof=[v for v in proofs.values() if v['label']==c]))
        mat=gram/len(seen)+cfg['ridge']*torch.eye(1537,dtype=torch.float64);rhs=torch.stack(mus,1)/len(seen);head=torch.linalg.solve(mat,rhs);assert float(abs(mat@head-rhs).max())<1e-9;cost['native_solves']+=1
        source=memory_rows+rows;source_tokens=torch.cat([memory,tokens]) if len(memory) else tokens
        caches={}
        for a,m in models.items():
            m.append(template,mm,new);m.cuda();caches[a]=cached(m,source_tokens,cfg['microbatch']);cost['old_cache_upper_rows'][a]+=len(source_tokens)*task
        opts={a:optimizer(m,cfg) for a,m in models.items()};pools={}
        for i,r in enumerate(source):pools.setdefault(r['label'],{}).setdefault(r['group'],[]).append(i)
        rng=np.random.default_rng(cfg['seed']+1000+task);steps=cfg['base_updates'] if task==0 else cfg['later_updates'];draws=[]
        for step in range(1,steps+1):
            ix=[]
            for c in rng.choice(seen,8,replace=False):
                gs=list(pools[c])
                for _ in range(8):
                    vals=pools[c][gs[int(rng.integers(len(gs)))]];ix.append(vals[int(rng.integers(len(vals)))])
            x=source_tokens[ix];y=torch.tensor([seen.index(source[i]['label']) for i in ix]);draws.append([source[i]['id'] for i in ix])
            if step==steps:before={a:dict(current=state(m.experts[-1]),heads=[e.head.detach().cpu().clone() for e in list(m.experts)[:-1]]) for a,m in models.items()}
            last={a:update(m,opts[a],x,y,cfg,cost['arms'][a],[v[ix] for v in caches[a]]) for a,m in models.items()}
            if step==1 or step%100==0 or step==steps:status('training',task=task,step=step,steps=steps,losses=last)
        heads={a:[e.head.detach().cpu().clone() for e in m.experts] for a,m in models.items()}
        for a,m in models.items():
            for j,e in enumerate(m.experts):
                if j<task:
                    birth=torch.load(p/f'expert_{a}_{j}.private.pt',map_location='cpu',weights_only=False)
                    assert all(torch.equal(v,e.upper.state_dict()[k].cpu()) for k,v in birth.items())
                    if a=='FULL_SCORE_FROZEN':assert torch.equal(heads[a][j],before[a]['heads'][j])
                else:save(p/f'expert_{a}_{j}.private.pt',state(e.upper))
        pick=selected(rows,cfg['memory_per_class'],cfg['seed']);memory=torch.cat([memory,tokens[pick]]);memory_rows.extend(rows[i] for i in pick)
        save(p/f'stage_{task}.private.pt',dict(seen=seen,heads=heads,native_head=head,native_mat=mat,native_rhs=rhs,arrived_gram=arrived,memory_ids=[r['id'] for r in memory_rows],draws=torch.tensor(draws,dtype=torch.int32),probe=dict(tokens=x,labels=y,before=before,losses={a:last[a]['CE'] for a in ARMS[1:]})))
        seals.append(dict(task=task,at=time.time(),models=3,memory_images=len(memory),updates={a:cost['arms'][a]['updates'] for a in ARMS[1:]}));write(p/'STAGE_SEALS.json',seals)
        del source_tokens,tokens,z,proofs,rows,source,caches,opts,pools,x,y,before,draws;torch.cuda.empty_cache()
    assert cost['science_train_encodes']==68058 and all(torch.equal(v,b.state_dict()[k].cpu()) for k,v in original.items())
    write(p/'FIT_SEAL.json',dict(models=18,at=time.time(),backbone_frozen_exact=True));write(p/'QUERY_OPEN.private.json',dict(at=time.time()))
    query,emb,_=extract(val,'validation_encodes');save(p/'query.private.pt',dict(tokens=query,embedding=emb,labels=torch.tensor([r['label'] for r in val])));del emb,models;torch.cuda.empty_cache()
    labels=torch.tensor([r['label'] for r in val]);groups=[cfg['order'][:10]]+[cfg['order'][10+2*j:12+2*j] for j in range(5)]
    for task in range(6):
        pack=torch.load(p/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);qt=query[torch.isin(labels,torch.tensor(pack['seen']))];u=copy.deepcopy(template).double();nz=forward(u,qt,cfg['microbatch']);del u;scores={'NATIVE_RIDGE':affine(nz)@pack['native_head']}
        for a in ARMS[1:]:
            states=[]
            for j,h in enumerate(pack['heads'][a]):
                upper=torch.load(p/f'expert_{a}_{j}.private.pt',map_location='cpu',weights_only=False);states.append(dict(head=h,**{'upper.'+k:v for k,v in upper.items()}))
            m=restore(template,groups[:task+1],states,a=='FULL_SCORE_FROZEN').double();scores[a]=forward(m,qt,cfg['microbatch']);del m,states
        save(p/f'scores_{task}.private.pt',scores);status('scored',task=task)
    status('COMPLETE_PENDING_VERIFY')

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        p=Path(cfg['output']);p.mkdir(exist_ok=True);write(p/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None));raise
