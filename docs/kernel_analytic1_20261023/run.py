"""True-arrival fixed-map moment accumulation, jointly solved seen-class decision anchors."""
import json,os,time,signal,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS,select_anchors,fixed_reference,feature_map,patient_weights,moments,system,solve
from data import read_rows,Images
from verify import mapping,stats,np_system

def write(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def save(p,x):
    q=p.with_suffix(p.suffix+'.tmp');torch.save(x,q);q.replace(p)
def cpu_record(r):return {k:({t:z.cpu() for t,z in v.items()} if isinstance(v,dict) and k in ['linear','kernel'] else v) for k,v in r.items()}

def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    p=Path(cfg['output']);p.mkdir(exist_ok=False)
    cost=dict(qualification_encodes=0,science_train_encodes=0,validation_encodes=0,pretrained_encoder_updates=0,formal_optimizer_updates=0,formal_solves=0,qualification_discarded_optimizer_updates=0,qualification_solves=0,mapping_rows={'linear':0,'kernel':0},moment_rows={'linear':0,'kernel':0},online_independent_mapping_rows={'linear':0,'kernel':0},online_independent_moment_rows={'linear':0,'kernel':0},encoder_seconds={},solve_seconds={},statistics_seconds=0.,qualification_seconds=None)
    def status(phase,**kw):
        cost['wall_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_reserved_peak_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(p/'COSTS.private.json',cost);write(p/'status.private.json',dict(state='RUNNING',phase=phase,at=time.time(),cost=cost,**kw));assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_reserved_peak_bytes']<cfg['gpu_reserved_budget_bytes']
    def interrupt(sig,frame):write(p/'INTERRUPTED.private.json',dict(at=time.time(),cost=cost));raise SystemExit(143)
    signal.signal(signal.SIGTERM,interrupt)
    tr,va=read_rows(cfg,'train'),read_rows(cfg,'val')
    for i,r in enumerate(tr):r['id']=i
    assert len(tr)==68058 and len(va)==300 and not {r['group'] for r in tr}&{r['group'] for r in va}
    import timm
    from safetensors.torch import load_file
    status('loading_encoder');enc=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);enc.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True);enc.eval().requires_grad_(False).cuda()
    def extract(rows,field):
        out=torch.empty(len(rows),1536);st=time.time();past=cost['encoder_seconds'].get(field,0.)
        for im,ix in DataLoader(Images(rows,cfg),batch_size=32,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=enc.forward_features(im.cuda())
            z=z.float();f=F.normalize(torch.cat([z[:,0],z[:,1:].mean(1)],1),dim=1);assert torch.isfinite(f).all();out[ix]=f.cpu();cost[field]+=len(ix);torch.cuda.synchronize();cost['encoder_seconds'][field]=past+time.time()-st;status(field,current_rows=int(ix[-1])+1,total_rows=len(rows))
        return out
    def build(features,rows,classes,ref,qualification=False):
        result=[];st=time.time()
        for c in classes:
            ix=[i for i,r in enumerate(rows) if r['label']==c];rr=[rows[i] for i in ix];x=features[ix];weights,patients=patient_weights(rr);rec=dict(label=c,images=len(rr),patients=patients);err=0.;probe={}
            for k in ['linear','kernel']:
                phi=feature_map(x.cuda(),ref,k);gram,mean=moments(phi,weights);refphi=mapping(x.numpy(),ref,k);ng,nm,n=stats(refphi,rr);err=max(err,float(abs(phi.cpu().numpy()-refphi).max()),float(abs(gram.cpu().numpy()-ng).max()),float(abs(mean.cpu().numpy()-nm).max()));assert n==patients and err<=cfg['score_absolute_tolerance']
                rec[k]=dict(gram=gram.detach().cpu(),mean=mean.detach().cpu());probe[k]=dict(features=x[:min(8,len(x))],mapping=phi[:min(8,len(x))].cpu())
                if not qualification:
                    cost['mapping_rows'][k]+=len(x);cost['moment_rows'][k]+=len(x);cost['online_independent_mapping_rows'][k]+=len(x);cost['online_independent_moment_rows'][k]+=len(x)
                del phi,gram,mean,refphi,ng,nm
            rec['online_independent']=dict(state='PASS',max_error=err,patient_weight_sum=float(weights.sum()),images=len(rr),patients=patients,at=time.time());rec['map_probe']=probe;result.append(rec)
        if not qualification:cost['statistics_seconds']+=time.time()-st
        return result
    # Qualification covers the actual 2049-dimensional kernel solve, not only a tiny surrogate.
    qr=[];used=set()
    for c in cfg['order'][:10]:
        n=0
        for r in tr:
            if r['label']==c and r['group'] not in used:qr.append(r);used.add(r['group']);n+=1
            if n==2:break
        assert n==2
    qx=extract(qr,'qualification_encodes');qs=time.time();anchors=qx[:16].repeat((cfg['anchors']+15)//16,1)[:cfg['anchors']].cuda();ref=fixed_reference(anchors,cfg);receipts={};qrecords=[]
    # Repeated qualification landmarks exercise full formal dimensions only; discarded before science.
    for stage in range(2):
        ix=list(range(16)) if stage==0 else list(range(16,20));new=cfg['order'][:8] if stage==0 else cfg['order'][8:10];qrecords+=build(qx[ix],[qr[i] for i in ix],new,ref,True)
        seen=cfg['order'][:8] if stage==0 else cfg['order'][:10]
        for a in ARMS:
            gpu=[{**r,**{k:{z:t.cuda() for z,t in r[k].items()} for k in ['linear','kernel']}} for r in qrecords];head,res=solve(gpu,a,cfg['ridge']);cost['qualification_solves']+=1
            if stage==1:
                kind='linear' if a=='LINEAR_BALANCED' else 'kernel';xx=qx.repeat(4,1)[:64];phi=feature_map(xx.cuda(),ref,kind);labels=torch.arange(64,device='cuda')%20//2
                # Discarded gradient/clip/optimizer engineering path is additional; formal fit is analytic.
                param=torch.nn.Parameter(head.clone());before=param.detach().clone();opt=torch.optim.SGD([param],lr=.01);opt.zero_grad();loss=F.mse_loss(phi@param,F.one_hot(labels,10).double());loss.backward();assert torch.isfinite(param.grad).all();torch.nn.utils.clip_grad_norm_([param],5.);opt.step();delta=float((param-before).abs().max());assert delta>0;cost['qualification_discarded_optimizer_updates']+=1
                save(p/f'qualification_{a}.private.pt',dict(head=head.cpu(),ref=ref));saved=torch.load(p/f'qualification_{a}.private.pt',map_location='cpu',weights_only=False)
                mat,rhs=np_system(qrecords,a,cfg['ridge']);nh=np.linalg.solve(mat,rhs);ns=mapping(xx.numpy(),saved['ref'],kind)@nh;score=(phi@head).cpu().numpy();err=float(abs(ns-score).max());assert err<=cfg['score_absolute_tolerance']
                receipts[a]=dict(score_error=err,normal_equation_residual=res,full_dimension=list(head.shape),actual_discarded_optimizer_delta=delta,engineering_forward_backward_rows=64,analytic_qualification_solves=2,qualification_mapping_and_moments_rows=20,independent_numpy_rows=64)
            del gpu,head
    write(p/'QUALIFICATION.json',dict(state='PASS',real_legal_base_encodes=20,actual_GPUFP16_encoder=True,actual_GPUFP64_fixed_map_moments_and_solve=True,formal_kernel_dimension=cfg['anchors']+1,base_and_8plus2_arrival=True,actual_GPU64_forward_backward_clip_discarded_SGD=True,save_load=True,arms=receipts,discarded_optimizer_updates=3,qualification_solves=6,formal_optimizer_updates=0,repeated_landmarks_only_discarded_qualification=True,post_encoding_seconds=time.time()-qs,validation_encodes=0))
    cost['qualification_seconds']=time.time()-qs;del qx,anchors,qrecords,ref,xx,phi,labels,param,before,opt,loss,mat,rhs,nh,ns,score,saved;torch.cuda.empty_cache();status('qualified_before_bulk')
    records=[];seals=[];ref=None
    for task in range(6):
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])];old=sum(cfg['task_sizes'][:task]);new=seen[old:];rows=[r for r in tr if r['label'] in new];features=extract(rows,'science_train_encodes')
        if task==0:
            ix=select_anchors(rows,cfg['anchors'],cfg['seed']);ref=fixed_reference(features[ix].cuda(),cfg);ref.update(anchor_paths=[rows[i]['relative_path'] for i in ix],anchor_groups=[rows[i]['group'] for i in ix],anchor_ids=[rows[i]['id'] for i in ix]);save(p/'reference.private.pt',ref);write(p/'REFERENCE_SEAL.json',dict(at=time.time(),landmarks=len(ix),patients=len(set(ref['anchor_groups'])),sigma=float(ref['sigma']),base_only=True,validation_encodes=0,science_train_encodes=cost['science_train_encodes']))
        newrecords=build(features,rows,new,ref);records+=newrecords
        for rec in newrecords:save(p/f"class_{rec['label']}.private.pt",rec)
        status('actual_arrival_statistics',task=task)
        gpu=[{**r,**{k:{z:t.cuda() for z,t in r[k].items()} for k in ['linear','kernel']}} for r in records];heads={};resids={};probe=dict(features=features[:64],scores={})
        for a in ARMS:
            st=time.time();head,res=solve(gpu,a,cfg['ridge']);cost['formal_solves']+=1;cost['solve_seconds'][a]=cost['solve_seconds'].get(a,0.)+time.time()-st;heads[a]=head.cpu();resids[a]=res;kind='linear' if a=='LINEAR_BALANCED' else 'kernel';probe['scores'][a]=(feature_map(probe['features'].cuda(),ref,kind)@head).cpu()
        save(p/f'stage_{task}.private.pt',dict(seen=seen,heads=heads,probe=probe,residuals=resids))
        seals.append(dict(task=task,at=time.time(),seen=seen,validation_encodes=0,science_train_encodes=cost['science_train_encodes'],formal_solves=cost['formal_solves'],normal_equation_residuals=resids));write(p/'STAGE_SEALS.json',seals);del features,gpu,head,heads,probe,newrecords;torch.cuda.empty_cache();status('stage_sealed',task=task)
    write(p/'FIT_SEAL.json',dict(at=time.time(),models=18,validation_encodes=0,science_train_encodes=cost['science_train_encodes']));write(p/'QUERY_OPEN.private.json',dict(at=time.time()));features=extract(va,'validation_encodes');save(p/'query.private.pt',dict(features=features,labels=torch.tensor([r['label'] for r in va])))
    phi={k:feature_map(features.cuda(),ref,k) for k in ['linear','kernel']}
    for task in range(6):
        pack=torch.load(p/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);mask=torch.tensor([r['label'] in pack['seen'] for r in va],device='cuda')
        for a in ARMS:
            kind='linear' if a=='LINEAR_BALANCED' else 'kernel';score=phi[kind][mask]@pack['heads'][a].cuda();save(p/f'scores_{task}_{a}.private.pt',score.cpu())
    write(p/'COSTS.json',cost);status('COMPLETE_PENDING_VERIFY')
if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception as e:
        p=Path(cfg['output']);p.mkdir(exist_ok=True,parents=True);write(p/'FAILURE.json',dict(at=time.time(),type=type(e).__name__,message=str(e),trace=traceback.format_exc()));raise
