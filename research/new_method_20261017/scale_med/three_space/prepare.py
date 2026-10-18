"""Privileged frozen-training audit; tests and previous predictions are denied."""
import json,os,sys,time,resource,traceback
from pathlib import Path
import torch
import numpy as np
from core_depth import mapped,stats
from verify import verify,selfcheck

def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def main(cfg):
    start=time.time();torch.set_num_threads(4);root=Path(cfg['output']);old=Path(cfg['reference_run']).resolve()
    bc=json.loads(Path(cfg['benchmark_config']).read_text());src=json.loads(Path(bc['transfer_config']).read_text());testroot=Path(bc['output']).resolve()
    allowed={old/'transforms.private.pt',old/'2.heads.private.pt',old/'2.statistics.private.pt',old/'MODEL_SEAL.json'}|{old/'inputs'/(str(s)+'.private.pt') for s in range(3)}
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();assert not p.is_relative_to(testroot),'Test access forbidden'
            if p.is_relative_to(old):assert p in allowed,'Only frozen training inputs and states allowed'
    sys.addaudithook(guard);selfcheck();assert json.loads((old/'MODEL_SEAL.json').read_text())['source_commit']=='19bf801'
    trans=torch.load(old/'transforms.private.pt',weights_only=False);heads=torch.load(old/'2.heads.private.pt',weights_only=False);saved=torch.load(old/'2.statistics.private.pt',weights_only=False)
    import timm
    from safetensors.torch import load_file
    from run import Images
    from torch.utils.data import DataLoader
    model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);model.load_state_dict({k:v for k,v in load_file(src['weight']).items() if not k.startswith('head.')},strict=True);model.eval().requires_grad_(False).cuda()
    blocks={a:[] for a in ['G','C']};ys=[];gs=[];statcheck={};acc={};stages={};parity=False
    for stage in range(3):
        obj=torch.load(old/'inputs'/(str(stage)+'.private.pt'),weights_only=False);h=[]
        for x,idx in DataLoader(Images(obj['rows'],dict(images=bc['datasets']['ISIC']['development_images'])),batch_size=64,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                z=model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x.cuda()))))
                for b in model.blocks[:6]:z=b(z)
                v=z[:,0].float()
                if not parity:
                    got=[];hook=model.blocks[5].register_forward_hook(lambda m,a,o:got.append(o[:,0].float()));model.forward_features(x.cuda());hook.remove()
                    error=float((v-got[0]).abs().max());assert error<1e-5;write(root/'FEATURE_PARITY.json',dict(block6_hook_error=error,samples=len(x)));parity=True
            h.append(v.cpu().double());write(root/'status.json',dict(state='PREPARING',stage=stage,processed=int(idx[-1])+1,total=len(obj['rows'])))
        h=torch.cat(h);tr=~obj['valid'];y=obj['y'][tr];groups=[obj['groups'][i] for i in torch.where(tr)[0].tolist()]
        assert not {obj['groups'][i] for i in torch.where(tr)[0]}&{obj['groups'][i] for i in torch.where(~tr)[0]}
        for arm in ['G','C']:
            X=mapped(obj['g'],h,trans[arm])[tr];blocks[arm].append(X);ab=stats(X,y,groups,6)
            acc[arm]=ab if arm not in acc else (acc[arm][0]+ab[0],acc[arm][1]+ab[1])
        ys.append(y);gs+=groups;stages[str(stage)]=dict(training_images=len(y),excluded_holdout_images=int((~tr).sum()))
        del obj,h,X
    y=torch.cat(ys).numpy();data={a:torch.cat(v).numpy() for a,v in blocks.items()}
    for a in ['G','C']:
        err=max(float((acc[a][j]-saved[a][j]).abs().max()) for j in range(2));assert err<1e-8,(a,err);statcheck[a]=err
    WG=heads['G']['0.01'].numpy();WC=heads['C']['0.01'].numpy();G=data['G'];C=data['C'];d=1318
    U=np.concatenate([C[:,:-1],G[:,d:-1],np.ones((len(y),1))],1);data['U']=U
    WGu=np.zeros((U.shape[1],6));WGu[:d]=(float(trans['C']['out_norm'][1])/float(trans['G']['out_norm'][1]))*WG[:d]
    WGu[C.shape[1]-1:-1]=WG[d:-1];WGu[-1]=WG[-1]+((trans['C']['out_norm'][0][:d]-trans['G']['out_norm'][0][:d])/trans['G']['out_norm'][1]).numpy()@WG[:d]
    WCu=np.zeros_like(WGu);WCu[:C.shape[1]-1]=WC[:-1];WCu[-1]=WC[-1]
    errors={a:float(np.max(np.abs(U@w-x@h))) for a,w,x,h in [('G',WGu,G,WG),('C',WCu,C,WC)]};assert max(errors.values())<1e-8
    reference={a:verify(data[a],y,w,[0]*6) for a,w in [('G',WG),('C',WC)]}
    assert all(r['numerical_correct']==r['certified_correct'] for r in reference.values()),'Reference boundary instability'
    k=np.maximum(reference['G']['numerical_correct'],reference['C']['numerical_correct']).astype(int)
    # One fixed sufficient-witness subset, never an infeasibility certificate.
    pg=(G@WG).argmax(1);pc=(C@WC).argmax(1);selected=np.zeros(len(y),dtype=bool)
    for c in range(6):
        pred=pg if reference['G']['numerical_correct'][c]>=reference['C']['numerical_correct'][c] else pc
        selected|=(y==c)&(pred==c)
    for a,X in data.items():np.save(root/(a+'.features.private.npy'),X)
    np.save(root/'labels.private.npy',y);np.save(root/'selected.private.npy',selected)
    costs=dict(wall_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated(),cache_bytes=sum(x.nbytes for x in data.values()),frozen_transforms_reused=True)
    write(root/'prepare.costs.json',costs)
    seal=dict(source_commit=cfg['source_commit'],reference_commit='19bf801',target_stage='final_six_class',training_counts=[int((y==c).sum()) for c in range(6)],reference=reference,quotas=k.tolist(),stages=stages,dimensions={a:X.shape[1]-1 for a,X in data.items()},statistics_max_errors=statcheck,union_embedding_max_errors=errors,test_used=False,holdout_used_for_targets=False,sealed_at=time.time(),per_space_seconds=300,lp_cap_seconds=90,memory_cap_GiB=16,solver_threads=4,total_solve_allocation_seconds=900)
    write(root/'QUOTA_SEAL.json',seal);write(root/'status.json',dict(state='QUOTAS_SEALED_WAITING_DISPATCH',quotas=k.tolist()));print(json.dumps(seal),flush=True)
if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());start=time.time()
    try:
        main(cfg);write(Path(cfg['output'])/'PREPARE_EXIT.json',dict(exit_code=0,wall_seconds=time.time()-start))
    except Exception:
        write(Path(cfg['output'])/'failure.private.json',dict(traceback=traceback.format_exc()));write(Path(cfg['output'])/'PREPARE_EXIT.json',dict(exit_code=1,wall_seconds=time.time()-start));raise
