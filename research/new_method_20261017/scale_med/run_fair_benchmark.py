"""Frozen first-transition comparison on existing test/reserved manifests."""
import json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from run import Images,write_json
from run_transfer import aggregate,paired
from run_v2 import local_scores,fuse
from run_b_distance import offsets
from run_b_consensus import consensus,retention


def bootstrap(outputs,y,groups,old,draws,seed):
    """Same class-stratified group draws for every arm; retain image-level BA."""
    rng=np.random.default_rng(seed);labels=y.numpy();arms=list(outputs)
    correct=np.stack([(outputs[k].argmax(1)==y).numpy() for k in arms]);k=outputs['B'].shape[1]
    sampled=np.zeros((len(arms),draws,k));group_ba=np.zeros((len(arms),k))
    for c in range(k):
        ids=np.where(labels==c)[0];keys=sorted({groups[i] for i in ids});members=[[i for i in ids if groups[i]==g] for g in keys]
        size=np.array([len(v) for v in members]);hits=np.stack([correct[:,v].sum(1) for v in members],1)
        group_ba[:,c]=(hits/size).mean(1)
        index=rng.integers(0,len(keys),size=(draws,len(keys)));denom=size[index].sum(1)
        for j in range(len(arms)):sampled[j,:,c]=hits[j,index].sum(1)/denom
    b=arms.index('B');result={}
    for j,arm in enumerate(arms):
        result[arm]={}
        for group,cs in [('all',slice(0,k)),('old',slice(0,old)),('new',slice(old,k))]:
            delta=(sampled[j,:,cs].mean(1)-sampled[b,:,cs].mean(1))*100
            lo,hi=np.quantile(delta,[.025,.975])
            result[arm][group]=dict(delta_BA_pp_CI95=[float(lo),float(hi)],group_equal_BA=float(group_ba[j,cs].mean()))
    return dict(draws=draws,seed=seed,unit='known_identity_group_within_class',method='paired percentile bootstrap, image-level BA',small_class_warning=True,arms=result)


def main(cfg):
    root=Path(cfg['output']);start=time.time();torch.set_num_threads(4)
    assert cfg['primary']=='consensus' and cfg['local_lambda']==.5
    assert (root/'B_FEATURE_PARITY.json').exists()
    source=json.loads(Path(cfg['transfer_config']).read_text());scale=json.loads(Path(cfg['scales']).read_text());pri=Path(cfg['prior_rows']);transfer=Path(source['output'])
    allowed={pri/(n+'.Qfit.rows.private.json') for n in ['ISIC','HK']}
    # Historical E predictions and unrelated historical heads are never inputs.
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.parent==pri:assert p in allowed,'Historical input outside parity rows'
    sys.addaudithook(audit)
    import timm
    from safetensors.torch import load_file
    torch.manual_seed(source['seed'])
    model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    model.load_state_dict({k:v for k,v in load_file(source['weight']).items() if not k.startswith('head.')},strict=True)
    model.eval().requires_grad_(False).cuda();proj=torch.load(source['projection'],weights_only=True).cuda()
    def extract(rows,images):
        for x,idx in DataLoader(Images(rows,dict(images=images)),batch_size=64,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=model.forward_features(x.cuda()).float()
            yield (z[:,1:]@proj).cpu(),z[:,0].cpu().double(),idx
    models={n:torch.load(transfer/n/'model.private.pt',weights_only=False) for n in ['ISIC','HK']}
    parity={}
    for n in ['ISIC','HK']:
        rows=json.loads((pri/(n+'.Qfit.rows.private.json')).read_text())[:64]
        p,f,idx=next(extract(rows,cfg['datasets'][n]['development_images']));obj=models[n]
        u,l=local_scores(p,obj['stats'],obj['reference'],source['tau']);r=torch.cat([f,torch.ones(len(f),1)],1)@obj['ridge']
        previous=torch.load(transfer/n/'Qfit.predictions.private.pt',weights_only=True)['scores']
        parity[n]={}
        for name,s in [('ridge',r),('uniform',u),('local_shrink',l)]:
            err=float((s-previous[name][:len(s)]).abs().max())
            assert err<cfg['score_parity_atol'] and torch.equal(s.argmax(1),previous[name][:len(s)].argmax(1)),f'Raw encoder parity failure: {name} {err}'
            parity[n][name]=dict(samples=len(s),max_abs_score_error=err,tolerance=cfg['score_parity_atol'])
    write_json(root/'LOCAL_FEATURE_PARITY.json',parity)
    write_json(root/'MODEL_SEAL.json',dict(sealed_at=time.time(),model_updates=0,scales_refitted=False,primary='consensus'))
    results=[]
    for case in source['datasets']:
        n=case['name'];old=case['old'];obj=models[n];sc=scale[n]['scales']
        rows=json.loads((root/(n+'.test.rows.private.json')).read_text());y=torch.tensor([r['label'] for r in rows]);groups=[r['identity_component'] for r in rows]
        cache=torch.load(root/(n+'.test.features.private.pt'),weights_only=True)
        assert torch.equal(cache['y'],y) and cache['components']==groups
        feature=F.normalize(cache['x'],dim=1)
        heads={k:torch.load(Path(path)/(n+suffix),weights_only=True) for k,path,suffix in [('A',cfg['a_root'],'.ridge.private.pt'),('B',cfg['b_root'],'.B.private.pt')]}
        assert all(all(not v.any() for v in h['model']) for h in heads.values())
        outputs={k:8*feature@h['frame'] for k,h in heads.items()}
        blocks={k:[] for k in ['ridge','uniform','local_shrink']}
        for p,f,idx in extract(rows,cfg['datasets'][n]['images']):
            u,l=local_scores(p,obj['stats'],obj['reference'],source['tau']);r=torch.cat([f,torch.ones(len(f),1)],1)@obj['ridge']
            for k,v in [('ridge',r),('uniform',u),('local_shrink',l)]:blocks[k].append(v)
            write_json(root/'status.json',dict(state='RUNNING',phase='frozen_test_scoring',dataset=n,processed=int(idx[-1])+1,total=len(rows)))
        outputs.update({k:torch.cat(v) for k,v in blocks.items()});b=outputs['B'];l=outputs['local_shrink'];d=l+offsets(obj,source['tau'])[None,:]
        outputs.update(distance_local=d,B_density=fuse(b,l,sc,.5),B_distance=fuse(b,d,sc,.5),consensus=consensus(b,l,d,sc))
        assert all(torch.isfinite(s).all() for s in outputs.values())
        metrics={k:aggregate(s,y,old) for k,s in outputs.items()}
        rr=retention(b,{k:outputs[k] for k in ['B_density','B_distance']},outputs['consensus'],y,old)
        changed=outputs['consensus'].argmax(1)!=b.argmax(1)
        assert all(not (changed&(outputs[k].argmax(1)==b.argmax(1))).any() for k in ['B_density','B_distance'])
        interval=bootstrap(outputs,y,groups,old,cfg['bootstrap_draws'],cfg['bootstrap_seed'])
        classes=[];bc=b.argmax(1)==y;cc=outputs['consensus'].argmax(1)==y
        for c in range(old+case['new']):
            m=y==c;classes.append(dict(class_index=c,count=int(m.sum()),known_groups=len({groups[i] for i in torch.where(m)[0].tolist()}),rescue=int((m&~bc&cc).sum()),harm=int((m&bc&~cc).sum())))
        state_bytes={k:sum(h['frame'].numel()*h['frame'].element_size() for h in [v]) for k,v in heads.items()}
        state_bytes['additional_local_mean_cov']=sum(t.numel()*t.element_size() for s in obj['stats'] for t in s['local'])
        state_bytes['additional_reference_cov']=obj['reference']['local'][1].numel()*obj['reference']['local'][1].element_size()
        state_bytes['additional_projection']=proj.numel()*proj.element_size()
        results.append(dict(dataset=n,split=cfg['datasets'][n]['public_split_name'],old=old,seen=old+case['new'],metrics=metrics,paired={k:paired(s,outputs['consensus'],y,old) for k,s in outputs.items() if k!='consensus'},retention=rr,classes=classes,intervals=interval,logical_state_bytes_excluding_encoders=state_bytes))
        torch.save(dict(scores=outputs,labels=y,groups=groups),root/(n+'.predictions.private.pt'))
        write_json(root/'results.json',results)
    write_json(root/'costs.json',dict(local_stage_wall_seconds=time.time()-start,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__,model_updates=0,scale_fits=0,prior_feature_cost_excluded=True))
    write_json(root/'status.json',dict(state='COMPLETE',datasets=2,conditions=2,arms=len(outputs),models_fit=0,primary='consensus',parameter_scan=False,standard_test_access=True))
    print('BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        write_json(Path(cfg['output'])/'failure.private.json',dict(phase='benchmark',error=traceback.format_exc()));raise
