"""Fresh base-to-new replay; fixed seven-arm information attribution."""
import json,os,sys,time,traceback,resource
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from core_attribution import *
from run import Images,write_json,save
from run_transfer import aggregate,paired
from run_fair_benchmark import bootstrap
from run_v2 import local_scores

def main(cfg):
    start=time.time();torch.set_num_threads(4);root=Path(cfg['output']);root.mkdir(exist_ok=True)
    bench=json.loads(Path(cfg['benchmark_config']).read_text());source=json.loads(Path(bench['transfer_config']).read_text());prior=Path(bench['prior_rows']);oldtest=Path(bench['output'])
    def status(phase,**kw):write_json(root/'status.json',dict(state='RUNNING',phase=phase,elapsed=time.time()-start,**kw))
    status('preflight');selfcheck();r=directions(128,256,74004);rk=directions(2432,256,74005)
    rows={};audit={};retired=set();retired_files=set();allbase=set()
    for case in source['datasets']:
        n=case['name'];rows[n]={role:json.loads((prior/(n+'.'+role+'.rows.private.json')).read_text()) for role in ['S_old','S_new']}
        rows[n]['test']=json.loads((oldtest/(n+'.test.rows.private.json')).read_text())
        sets=[{x['identity_component'] for x in rr} for rr in rows[n].values()];paths=[{x['relative_path'] for x in rr} for rr in rows[n].values()]
        assert all(not sets[i]&sets[j] and not paths[i]&paths[j] for i in range(3) for j in range(i))
        assert len(rows[n]['test'])==bench['expected_test_images'][n]
        for role in ['S_old','S_new']:
            rr=rows[n][role];allowed=range(case['old']) if role=='S_old' else range(case['old'],case['old']+case['new'])
            assert set(x['label'] for x in rr)==set(allowed)
            assert all(sum(x['label']==c for x in rr)==source['support_counts'][n][c] for c in allowed)
            assert all((Path(bench['datasets'][n]['development_images'])/x['relative_path']).is_file() for x in rr)
        for role,rr in rows[n].items():
            group_labels={}
            for row in rr:group_labels.setdefault(row['identity_component'],set()).add(row['label'])
            assert all(len(v)==1 for v in group_labels.values())
        audit[n]={role:dict(images=len(rr),groups=len({x['identity_component'] for x in rr})) for role,rr in rows[n].items()}
        allbase.update(str((Path(bench['datasets'][n]['development_images'])/x['relative_path']).resolve()) for x in rows[n]['S_old'])
    write_json(root/'DATA_AUDIT.json',dict(datasets=audit,group_path_overlap=0,content_audit_reused=True,patient_isolation='UNKNOWN'))
    def access(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=str(Path(os.fsdecode(args[0])).resolve());assert p not in retired and p not in retired_files,'Retired base input accessed'
    sys.addaudithook(access)
    import timm
    from safetensors.torch import load_file
    torch.manual_seed(source['seed']);model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    model.load_state_dict({k:v for k,v in load_file(source['weight']).items() if not k.startswith('head.')},strict=True)
    model.eval().requires_grad_(False).cuda();proj=torch.load(source['projection'],weights_only=True).cuda()
    def extract(rr,images):
        for x,idx in DataLoader(Images(rr,dict(images=images)),batch_size=64,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=model.forward_features(x.cuda()).float()
            yield (z[:,1:]@proj).cpu(),z[:,0].cpu().double(),idx
    parity={}
    for n in rows:
        rr=json.loads((prior/(n+'.Qfit.rows.private.json')).read_text())[:64];p,h,idx=next(extract(rr,bench['datasets'][n]['development_images']))
        obj=torch.load(Path(source['output'])/n/'model.private.pt',weights_only=False);u,l=local_scores(p,obj['stats'],obj['reference'],source['tau'])
        ss={'ridge':torch.cat([h,torch.ones(len(h),1)],1)@obj['ridge'],'uniform':u,'local_shrink':l};previous=torch.load(Path(source['output'])/n/'Qfit.predictions.private.pt',weights_only=True)['scores'];parity[n]={}
        for k,s in ss.items():
            err=float((s-previous[k][:len(s)]).abs().max());assert err<1e-5 and torch.equal(s.argmax(1),previous[k][:len(s)].argmax(1));parity[n][k]=err
    del obj,previous,ss,p,h,u,l
    write_json(root/'PARITY.json',parity)
    costs={};selection={};models={}
    def raw(n,role):
        t=time.time();rr=rows[n][role];path=(oldtest/(n+'.test.features.private.pt')) if role=='test' else (prior/(n+'.'+role+'.features.private.pt'))
        cached=torch.load(path,weights_only=True);y=torch.tensor([x['label'] for x in rr]);groups=[x['identity_component'] for x in rr]
        assert torch.equal(cached['y'],y) and cached['components']==groups
        z={'g':F.normalize(cached['x'].double(),dim=1),'h':[],'m':[],'q':[],'v':[]};del cached
        images=bench['datasets'][n]['images' if role=='test' else 'development_images']
        for p,h,idx in extract(rr,images):
            m,q,v=local_moments(p,r)
            for k,value in [('h',h),('m',m),('q',q),('v',v)]:z[k].append(value)
            status('extract',dataset=n,role=role,processed=int(idx[-1])+1,total=len(rr))
        z.update({k:torch.cat(z[k]) for k in ['h','m','q','v']});assert all(torch.isfinite(x).all() for x in z.values())
        costs.setdefault(n,{})[role+'_extraction_seconds']=time.time()-t
        return z,y,groups
    for case in source['datasets']:
        n=case['name'];old=case['old'];seen=old+case['new'];z,y,groups=raw(n,'S_old');valid,missing=holdout(y,groups);tr=~valid
        wt=weights(y[tr],[groups[i] for i in torch.where(tr)[0].tolist()]);norm={k:norm_fit(v[tr],wt) for k,v in z.items()}
        z=normalized(z,norm,rk);norm['k']=norm_fit(z['k'][tr],wt);z['k']=(z['k']-norm['k'][0])/norm['k'][1]
        selection[n]=dict(validation_images=int(valid.sum()),excluded_singleton_classes=missing,arms={});state={};costs[n]['arms']={}
        for arm in ARMS:
            t=time.time();x=features(z,arm);xx,xy=stats(x[tr],y[tr],[groups[i] for i in torch.where(tr)[0].tolist()],seen);scores={}
            for alpha in ALPHAS:
                w=solve(xx,xy[:,:old],alpha,old);scores[str(alpha)]=ba(x[valid]@w,y[valid])
            alpha=max(ALPHAS,key=lambda a:(scores[str(a)],a));selection[n]['arms'][arm]=dict(alpha=alpha,validation_BA=scores)
            state[arm]=stats(x,y,groups,seen);costs[n]['arms'][arm]=dict(base_fit_seconds=time.time()-t,dimension=x.shape[1],sufficient_stats_bytes=sum(v.numel()*v.element_size() for v in state[arm]))
            status('base_stats',dataset=n,arm=arm)
        save(root/(n+'.base_stats.private.pt'),state);save(root/(n+'.normalization.private.pt'),norm)
        # Retire old per-image inputs before opening new-class images or caches.
        del z,y,groups,x,xx,xy,w,wt,tr,valid
        retired.update(str((Path(bench['datasets'][n]['development_images'])/x['relative_path']).resolve()) for x in rows[n]['S_old'])
        retired_files.add(str((prior/(n+'.S_old.features.private.pt')).resolve()))
        probe=next(iter(retired));caught=False
        try:open(probe,'rb')
        except AssertionError:caught=True
        assert caught
        write_json(root/(n+'.RETIREMENT.json'),dict(base_inputs_blocked=True,guard_probe_pass=True,retained='aggregate sufficient statistics, fixed transforms and selected alpha'))
        z,y,groups=raw(n,'S_new');z=normalized(z,norm,rk);heads={}
        for arm in ARMS:
            t=time.time();x=features(z,arm);xx,xy=stats(x,y,groups,seen);a,b=state[arm];a+=xx;b+=xy;alpha=selection[n]['arms'][arm]['alpha'];heads[arm]=solve(a,b,alpha,seen)
            extra=sum(t.numel()*t.element_size() for k in ARMS[arm] for t in norm[k])
            if arm in ['N','S','K']:extra+=r.numel()*r.element_size()
            if arm=='K':extra+=rk.numel()*rk.element_size()
            if arm in ['M','N','S','K']:extra+=proj.numel()*proj.element_size()
            costs[n]['arms'][arm].update(incremental_fit_seconds=time.time()-t,head_bytes=heads[arm].numel()*heads[arm].element_size(),normalizers_and_projection_bytes=extra)
            status('incremental_fit',dataset=n,arm=arm)
        save(root/(n+'.joint_stats.private.pt'),state);save(root/(n+'.heads.private.pt'),heads)
        models[n]=(norm,heads);del state,z,y,groups,x,xx,xy,a,b
        write_json(root/'SELECTION.json',selection);write_json(root/'costs.json',costs)
    assert allbase<=retired
    write_json(root/'MODEL_SEAL.json',dict(datasets=2,heads=12,test_used_for_fit=False,projection_seed=74004,capacity_seed=74005,old_inputs_retired=True))
    results=[];baseline=json.loads((oldtest/'results.json').read_text())
    for case in source['datasets']:
        n=case['name'];old=case['old'];norm,heads=models[n];z,y,groups=raw(n,'test');basehead=torch.load(Path(bench['b_root'])/(n+'.B.private.pt'),weights_only=True)
        outputs={'B':8*z['g']@basehead['frame']};z=normalized(z,norm,rk)
        for arm in ARMS:
            t=time.time();outputs[arm]=features(z,arm)@heads[arm];costs[n]['arms'][arm]['score_seconds']=time.time()-t
        assert all(torch.isfinite(s).all() for s in outputs.values());metrics={k:aggregate(v,y,old) for k,v in outputs.items()}
        reference=next(a for a in baseline if a['dataset']==n)['metrics']['B']
        assert metrics['B']==reference,'Frozen B changed'
        intervals={};pairs={}
        for candidate,ref in [('G0','B'),('G1','G0'),('M','G1'),('N','M'),('S','N'),('S','K')]:
            label=candidate+'-'+ref;intervals[label]=bootstrap({'B':outputs[ref],'candidate':outputs[candidate]},y,groups,old,10000,74003)['arms']['candidate'];pairs[label]=paired(outputs[ref],outputs[candidate],y,old)
        results.append(dict(dataset=n,metrics=metrics,intervals=intervals,paired=pairs,old=old,seen=old+case['new']))
        save(root/(n+'.predictions.private.pt'),dict(scores=outputs,y=y,groups=groups));write_json(root/'results.json',results)
    costs['total']=dict(wall_seconds=time.time()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,historical_cache_creation_excluded=True,torch_version=torch.__version__)
    write_json(root/'costs.json',costs);write_json(root/'status.json',dict(state='COMPLETE',datasets=2,arms=7,test_selection=False,extra_settings=False));print('COMPLETE',flush=True)
if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        write_json(Path(cfg['output'])/'failure.private.json',dict(error=traceback.format_exc()));write_json(Path(cfg['output'])/'status.json',dict(state='FAILED'));raise
