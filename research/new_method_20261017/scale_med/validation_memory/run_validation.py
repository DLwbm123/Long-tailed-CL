"""Fixed heads, permanent validation, and a separate privileged auditor."""
import json, os, sys, time, resource, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from core_attribution import ALPHAS, holdout, weights, norm_fit, stats, solve, ba

def write(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def load(path):return torch.load(path,weights_only=False)
def size(obj):
    if isinstance(obj,torch.Tensor):return obj.numel()*obj.element_size()
    if isinstance(obj,dict):return sum(size(v) for v in obj.values())
    if isinstance(obj,(tuple,list)):return sum(size(v) for v in obj)
    return 0
def augment(x):return torch.cat([x,torch.ones(len(x),1,dtype=x.dtype)],1)
def choose(scores,maximize=True):
    return max(ALPHAS,key=lambda a:((1 if maximize else -1)*scores[str(a)],a))
def vstats(x,y,seen):
    w=torch.tensor([1/int((y==c).sum()) for c in y],dtype=torch.float64)
    h=F.one_hot(y,seen).double()
    return x.T@(w[:,None]*x), x.T@(w[:,None]*h),len(y.unique())
def loss(st,w):
    g,h,c=st
    return float(((w*(g@w)).sum()-2*(w*h).sum()+c)/c)
def direct(x,y,w):
    e=(x@w-F.one_hot(y,w.shape[1])).square().sum(1)
    return float(torch.stack([e[y==c].mean() for c in y.unique()]).mean())
def merge(a,b):return a[0]+b[0],a[1]+b[1],a[2]+b[2]
def pad(st,seen):return st[0],F.pad(st[1],(0,seen-st[1].shape[1])),st[2]

def selfcheck():
    torch.manual_seed(74006)
    x=augment(torch.randn(24,4,dtype=torch.float64));y=torch.tensor([0]*8+[1]*8+[2]*8)
    w=torch.randn(5,3,dtype=torch.float64)
    assert abs(loss(vstats(x,y,3),w)-direct(x,y,w))<1e-10
    a=vstats(x[:16],y[:16],2);b=vstats(x[16:],y[16:],3)
    assert abs(loss(merge(pad(a,3),b),w)-direct(x,y,w))<1e-10
    # New competitor errors on old rows must be included.
    wp=w.clone();wp[:,2]+=2
    assert abs(loss(pad(a,3),wp)-direct(x[:16],y[:16],wp))<1e-10
    groups=[str(i//2) for i in range(len(y))];valid,_=holdout(y,groups)
    assert not {groups[i] for i in torch.where(valid)[0]}&{groups[i] for i in torch.where(~valid)[0]}
    xx,xy=stats(x,y,groups,3);a0,b0=stats(x[:16],y[:16],groups[:16],3);a1,b1=stats(x[16:],y[16:],groups[16:],3)
    assert torch.allclose(solve(xx,xy,.01,3,'cpu'),solve(a0+a1,b0+b1,.01,3,'cpu'),atol=1e-10)

def main(cfg):
    start=time.time();torch.set_num_threads(4)
    root=Path(cfg['output']);phase=os.environ['PHASE'];bc=json.loads(Path(cfg['benchmark_config']).read_text())
    source=json.loads(Path(bc['transfer_config']).read_text());prior=Path(bc['prior_rows']).resolve();testroot=Path(bc['output']).resolve()
    auditroot=root/'audit';retired=set();sealed=False
    def status(step,**kw):write(root/'status.json',dict(state='RUNNING',phase=phase,step=step,**kw))
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if phase=='fit':
            assert p not in retired,'Retired input accessed'
            if p.is_relative_to(testroot):assert p.name.endswith('.rows.private.json'),'Test features forbidden before seal'
            if p.is_relative_to(auditroot):assert len(args)>1 and isinstance(args[1],str) and 'w' in args[1],'Learner cannot read auditor rows'
        elif phase=='audit':
            assert not p.is_relative_to(prior),'Auditor cannot access training inputs'
            assert not p.is_relative_to(testroot),'Auditor cannot access test inputs'
        elif phase=='test':assert not p.is_relative_to(prior) and not p.is_relative_to(auditroot),'Test cannot access support/validation rows'
    sys.addaudithook(guard)
    status('start')
    if phase=='fit':
        selfcheck();write(root/'SELFCHECK.json',dict(pass_all=True))
        audits={};costs={};choices={}
        def birth(case,role,norm=None):
            n=case['name'];seen=case['old']+case['new'];rp=prior/(n+'.'+role+'.rows.private.json');fp=prior/(n+'.'+role+'.features.private.pt')
            rr=json.loads(rp.read_text());cached=load(fp);y=torch.tensor([r['label'] for r in rr]);groups=[r['identity_component'] for r in rr]
            assert torch.equal(cached['y'],y) and cached['components']==groups
            assert cached['x'].shape==(len(y),1536) and torch.isfinite(cached['x']).all()
            classes=range(case['old']) if role=='S_old' else range(case['old'],seen)
            assert set(y.tolist())==set(classes)
            assert all(int((y==c).sum())==source['support_counts'][n][c] for c in classes)
            valid,missing=holdout(y,groups);tr=~valid
            assert not {groups[i] for i in torch.where(tr)[0]}&{groups[i] for i in torch.where(valid)[0]}
            x=F.normalize(cached['x'].double(),dim=1);del cached
            trgroups=[groups[i] for i in torch.where(tr)[0].tolist()]
            if norm is None:norm=norm_fit(x[tr],weights(y[tr],trgroups))
            x=augment((x-norm[0])/norm[1]);train=stats(x[tr],y[tr],trgroups,seen);val=vstats(x[valid],y[valid],seen)
            with (auditroot/(n+'.'+role+'.private.pt')).open('wb') as f:
                torch.save(dict(x=x[valid],y=y[valid],groups=[groups[i] for i in torch.where(valid)[0].tolist()]),f)
            count=case['old'] if role=='S_old' else seen;tick=time.time()
            heads={str(a):solve(train[0],train[1][:,:count],a,count) for a in ALPHAS} if role=='S_old' else {}
            scores={a:ba(x[valid]@w,y[valid]) for a,w in heads.items()}
            info=dict(images=len(y),train_images=int(tr.sum()),validation_images=int(valid.sum()),validation_classes=len(y[valid].unique()),missing_classes=missing,solve_seconds=time.time()-tick,raw_validation_feature_bytes=int(valid.sum())*1536*cached_element_bytes)
            return train,val,norm,scores,info,(str(rp),str(fp)),heads
        cached_element_bytes=8
        for case in source['datasets']:
            n=case['name'];status('base',dataset=n);t=time.time()
            # Metadata-only preflight reuses the existing content audit.
            rr=[json.loads((prior/(n+'.'+r+'.rows.private.json')).read_text()) for r in ['S_old','S_new']]+[json.loads((testroot/(n+'.test.rows.private.json')).read_text())]
            for key in ['identity_component','relative_path']:
                ss=[{r[key] for r in rows} for rows in rr]
                assert all(not ss[i]&ss[j] for i in range(3) for j in range(i))
            assert len(rr[2])==bc['expected_test_images'][n];del rr
            train,oldval,norm,base,oldinfo,paths,baseheads=birth(case,'S_old')
            first=choose(base);del baseheads
            retired.update(str(Path(p).resolve()) for p in paths)
            caught=False
            try:open(paths[1],'rb')
            except AssertionError:caught=True
            assert caught
            status('increment',dataset=n)
            newtrain,newval,_,newbirth,newinfo,_,unused=birth(case,'S_new',norm);del unused,newbirth
            total=(train[0]+newtrain[0],train[1]+newtrain[1]);tick=time.time();seen=case['old']+case['new']
            heads={str(a):solve(*total,a,seen) for a in ALPHAS};fit_seconds=time.time()-tick
            # CURRENT/BIRTH need only current-stage validation, never auditor files.
            # Reopen current input during its birth stage; old inputs remain blocked.
            rp=prior/(n+'.S_new.rows.private.json');fp=prior/(n+'.S_new.features.private.pt')
            rows=json.loads(rp.read_text());cache=load(fp);y=torch.tensor([r['label'] for r in rows]);groups=[r['identity_component'] for r in rows]
            valid,_=holdout(y,groups);x=augment((F.normalize(cache['x'].double(),dim=1)-norm[0])/norm[1])
            current={a:ba(x[valid]@w,y[valid]) for a,w in heads.items()};del cache,x,y,groups,rows,valid
            archive={a:(base[a]*oldval[2]+current[a]*newval[2])/(oldval[2]+newval[2]) for a in base}
            vtotal=merge(oldval,newval);moment={a:loss(vtotal,w) for a,w in heads.items()};assert all(torch.isfinite(w).all() for w in heads.values())
            choices[n]=dict(FIRST=first,CURRENT=choose(current),BIRTH=choose(archive),MOMENT=choose(moment,False))
            torch.save(dict(heads=heads,norm=norm),root/(n+'.heads.private.pt'))
            torch.save(dict(train=total,validation=vtotal),root/(n+'.statistics.private.pt'))
            audits[n]=dict(old=oldinfo,new=newinfo,old_input_guard_pass=caught,full_validation_coverage=(oldval[2]+newval[2]==seen),covered_classes=oldval[2]+newval[2],seen_classes=seen,base_BA=base,current_BA=current,birth_BA=archive,moment_MSE=moment)
            d=len(total[0]);costs[n]=dict(fit_wall_seconds=time.time()-t,joint_solve_seconds=fit_seconds,train_dense_bytes=size(total),validation_dense_bytes=size(vtotal),train_packed_symmetric_bytes=d*(d+1)//2*8+size(total[1]),validation_packed_symmetric_bytes=d*(d+1)//2*8+size(vtotal[1]),candidate_heads_bytes=size(heads),normalization_bytes=size(norm),raw_validation_equivalent_float64_bytes=oldinfo['raw_validation_feature_bytes']+newinfo['raw_validation_feature_bytes'],auditor_file_bytes=sum((auditroot/(n+'.'+r+'.private.pt')).stat().st_size for r in ['S_old','S_new']))
            del train,newtrain,total,oldval,newval,vtotal,heads,norm
            write(root/'DATA_AUDIT.json',audits);write(root/'costs.json',costs);write(root/'LEARNER_CHOICES.json',choices)
        write(root/'MODEL_SEAL.json',dict(datasets=2,heads_per_dataset=3,learner_choices=choices,permanent_validation=True,old_inputs_retired=True,test_used=False,sealed_at=time.time()))
    elif phase=='audit':
        assert (root/'MODEL_SEAL.json').exists();selection=json.loads((root/'LEARNER_CHOICES.json').read_text());audit={}
        for case in source['datasets']:
            n=case['name'];status('validation',dataset=n);o=load(auditroot/(n+'.S_old.private.pt'));v=load(auditroot/(n+'.S_new.private.pt'))
            x=torch.cat([o['x'],v['x']]);y=torch.cat([o['y'],v['y']]);heads=load(root/(n+'.heads.private.pt'))['heads']
            fullba={a:ba(x@w,y) for a,w in heads.items()};mse={a:direct(x,y,w) for a,w in heads.items()}
            st=load(root/(n+'.statistics.private.pt'))['validation'];err=max(abs(mse[a]-loss(st,w)) for a,w in heads.items())
            assert err<1e-8 and choose(mse,False)==selection[n]['MOMENT']
            selection[n].update(FULL_BA=choose(fullba),FULL_MSE=choose(mse,False))
            controls=['FIRST','CURRENT','BIRTH'];strong=max(controls,key=lambda k:fullba[str(selection[n][k])])
            best=max(fullba.values());regret={k:100*(best-fullba[str(a)]) for k,a in selection[n].items()}
            if n=='ISIC':
                gate='STOP_NO_SELECTION_HEADROOM' if regret[strong]<=.5 else ('PASS_TO_TRANSFER' if regret['MOMENT']<=.5 and regret['MOMENT']<=.5*regret[strong] else 'STOP_SURROGATE_MISMATCH')
            else:gate='AUXILIARY_INCOMPLETE_COVERAGE'
            audit[n]=dict(selection=selection[n],validation_BA=fullba,validation_MSE=mse,regret_pp=regret,strongest_simple_control=strong,quadratic_max_abs_error=err,validation_BA_spread_pp=100*(max(fullba.values())-min(fullba.values())),old_validation_BA={a:ba(o['x']@w,o['y']) for a,w in heads.items()},new_validation_BA={a:ba(v['x']@w,v['y']) for a,w in heads.items()},gate=gate)
        write(root/'VALIDATION_AUDIT.json',audit);write(root/'AUDIT_SEAL.json',dict(selection=selection,choices_fixed_before_test=True,sealed_at=time.time()))
    elif phase=='test':
        assert (root/'MODEL_SEAL.json').exists() and (root/'AUDIT_SEAL.json').exists()
        from run_transfer import aggregate
        from run_fair_benchmark import bootstrap
        audit=json.loads((root/'VALIDATION_AUDIT.json').read_text());results={}
        for case in source['datasets']:
            n=case['name'];status('test_scoring',dataset=n);obj=load(root/(n+'.heads.private.pt'));cache=load(testroot/(n+'.test.features.private.pt'))
            rows=json.loads((testroot/(n+'.test.rows.private.json')).read_text());y=torch.tensor([r['label'] for r in rows]);groups=[r['identity_component'] for r in rows]
            assert torch.equal(cache['y'],y) and cache['components']==groups
            norm=obj['norm'];x=augment((F.normalize(cache['x'].double(),dim=1)-norm[0])/norm[1]);scores={a:x@w for a,w in obj['heads'].items()}
            assert all(torch.isfinite(s).all() for s in scores.values())
            sel=audit[n]['selection'];outputs={k:scores[str(a)] for k,a in sel.items()};metrics={k:aggregate(s,y,case['old']) for k,s in outputs.items()}
            intervals={}
            for ref in dict.fromkeys(['FIRST',audit[n]['strongest_simple_control']]):
                out={'B':outputs[ref],**{k:v for k,v in outputs.items() if k!=ref}}
                intervals[ref]=bootstrap(out,y,groups,case['old'],10000,74003)['arms']
            ref=audit[n]['strongest_simple_control'];m=metrics['MOMENT']['summary'];r=metrics[ref]['summary']
            lower=intervals[ref]['MOMENT']['all']['delta_BA_pp_CI95'][0] if ref!='MOMENT' else 0
            transfer=lower>0 and all(100*(m[k]['BA']-r[k]['BA'])>=-.5 for k in ['old','new'])
            gate=audit[n]['gate']
            if gate=='PASS_TO_TRANSFER':gate='EXPLORATORY_GATES_PASS_NO_AUTOMATIC_EXTENSION' if transfer else 'STOP_TRANSFER_UNCERTAIN'
            results[n]=dict(validation=audit[n],metrics=metrics,intervals=intervals,transfer_condition=transfer,decision=gate,candidate_test_BA={a:ba(s,y) for a,s in scores.items()})
            torch.save(dict(scores=outputs,y=y,groups=groups),root/(n+'.test_predictions.private.pt'));write(root/'results.json',results)
        write(root/'status.json',dict(state='COMPLETE',datasets=2,primary='ISIC',HK_validation_coverage='14/15',decision=results['ISIC']['decision'],automatic_extension=False))
    else:raise ValueError(phase)
    write(root/(phase+'.costs.json'),dict(wall_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,historical_cache_creation_excluded=True))
    print('COMPLETE',phase,flush=True)

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        write(Path(cfg['output'])/'failure.private.json',dict(phase=os.environ.get('PHASE'),traceback=traceback.format_exc()));write(Path(cfg['output'])/'status.json',dict(state='FAILED'));raise
