"""Frozen R1 point estimates and paired component bootstrap, saved predictions only."""
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import f1_score

METHODS=('C','H','K','RA','RD');SEEDS=(1993,1994,1995)
METRICS=('balanced_accuracy','old_macro_recall','current_macro_recall','tail_rank2')

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.part');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
def mean(x):return float(np.mean(x)) if len(x) else None

def predict_columns(raw, order=None):
    if order is None:return raw.argmax(1)
    columns=np.argsort(np.asarray(order)[:raw.shape[1]],kind='stable')
    return columns[raw[:,columns].argmax(1)]

def analyze_unit(p,method,seed,stage,tie_original_label=False):
    raw=p['raw'];y=p['y'];order=p['order'];pred=predict_columns(raw,order if tie_original_label else None);correct=pred==y;seen=raw.shape[1];known=(0,4,6)[stage]
    prefix=dict(method=method,order_seed=seed,session=stage,seen_classes=seen,old_classes=known,current_classes=seen-known,split='test',predictor='raw_sum' if method in ('C','H','K') else 'original_V3_CBRidge')
    assert len(set(p['ids'].tolist()))==len(y) and np.isfinite(raw).all()
    assert np.array_equal(order[y],p['original'])
    pc=[]
    for c in range(seen):
        mask=y==c;components=np.unique(p['component'][mask]);lesions=np.unique(p['lesion'][mask])
        pc.append(dict(**prefix,head_index=c,original_label=int(order[c]),n_images=int(mask.sum()),n_components=len(components),n_lesions=len(lesions),
                       recall=100*float(correct[mask].mean()),component_equal_recall=100*float(np.mean([correct[mask & (p['component']==g)].mean() for g in components])),
                       lesion_equal_recall=100*float(np.mean([correct[mask & (p['lesion']==g)].mean() for g in lesions]))))
    recalls=np.array([r['recall'] for r in pc]);old=list(range(known));current=list(range(known,seen))
    m=dict(**prefix,balanced_accuracy=float(recalls.mean()),accuracy=100*float(correct.mean()),macro_f1=100*float(f1_score(y,pred,labels=range(seen),average='macro',zero_division=0)),
           old_macro_recall=mean(recalls[old]),current_macro_recall=mean(recalls[current]),
           component_equal_balanced_accuracy=mean([r['component_equal_recall'] for r in pc]),lesion_equal_balanced_accuracy=mean([r['lesion_equal_recall'] for r in pc]),
           n_images=len(y),n_components=len(np.unique(p['component'])),worst_class_recall=float(recalls.min()),zero_recall_classes=[r['original_label'] for r in pc if r['recall']==0])
    a=m['old_macro_recall'];b=m['current_macro_recall'];m['HM_old_current_macro']=None if a is None else (2*a*b/(a+b) if a+b else 0.)
    for group,classes in [('head_rank2',(0,1)),('mid_rank4',(2,3,4,5)),('tail_rank2',(6,7))]:
        cs=[c for c in range(seen) if order[c] in classes];m[group]=mean(recalls[cs]);m[group+'_n_classes']=len(cs)
        if group=='tail_rank2':
            for scope,cs0 in [('old',old),('current',current)]:
                ids=[c for c in cs0 if order[c] in classes];m[scope+'_tail_rank2']=mean(recalls[ids]);m[scope+'_tail_n_classes']=len(ids)
    errors=[]
    for scope,cs in [('current',current),('old',old)]:
        mask=np.isin(y,cs)
        same=np.isin(pred,cs);events={'correct':correct,'to_other':~same,'wrong_within':same&~correct}
        r=dict(**prefix,scope=scope,n_images=int(mask.sum()),n_classes=len(cs))
        for name,event in events.items():
            r[name+'_sample_weighted']=100*float(event[mask].mean()) if mask.any() else None
            r[name+'_class_macro']=100*mean([event[y==c].mean() for c in cs]) if cs else None
        if cs:
            for suffix in ('sample_weighted','class_macro'):assert abs(sum(r[k+'_'+suffix] for k in events)-100)<1e-10
            restricted=np.array(cs)[predict_columns(raw[mask][:,cs],order[cs] if tie_original_label else None)]
            m['restricted_'+scope+'_BA_diagnostic']=100*mean([(restricted[y[mask]==c]==c).mean() for c in cs])
        else:m['restricted_'+scope+'_BA_diagnostic']=None
        errors.append(r)
    return m,pc,errors

def bootstrap_weights(p,resamples=2000,seed=91001):
    rng=np.random.default_rng(seed);weights=np.zeros((resamples,len(p['y'])),dtype=np.int32)
    for label in range(8):
        indices=np.where(p['original']==label)[0];groups=p['component'][indices];unique,inverse=np.unique(groups,return_inverse=True)
        assert len(unique)>0
        for group in unique:assert len(np.unique(p['original'][p['component']==group]))==1,'GROUP_LABEL_CONFLICT'
        # Multinomial counts exactly implement n_groups cluster draws with replacement.
        counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=resamples)
        weights[:,indices]=counts[:,inverse]
    return weights

def boot_unit(p,weights,tie_original_label=False):
    pred=predict_columns(p['raw'],p['order'] if tie_original_label else None);correct=pred==p['y'];seen=p['raw'].shape[1];recalls=[]
    for c in range(seen):
        ix=np.where(p['y']==c)[0];w=weights[:,ix];recalls.append(100*(w@correct[ix])/w.sum(axis=1))
    return np.stack(recalls,axis=1)

def complete(out):
    out=Path(out);start=time.monotonic();assert read(out/'E2_COMPLETE.json')['status']=='PASS'
    private=out/'private/test';res=out/'results';res.mkdir(exist_ok=True)
    data={};metrics=[];classes=[];errors=[];diag=[]
    for method in METHODS:
        for seed in SEEDS:
            for stage in range(3):
                owner='C' if stage==0 and method in ('H','K') else method
                with np.load(private/f'{owner}_{seed}_s{stage}.npz') as f:p={k:f[k] for k in f.files}
                data[method,seed,stage]=p;m,pc,er=analyze_unit(p,method,seed,stage);metrics.append(m);classes+=pc;errors+=er
                if method in ('C','H','K'):
                    for head in ('main','few'):
                        dm,_,_=analyze_unit(dict(p,raw=p[head]),method,seed,stage);dm['diagnostic_head']=head;diag.append(dm)
    assert len(metrics)==45 and len(classes)==270
    assert len({(r['method'],r['order_seed'],r['session'],r['original_label']) for r in classes})==270
    # All methods/stages are exact subsets of one fixed sample universe with identical class/group identities.
    canonical=data['C',1993,2];idmap={v:i for i,v in enumerate(canonical['ids'])}
    for p in data.values():
        ix=np.array([idmap[s] for s in p['ids']]);assert np.array_equal(p['original'],canonical['original'][ix]) and np.array_equal(p['component'],canonical['component'][ix])
    for method in ('RA','RD'):
        preds=[data[method,s,2]['order'][data[method,s,2]['raw'].argmax(1)] for s in SEEDS]
        assert all(np.array_equal(preds[0],p) for p in preds[1:]),'BLOCKED_RIDGE_FINAL_ORDER_PREDICTIONS'
    by={(r['method'],r['order_seed'],r['session']):r for r in metrics};final=[];summary=[];paired=[];forgetting=[]
    for method in METHODS:
        for seed in SEEDS:
            ms=[by[method,seed,t] for t in range(3)];last=dict(ms[-1]);last.update(AvgBA_all=mean([m['balanced_accuracy'] for m in ms]),AvgBA_inc=mean([m['balanced_accuracy'] for m in ms[1:]]))
            final.append(last)
            for label in range(8):
                rs=[r for r in classes if r['method']==method and r['order_seed']==seed and r['original_label']==label]
                assert rs;eligible=len(rs)>1
                forgetting.append(dict(method=method,order_seed=seed,original_label=label,first_session=rs[0]['session'],eligible=eligible,
                  first_recall=rs[0]['recall'],final_recall=rs[-1]['recall'],first_minus_final=rs[0]['recall']-rs[-1]['recall'] if eligible else None,
                  peak_minus_final=max(r['recall'] for r in rs)-rs[-1]['recall'] if eligible else None))
        fs=[r for r in final if r['method']==method]
        for metric in METRICS+('AvgBA_all','AvgBA_inc','accuracy','macro_f1','component_equal_balanced_accuracy'):
            vs=[r[metric] for r in fs];det=method in ('RA','RD')
            summary.append(dict(method=method,metric=metric,values=vs,mean=mean(vs),std_ddof1=float(np.std(vs,ddof=1)) if not det else None,
                min_fixed_order_value=min(vs),max_fixed_order_value=max(vs),n_trained_runs=3 if not det else 0,
                interpretation='one deterministic final fit; old/current and stage averages use distinct partitions' if det else 'three fixed paired trained seed/order runs'))
    for other in ('C','H','RA','RD'):
        for stage in range(3):
            for metric in METRICS:
                vs=[]
                for seed in SEEDS:
                    k=by['K',seed,stage][metric];v=by[other,seed,stage][metric];d=None if k is None or v is None else k-v;vs.append(d)
                    paired.append(dict(contrast='K-'+other,session=stage,order_seed=seed,metric=metric,difference_pp=d,mean=None,std_ddof1=None))
                valid=all(v is not None for v in vs)
                paired.append(dict(contrast='K-'+other,session=stage,order_seed='fixed_three_mean',metric=metric,difference_pp=None,mean=mean(vs) if valid else None,std_ddof1=float(np.std(vs,ddof=1)) if valid else None))
    differences={k:[by['K',s,2][k]-by['C',s,2][k] for s in SEEDS] for k in METRICS}
    zeros=[]
    for r in classes:
        if r['method']=='K' and r['session'] in (1,2) and r['head_index']>=r['old_classes'] and r['recall']==0:
            cr=next(x for x in classes if (x['method'],x['order_seed'],x['session'],x['original_label'])==('C',r['order_seed'],r['session'],r['original_label']))
            if cr['recall']>0:zeros.append(dict(order_seed=r['order_seed'],session=r['session'],original_label=r['original_label'],C_recall=cr['recall'],K_recall=0))
    improving=sum(differences['balanced_accuracy'][i]>0 and differences['current_macro_recall'][i]>0 for i in range(3))
    gate_checks={'final_BA_mean_improves':mean(differences['balanced_accuracy'])>0,'current_mean_gain_ge_10pp':mean(differences['current_macro_recall'])>=10,
      'old_mean_loss_le_5pp':mean(differences['old_macro_recall'])>=-5,'at_least_2_of_3_ba_current_improve':improving>=2,'no_additional_current_zero':len(zeros)==0}
    tail={o:mean([by['K',s,2]['tail_rank2']-by[o,s,2]['tail_rank2'] for s in SEEDS]) for o in ('C','H','RA','RD')}
    gate=dict(status='MECHANICAL_CHECK_COMPLETE',checks=gate_checks,transfer_gates_pass=all(gate_checks.values()),K_minus_C={k:dict(values=v,mean=mean(v),std_ddof1=float(np.std(v,ddof=1))) for k,v in differences.items()},
      paired_improving=improving,new_current_zero= zeros,tail_differences_pp=tail,tail_improved_vs_C=tail['C']>0,worst_seed_old_change_pp=min(differences['old_macro_recall']),independent_confirmation=False,clinical_noninferiority=False)
    write(out/'TRANSFER_GATE_CHECK.json',gate)
    for name,rows in [('test_session_metrics',metrics),('test_per_class_metrics',classes),('test_paired_differences',paired),('test_final_summary',summary),('test_final_by_order',final),('test_forgetting',forgetting),('test_error_decomposition',errors),('diagnostic_head_metrics',diag)]:csvwrite(res/(name+'.csv'),rows)
    # Shared component weights across every method, fixed seed and stage; no forward and no refit.
    btstart=time.monotonic();weights=bootstrap_weights(canonical);boots={};intervals=[]
    for key,p in data.items():
        method,seed,stage=key;ix=[idmap[s] for s in p['ids']];rc=boot_unit(p,weights[:,ix]);known=(0,4,6)[stage];seen=(4,6,8)[stage]
        subsets={'balanced_accuracy':list(range(seen)),'old_macro_recall':list(range(known)),'current_macro_recall':list(range(known,seen)),
                 'tail_rank2':[i for i in range(seen) if p['order'][i] in (6,7)]}
        for metric,cs in subsets.items():
            if cs:boots[key+(metric,)]=rc[:,cs].mean(1)
    def ci(name,seed,stage,metric,values,point,kind):
        low,high=np.percentile(values,[2.5,97.5]);intervals.append(dict(object=name,order_seed=seed,session=stage,metric=metric,point=point,lower95=float(low),upper95=float(high),type=kind,
            bootstrap_unit='class-stratified identity_component',resamples=2000,rng_seed=91001,estimand='image-within-class recall; fixed trained models',independent_confirmation=False))
    for method in METHODS:
        for metric in METRICS:
            vals=[]
            for seed in SEEDS:
                b=boots[method,seed,2,metric];vals.append(b);ci(method,seed,2,metric,b,by[method,seed,2][metric],'conditional point-metric interval')
            # Ridge overall BA/tail is a single deterministic reference, not three-training inference.
            ci(method,'fixed_three_mean' if method in ('C','H','K') else 'deterministic_partition_average',2,metric,np.mean(vals,axis=0),mean([by[method,s,2][metric] for s in SEEDS]),'conditional fixed-model summary')
    for other in ('C','H','RA','RD'):
        for stage in range(3):
            for metric in METRICS:
                vals=[];points=[]
                for seed in SEEDS:
                    if ('K',seed,stage,metric) not in boots:continue
                    d=boots['K',seed,stage,metric]-boots[other,seed,stage,metric];point=by['K',seed,stage][metric]-by[other,seed,stage][metric]
                    vals.append(d);points.append(point);ci('K-'+other,seed,stage,metric,d,point,'conditional paired difference')
                if vals:ci('K-'+other,'fixed_three_mean',stage,metric,np.mean(vals,axis=0),mean(points),'conditional paired difference; fixed models averaged, seeds not resampled')
    csvwrite(res/'test_bootstrap_intervals.csv',intervals)
    write(out/'BOOTSTRAP_AUDIT.json',dict(status='PASS',seconds=time.monotonic()-btstart,resamples=2000,seed=91001,component_counts_by_class={str(c):len(np.unique(canonical['component'][canonical['original']==c])) for c in range(8)},
      statistical_unit='class-stratified connected component',shared_draws_all_methods_orders_stages=True,training_seeds_resampled=False,additional_forwards=0,
      exclusions='does not cover adaptive selection, training-seed population, unknown patient correlation or external domain shift'))
    audit=dict(status='COMPLETE_LOCKED_HOLDOUT_EVAL',logical_main_rows=45,logical_per_class_rows=270,logical_predictions=sum(len(p['y']) for p in data.values()),
        transfer_gates_pass=gate['transfer_gates_pass'],independent_confirmation=False,new_neural_epochs=0,new_optimizer_steps=0,new_test_prediction_usage=read(out/'PREDICTION_USAGE_R1.json'),
        analysis_seconds=time.monotonic()-start,further_run_authorized=False,scheduled_monitoring='PAUSED; not restarted')
    write(out/'COMPLETION_AUDIT.json',audit)
    write(out/'NEXT_DECISION.json',dict(status='STOP',technical_status=audit['status'],transfer_gates_pass=gate['transfer_gates_pass'],further_run_authorized=False,
      independent_confirmation=False,reason='R1 E0-E3 complete; no follow-on training or testing authorized'))
    print(json.dumps({'status':audit['status'],'gate':gate,'summary':summary},ensure_ascii=False),flush=True)

def selfcheck():
    p=dict(raw=np.eye(8),y=np.arange(8),original=np.arange(8),order=np.arange(8),ids=np.array([str(i) for i in range(8)]),component=np.array([str(i) for i in range(8)]),lesion=np.array([str(i) for i in range(8)]))
    m,pc,e=analyze_unit(p,'C',1993,2);assert m['balanced_accuracy']==100 and len(pc)==8 and all(r['correct_sample_weighted']==100 for r in e)
    w=bootstrap_weights(p,25);assert np.array_equal(w,np.ones_like(w));assert np.all(boot_unit(p,w)==100)
    q=dict(p,raw=np.roll(np.eye(8),1,axis=1));m,_,e=analyze_unit(q,'C',1993,2);assert m['balanced_accuracy']==0
    assert all(abs(sum(r[k+'_class_macro'] for k in ('correct','to_other','wrong_within'))-100)<1e-10 for r in e)
    # Two correlated images in one component get equal weights, with variable image denominator.
    ix=np.array([0,0,0,1,2,3,4,5,6,7]);r={k:v[ix] if k not in ('order',) else v for k,v in p.items()};r['component']=np.array(['a','a','b','1','2','3','4','5','6','7'])
    w=bootstrap_weights(r,100);assert np.array_equal(w[:,0],w[:,1]);assert np.all(w[:,0]+w[:,2]==2)
    print('R1_ANALYSIS_SELFCHECK_PASS')

if __name__=='__main__':
    if os.environ.get('R1_SELFCHECK')=='1':selfcheck()
    else:complete(read(os.environ['R1_CONFIG'])['out'])
