"""Finite full-support replay reference; diagnostics never select or update models."""
import json
import os
import time
from pathlib import Path
import torch
from check_fullslot_retention_memory import train
from check_reserved_competition_memory import DT, scores, lowrank_fixture, training_batch
from check_memory_stability import uncertainty, decision

MODES=('moment','selected','candidate','full_support')
GROUPS=('all','old','new','accepted_old','fallback_old')


def statistics(ss,y,old_n):
    assert bool(torch.isfinite(ss).all()) and 0<old_n<ss.shape[1]
    correct_score=ss[torch.arange(len(y)),y]
    other=ss.clone();other[torch.arange(len(y)),y]=-torch.inf
    old_max=ss[:,:old_n].max(1).values;new_max=ss[:,old_n:].max(1).values
    return dict(pred=ss.argmax(1),CE=-ss.log_softmax(1)[torch.arange(len(y)),y],margin=correct_score-other.max(1).values,
                opposite_margin=correct_score-torch.where(y<old_n,new_max,old_max),old_new_gap=old_max-new_max)


def transitions(a,b,c,old_n):
    ac=a==c;bc=b==c
    r=dict(both_correct=int((ac&bc).sum()),rescued=int((~ac&bc).sum()),harmed=int((ac&~bc).sum()),both_wrong=int((~ac&~bc).sum()),wrong_to_different_wrong=int((~ac&~bc&(a!=b)).sum()))
    for tag,mask_a,mask_b in [('old',a<old_n,b<old_n),('new',a>=old_n,b>=old_n)]:
        r['baseline_errors_to_'+tag]=int((~ac&mask_a).sum());r['full_support_errors_to_'+tag]=int((~bc&mask_b).sum())
        r['correct_to_wrong_'+tag]=int((ac&~bc&mask_b).sum());r['wrong_'+tag+'_to_correct']=int((~ac&bc&mask_a).sum())
    assert sum(r[k] for k in ('both_correct','rescued','harmed','both_wrong'))==len(a)
    assert r['rescued']-r['harmed']==int(bc.sum()-ac.sum())
    assert sum(r['correct_to_wrong_'+g] for g in ('old','new'))==r['harmed']
    assert sum(r['wrong_'+g+'_to_correct'] for g in ('old','new'))==r['rescued']
    return r


def unit_checks():
    ss=torch.tensor([[2.,1.,0.],[0.,2.,2.],[1.,0.,2.]],dtype=DT);y=torch.tensor([0,2,1]);s=statistics(ss,y,2)
    assert s['pred'].tolist()==[0,1,2] and s['margin'].tolist()==[1.,0.,-2.]
    literal=torch.log(torch.exp(ss).sum(1))-ss[torch.arange(3),y]
    assert float((literal-s['CE']).abs().max())<1e-12
    assert s['opposite_margin'].tolist()==[2.,0.,-2.] and s['old_new_gap'].tolist()==[2.,0.,-1.]
    a=torch.tensor([0,0,1,2,1]);b=torch.tensor([0,1,0,0,2]);r=transitions(a,b,0,2)
    assert [r[k] for k in ('both_correct','rescued','harmed','both_wrong')]==[1,2,1,1]
    old={0:(torch.zeros(2,3,dtype=DT),torch.ones(2,dtype=DT)/2),1:(torch.zeros(5,3,dtype=DT),torch.ones(5,dtype=DT)/5)}
    _,yy,p=training_batch({2:torch.zeros(3,3,dtype=DT)},old)
    assert abs(float(p.sum())-1)<1e-12 and all(abs(float(p[yy==c].sum())-1/3)<1e-12 for c in range(3))
    return dict(status='PASS',CE_checked=True,margin_and_tie_checked=True,paired_transitions_checked=True,class_mass_checked=True,adapter_updates=0,memory_updates=0)


def load(cfg):
    prior=Path(os.environ['Q354_PRIOR']);sid=cfg['id']
    state=torch.load(prior/(sid+'.state.private.pt'),map_location='cpu',weights_only=True)
    sealed=torch.load(prior/(sid+'.sealed.private.pt'),map_location='cpu',weights_only=True)
    candidate=torch.load(Path(os.environ['Q354_CANDIDATE'])/(sid+'.sealed.private.pt'),map_location='cpu',weights_only=True)
    assert state['case']==sealed['case']==candidate['case']==cfg and candidate['adapter_updates']==48
    return state,sealed,candidate


def train_case(cfg,lock,journal):
    start=time.monotonic();root=Path(os.environ['Q354_ROOT']);state,sealed,_=load(cfg)
    old_n=cfg['task_sizes'][0];seen=sum(cfg['task_sizes'][:2]);x,y,counts=lowrank_fixture(cfg)
    old={c:(x[y==c],torch.ones(int((y==c).sum()),dtype=DT)/int((y==c).sum())) for c in range(old_n)}
    current={c:x[y==c] for c in range(old_n,seen)}
    journal(dict(phase='full_support_training',completed_updates=0))
    model,cost=train(state['teacher'],current,old,state['frame'],list(range(seen)),steps=lock['full_support_steps'],full=False)
    result=dict(status='COMPLETED',case=cfg,training=cost,adapter_updates=48,memory_updates=0,synthetic_counts=counts,
                old_support_samples=sum(len(z) for z,p in old.values()),new_support_samples=sum(len(z) for z in current.values()),
                accepted_classes=sealed['accepted_classes'],sealed=True,evaluation_generated=False,pipeline_seconds=time.monotonic()-start)
    torch.save(dict(case=cfg,model=model,training=cost,adapter_updates=48,memory_updates=0),root/(cfg['id']+'.sealed.private.pt'))
    (root/(cfg['id']+'.training.private.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    journal(dict(phase='full_support_sealed',completed_updates=48))
    return result


def evaluate(lock,journal):
    start=time.monotonic();root=Path(os.environ['Q354_ROOT']);cases=lock['cases'][1:-1]
    for cfg in cases:
        saved=torch.load(root/(cfg['id']+'.sealed.private.pt'),map_location='cpu',weights_only=True)
        assert saved['case']==cfg and saved['adapter_updates']==48 and saved['training']['updates']==48
    prior=json.loads((Path(os.environ['Q354_SOURCE'])/'docs/candidate_transfer_design1_20261016/CLASS_RECALL.json').read_text())
    prior_counts={(r['scene'],r['class_index'],m):r[m+'_correct'] for r in prior for m in MODES[:3]}
    classes=[];paired=[];summaries=[];groups=[];intervals=[];decisions=[];evaluation=[]
    for base in lock['base_cases']:
        scale=base['id'];old_n=base['task_sizes'][0];seen=sum(base['task_sizes'][:2])
        ev=dict(base,seed=lock['evaluation_seeds'][scale],n=base['classes']*1000,count_weights=[1]*base['classes'])
        x,y,counts=lowrank_fixture(ev);keep=y<seen;x=x[keep];y=y[keep];assert counts==[1000]*base['classes']
        for cfg in [c for c in cases if c['scale']==scale]:
            sid=cfg['id'];journal(dict(phase='evaluation',scene=sid));state,sealed,candidate=load(cfg)
            full=torch.load(root/(sid+'.sealed.private.pt'),map_location='cpu',weights_only=True)
            models=dict(sealed['models'],candidate=candidate['model'],full_support=full['model']);stats={}
            for m,model in models.items():
                with torch.no_grad():ss=torch.cat([scores(x[i:i+512],model,state['frame'])[:,:seen] for i in range(0,len(x),512)])
                stats[m]=statistics(ss,y,old_n)
                for c in range(seen):
                    cm=y==c;s=stats[m];p=s['pred'][cm];margin=s['margin'][cm];correct=int((p==c).sum())
                    if m!='full_support':assert correct==prior_counts[sid,c,m]
                    classes.append(dict(scene=sid,model=m,class_index=c,group='old' if c<old_n else 'new',subgroup='accepted_old' if c in sealed['accepted_classes'] else 'fallback_old' if c<old_n else 'new',count=int(cm.sum()),correct=correct,
                        CE=float(s['CE'][cm].mean()),margin_mean=float(margin.mean()),margin_quantiles=torch.quantile(margin,torch.tensor(lock['margin_quantiles'],dtype=DT)).tolist(),
                        negative_margin=int((margin<0).sum()),zero_margin=int((margin==0).sum()),near_zero_margin=int((margin.abs()<=lock['near_zero_margin_absolute_limit']).sum()),
                        opposite_margin_mean=float(s['opposite_margin'][cm].mean()),old_new_gap_mean=float(s['old_new_gap'][cm].mean()),
                        errors_to_old=int(((p!=c)&(p<old_n)).sum()),errors_to_new=int(((p!=c)&(p>=old_n)).sum())))
                for group in GROUPS:
                    rr=[r for r in classes if r['scene']==sid and r['model']==m and (group=='all' or r['group']==group or r['subgroup']==group)]
                    avg=lambda key:sum(r[key] for r in rr)/len(rr) if rr else None
                    groups.append(dict(scene=sid,model=m,group=group,classes=len(rr),count=sum(r['count'] for r in rr),BA=sum(r['correct']/r['count'] for r in rr)/len(rr) if rr else None,
                                       macro_CE=avg('CE'),margin_mean=avg('margin_mean'),opposite_margin_mean=avg('opposite_margin_mean'),old_new_gap_mean=avg('old_new_gap_mean')))
            for i,m in enumerate(MODES[:3]):
                pair='full_support_vs_'+m;pp=[]
                for row in [r for r in classes if r['scene']==sid and r['model']=='full_support']:
                    c=row['class_index'];cm=y==c;a=stats[m]['pred'][cm];b=stats['full_support']['pred'][cm]
                    pp.append(dict(scene=sid,pair=pair,baseline=m,class_index=c,group=row['group'],subgroup=row['subgroup'],count=row['count'],baseline_correct=int((a==c).sum()),full_support_correct=row['correct'],**transitions(a,b,c,old_n)))
                paired.extend(pp);ci=uncertainty(pp,old_n,lock['bootstrap_seeds'][sid]+i,lock['bootstrap_draws'])
                intervals.extend(dict(scene=sid,pair=pair,**v) for v in ci);decisions.append(dict(scene=sid,pair=pair,confirmatory=False,**decision(ci)))
                for group in GROUPS:
                    a=next(v for v in groups if v['scene']==sid and v['model']==m and v['group']==group)
                    b=next(v for v in groups if v['scene']==sid and v['model']=='full_support' and v['group']==group)
                    summaries.append(dict(scene=sid,pair=pair,group=group,classes=a['classes'],baseline_BA=a['BA'],full_support_BA=b['BA'],BA_difference=b['BA']-a['BA'] if a['classes'] else None,
                        CE_difference=b['macro_CE']-a['macro_CE'] if a['classes'] else None,margin_difference=b['margin_mean']-a['margin_mean'] if a['classes'] else None,
                        opposite_margin_difference=b['opposite_margin_mean']-a['opposite_margin_mean'] if a['classes'] else None))
        evaluation.append(dict(scale=scale,seed=ev['seed'],per_class=1000,samples=len(y),same_prior_bank=True,all_three_prior_counts_reproduced=True,all_four_new_models_sealed=True))
    return dict(status='COMPLETED',class_rows=classes,paired=paired,summaries=summaries,groups=groups,intervals=intervals,decisions=decisions,evaluation=evaluation,evaluation_seconds=time.monotonic()-start,adapter_updates=0,memory_updates=0)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=unit_checks() if cfg['kind']=='unit' else evaluate(lock,journal) if cfg['kind']=='evaluation' else train_case(cfg,lock,journal)
    r.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,val_accessed=False,test_accessed=False)
    assert not r['GPU_initialized'];return r
