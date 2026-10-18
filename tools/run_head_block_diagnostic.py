"""Four sealed head blocks; score interventions only, no fitting or selection."""
import itertools
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA, DT, features
from query_node_weights import select
from run_affine_pair_diagnostic import paired
from run_head_competition_diagnostic import competition, stats, unit as competition_unit
from run_real_query_node import load, save

MODELS=('A','old_B_new_A','old_A_new_B','B')
COMPARISONS=tuple(itertools.combinations(MODELS,2))


def score_parts(score,y,old):
    i=torch.arange(len(y));other=score.clone();other[i,y]=-torch.inf
    rival,margin,rank,ties=competition(score,y)
    ro=other[:,:old].argmax(1);rn=old+other[:,old:].argmax(1)
    return dict(true_score=score[i,y],rival_score=score[i,rival],old_rival_score=score[i,ro],
                new_rival_score=score[i,rn],margin=margin,old_margin=score[i,y]-score[i,ro],
                new_margin=score[i,y]-score[i,rn],rank=rank,ties=ties,rival=rival,old_rival=ro,new_rival=rn)


def score_effects(a,b,y,old):
    i=torch.arange(len(y));pa=score_parts(a,y,old);pb=score_parts(b,y,old);j=pa['rival']
    own=b[i,y]-a[i,y];fixed=b[i,j]-a[i,j]
    switch_gain=pb['rival_score']-b[i,j]
    effects=dict(true_score_delta=own,fixed_rival_score_delta=fixed,fixed_rival_margin_delta=own-fixed,
                 actual_rival_score_delta=pb['rival_score']-pa['rival_score'],actual_margin_delta=pb['margin']-pa['margin'],
                 rival_switch_penalty=switch_gain,old_rival_score_delta=pb['old_rival_score']-pa['old_rival_score'],
                 new_rival_score_delta=pb['new_rival_score']-pa['new_rival_score'],
                 old_margin_delta=pb['old_margin']-pa['old_margin'],new_margin_delta=pb['new_margin']-pa['new_margin'])
    error=float((effects['actual_margin_delta']-(own-fixed-switch_gain)).abs().max())
    assert error<1e-10 and bool((switch_gain>=-1e-12).all())
    return effects,pa,pb,error


def unit():
    competition_unit()
    a=torch.tensor([[4.,3.,2.,1.],[3.,2.,5.,1.],[1.,4.,2.,3.]],dtype=DT)
    b=torch.tensor([[3.,5.,4.,1.],[4.,3.,4.,2.],[4.,2.,5.,1.]],dtype=DT);y=torch.tensor([0,2,3])
    effects,pa,pb,error=score_effects(a,b,y,2)
    assert pa['rival'].tolist()==[1,0,1] and pb['rival'].tolist()==[1,0,2]
    assert effects['true_score_delta'].tolist()==[-1.,-1.,-2.]
    assert effects['fixed_rival_margin_delta'].tolist()==[-3.,-2.,0.]
    assert effects['rival_switch_penalty'].tolist()==[0.,0.,3.]
    assert effects['actual_margin_delta'].tolist()==[-3.,-2.,-3.]
    o=torch.cat((b[:,:2],a[:,2:]),1);n=torch.cat((a[:,:2],b[:,2:]),1)
    assert torch.equal(b-a,(o-a)+(n-a))
    z=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=DT);wa=a[:2];wb=b[:2]
    assert torch.equal(z@torch.cat((wb[:,:2],wa[:,2:]),1),torch.cat(((z@wb)[:,:2],(z@wa)[:,2:]),1))
    return dict(status='PASS',fixed_rival_switch_identity=True,head_block_score_equivalence=True,
                ties_and_paired_accounting=True,head_fits=0,linear_solves=0,gradient_evaluations=0,margin_identity_error=error)


def flat_stats(values,mask):
    return {key+'_'+stat:value for key,array in values.items() for stat,value in stats(array[mask]).items()}


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    name=json.loads(os.environ['Q356_JOB'])['dataset'];job_id=name+'_blocks';case=next(v for v in lock['datasets'] if v['name']==name)
    old=case['old'];seen=old+case['new'];sealed=False
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['logit_scale']==BETA==8 and list(MODELS)==lock['models']
    assert [list(v) for v in COMPARISONS]==lock['comparisons']
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit_info=unit()
    bases={k:Path(cfg[k+'_root']) for k in ('prior','ridge','offline')}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        allowed=dict(prior={'DELIVERY.private.json'},ridge={'DELIVERY.private.json',name+'.ridge.private.pt'},offline={'DELIVERY.private.json',name+'.B.private.pt'})
        if sealed:
            allowed['prior'].update(name+'.'+q+'.features.private.pt' for q in ('Qfit','Qgate'))
            allowed['offline'].add(name+'_evaluate.json')
        for key,base in bases.items():
            if p.is_relative_to(base):assert p.parent==base and p.name in allowed[key],'Sealed input boundary'
        if p.is_relative_to(bases['prior'].parent) and not p.is_relative_to(root) and not any(p.is_relative_to(b) for b in bases.values()):
            raise AssertionError('Unfrozen experiment input')
    sys.addaudithook(audit)
    blocked=[bases['prior']/(name+'.'+s) for s in ('S_old.features.private.pt','S_new.features.private.pt','teacher.private.pt','E.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt')]
    blocked += [bases['offline']/(name+'.moments.private.pt'),bases['offline']/(name+'_evaluate.json'),bases['ridge']/(name+'.herding.private.pt')]
    for p in blocked:
        try:p.open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    for key,base in bases.items():
        receipt=json.loads((base/'DELIVERY.private.json').read_text())
        assert receipt['design_source_commit']==lock['inputs'][key]['source'] and receipt['public_commit']==lock['inputs'][key]['delivery']
    a=load(bases['ridge']/(name+'.ridge.private.pt'));b=load(bases['offline']/(name+'.B.private.pt'))
    for obj in (a,b):
        assert obj['frame'].shape==(1536,seen) and obj['frame'].dtype==DT and bool(torch.isfinite(obj['frame']).all()) and 'bias' not in obj
        assert len(obj['model'])==2 and all(v.shape==(1536,8) and v.dtype==DT and not bool(v.any()) for v in obj['model'])
    assert all(torch.equal(x,y) for x,y in zip(a['model'],b['model']))
    heads=dict(A=a['frame'],old_B_new_A=torch.cat((b['frame'][:,:old],a['frame'][:,old:]),1),
               old_A_new_B=torch.cat((a['frame'][:,:old],b['frame'][:,old:]),1),B=b['frame'])
    sealed=True;sealed_at=time.time();save(root/(job_id+'.progress.json'),dict(phase='all_four_fixed_heads_sealed',sealed_at=sealed_at))
    reference=json.loads((bases['offline']/(name+'_evaluate.json')).read_text());assert reference['source_commit']==lock['inputs']['offline']['source']
    classes=[];summary=[];confusion=[];pairs=[];group_pairs=[];transitions=[];score_rows=[];effects_rows=[];rivals=[];identities=[];gates=[];counts=[]
    for query in ('Qfit','Qgate'):
        data=load(bases['prior']/(name+'.'+query+'.features.private.pt'));x,y=data['x'],data['y'];n=len(y)
        assert x.shape==(n,1536) and x.dtype==DT and bool(torch.isfinite(x).all()) and y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
        assert [int((y==c).sum()) for c in range(seen)]==lock['query_counts'][name][query]
        with torch.no_grad():
            z=torch.cat([features(x[i:i+256],a['model']) for i in range(0,n,256)])
            scores={key:BETA*z@head for key,head in heads.items()}
        score_identity=float((scores['B']-scores['old_B_new_A']-scores['old_A_new_B']+scores['A']).abs().max());assert score_identity<1e-10
        assert torch.allclose(scores['old_B_new_A'],torch.cat((scores['B'][:,:old],scores['A'][:,old:]),1),atol=1e-12,rtol=0)
        assert torch.allclose(scores['old_A_new_B'],torch.cat((scores['A'][:,:old],scores['B'][:,old:]),1),atol=1e-12,rtol=0)
        predictions={key:s.argmax(1) for key,s in scores.items()};counts.append(dict(query=query,count=n))
        for key,s in scores.items():
            pred=predictions[key];part=score_parts(s,y,old);ce=-s.log_softmax(1)[torch.arange(n),y]
            cm=torch.bincount(y*seen+pred,minlength=seen*seen).reshape(seen,seen)
            assert int(cm.sum())==n
            for c in range(seen):
                mask=y==c;pc=pred[mask];count=int(mask.sum());group='old' if c<old else 'new'
                row=dict(dataset=name,query=query,model=key,class_index=c,group=group,count=count,correct=int((pc==c).sum()),recall=float((pc==c).to(DT).mean()),CE=float(ce[mask].mean()),errors_to_old=int(((pc!=c)&(pc<old)).sum()),errors_to_new=int(((pc!=c)&(pc>=old)).sum()))
                classes.append(row)
                assert int(cm[c].sum())==count and int(cm[c,c])==row['correct']
                if key in ('A','B'):
                    ref=next(v for v in reference['class_rows'] if v['query']==query and v['model']==key and v['class_index']==c)
                    assert all(row[k]==ref[k] for k in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-ref['CE'])<1e-10
                for j in range(seen):confusion.append(dict(dataset=name,query=query,model=key,true_class=c,predicted_class=j,count=int(cm[c,j])))
                quantities={k:part[k] for k in ('true_score','rival_score','old_rival_score','new_rival_score','margin','old_margin','new_margin','rank')}
                score_rows.append(dict(dataset=name,query=query,model=key,class_index=c,group=group,count=count,true_score_tie_samples=int((part['ties'][mask]>0).sum()),**flat_stats(quantities,mask)))
                for kind in ('rival','old_rival','new_rival'):
                    for j in sorted(set(part[kind][mask].tolist())):
                        m=mask&(part[kind]==j);rivals.append(dict(dataset=name,query=query,model=key,class_index=c,kind=kind,competitor=j,count=int(m.sum()),wrong_count=int((m&(pred!=y)).sum())))
            for group in ('all','old','new'):
                rr=[v for v in classes if v['query']==query and v['model']==key and (group=='all' or v['group']==group)]
                summary.append(dict(dataset=name,query=query,model=key,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
        max_error=0.
        for ka,kb in COMPARISONS:
            comparison=ka+'__to__'+kb;pa=predictions[ka];pb=predictions[kb];ca=pa==y;cb=pb==y
            effects,sa,sb,error=score_effects(scores[ka],scores[kb],y,old);max_error=max(max_error,error)
            masks=dict(all=torch.ones_like(y,dtype=torch.bool),rescued=~ca&cb,harmed=ca&~cb,both_correct=ca&cb,both_wrong=~ca&~cb)
            for c in range(seen):
                mask=y==c;group='old' if c<old else 'new'
                pairs.append(dict(dataset=name,query=query,comparison=comparison,class_index=c,group=group,**paired(pa[mask],pb[mask],y[mask])))
                for outcome,m in masks.items():
                    v=mask&m
                    if not bool(v.any()):continue
                    effects_rows.append(dict(dataset=name,query=query,comparison=comparison,class_index=c,group=group,outcome=outcome,count=int(v.sum()),competitor_changed=int((sa['rival'][v]!=sb['rival'][v]).sum()),old_competitor_changed=int((sa['old_rival'][v]!=sb['old_rival'][v]).sum()),new_competitor_changed=int((sa['new_rival'][v]!=sb['new_rival'][v]).sum()),**flat_stats(effects,v)))
            for group in ('all','old','new'):
                mask=torch.ones_like(y,dtype=torch.bool) if group=='all' else y<old if group=='old' else y>=old
                group_pairs.append(dict(dataset=name,query=query,comparison=comparison,group=group,**paired(pa[mask],pb[mask],y[mask])))
            changed=pa!=pb
            for c,aa,bb in sorted(set(zip(y[changed].tolist(),pa[changed].tolist(),pb[changed].tolist()))):
                transitions.append(dict(dataset=name,query=query,comparison=comparison,class_index=c,from_prediction=aa,to_prediction=bb,outcome='harmed' if aa==c else 'rescued' if bb==c else 'wrong_to_other_wrong',count=int(((y==c)&(pa==aa)&(pb==bb)).sum())))
            assert sum(v['count'] for v in transitions if v['query']==query and v['comparison']==comparison)==int(changed.sum())
        get=lambda key:{v['group']:v['BA'] for v in summary if v['query']==query and v['model']==key}
        for key in MODELS[1:]:
            gate=select(get('A'),get(key));gates.append(dict(dataset=name,query=query,model=key,pass_gate=gate['selected']=='candidate',**gate))
        identities.append(dict(query=query,score_additivity_max_error=score_identity,margin_decomposition_max_error=max_error))
    save(root/(job_id+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,
         class_rows=classes,summaries=summary,confusion_rows=confusion,paired_rows=pairs,paired_group_rows=group_pairs,transition_rows=transitions,
         score_rows=score_rows,effect_rows=effects_rows,competitor_rows=rivals,gates=gates,identities=identities,query_counts=counts,unit=unit_info,
         published_A_B_aggregates_matched=True,role_guard_passed=True,shared_feature_sample_forwards=sum(v['count'] for v in counts),
         model_logit_sample_evaluations=4*sum(v['count'] for v in counts),mixed_head_assemblies=2,mixed_head_payload_bytes=2*a['frame'].numel()*a['frame'].element_size(),
         head_fits=0,production_linear_solves=0,engineering_linear_solves=0,gradient_evaluations=0,adapter_updates=0,encoder_updates=0,
         support_access=False,full_statistics_access=False,E_access=False,GPU_used=False,new_labels=0,query_used_for_selection=False,
         operation_seconds=time.monotonic()-start,scope='Fixed four-cell head-block diagnostic only. Mixed heads use B historical information and need not minimize a shared ridge objective. No equal-budget, upper-bound, selected-method or independent-validation claim.'))


if __name__=='__main__':main()
