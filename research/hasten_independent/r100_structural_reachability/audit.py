"""Saved-only closeout: recompute decisions, gates and finite constructions."""
import json,csv,time,resource,itertools,statistics,os
from pathlib import Path
import torch
from common import write

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(4000,4000));torch.set_num_threads(4)
    start=time.time();cpu=time.process_time();r=Path(__file__).resolve().parent
    cfg=json.loads((r/'RUN_CONFIG.private.json').read_text());summary=json.loads((r/'RESULT_SUMMARY.json').read_text())
    rows=list(csv.DictReader((r/'CONTEXTS.csv').open()));assert len(rows)==24
    totals={role:{k:0 for k in ['rescue','harm','changes','tiny_harm','teacher_correct_tiny_exposures']} for role in ['B','F']}
    for row in rows:
        s,b=int(row['stage']),int(row['block']);p=torch.load(r/f'CONTEXT_s{s}_b{b}.private.pt',map_location='cpu',weights_only=False)
        C=sum(map(len,cfg['stages'][:s+1]));tail=torch.tensor(cfg['train_counts'][:C])<250;old=torch.arange(C)<(sum(map(len,cfg['stages'][:s])) if s else 0)
        for role in ['B','F']:
            d=p['results'][role];y=d['labels'];pred=d['predictions'];assert pred.shape==(9,len(y)) and int(pred.min())>=0 and int(pred.max())<C
            den=torch.bincount(y,minlength=C).double();assert (den>0).all()
            rec=torch.stack([torch.bincount(y[v==y],minlength=C).double()/den for v in pred])
            macro=rec.mean(1);t=rec[:,tail].mean(1);o=rec[:,old].mean(1) if old.any() else macro*0
            reward=macro-macro[0]-(o[0]-o).clamp_min(0)-(t[0]-t).clamp_min(0)
            if role=='B':
                actions=list(itertools.product(range(3),repeat=2));chosen=min(range(9),key=lambda j:(-float(reward[j]),sum(a!=0 for a in actions[j]),j));assert chosen==p['selected_on_B']==int(row['B_selected_plan'])
            j=int(row['B_selected_plan']);refgood=pred[0]==y;v=pred[j];tiny=torch.isin(y,torch.tensor([1,2,12]))
            for metric,x in [('macro',macro),('tail',t),('old',o)]:assert abs(float((x[j]-x[0])*100)-float(row[f'{role}_selected_{metric}_delta_pp']))<1e-10
            vals={'rescue':int(((v==y)&~refgood).sum()),'harm':int(((v!=y)&refgood).sum()),'changes':int((v!=pred[0]).sum()),'tiny_harm':int(((v!=y)&refgood&tiny).sum()),'teacher_correct_tiny_exposures':int((refgood&tiny).sum())}
            for k,val in vals.items():
                if k!='teacher_correct_tiny_exposures':assert val==float(row[f'{role}_selected_{k}'])
                totals[role][k]+=val
    for s in range(6):
        p=torch.load(r/f'CONSTRUCTION_s{s}.private.pt',map_location='cpu',weights_only=False)
        assert torch.isfinite(p['base']).all() and len(p['blocks'])==4 and len(set(c for b in p['blocks'] for c in b))==8
        for options in p['candidates']:
            for v in options:assert torch.isfinite(v).all() and torch.allclose(v.norm(dim=1),torch.ones(2),atol=2e-5)
    stages=list(csv.DictReader((r/'STAGES.csv').open()));assert len(stages)==6
    for k,v in summary['mean'].items():assert abs(statistics.mean(float(x[k]) for x in stages)-v)<1e-10
    mean=lambda key:statistics.mean(float(x[key]) for x in stages)
    both=lambda role:sum(float(x[f'{role}_selected_macro_delta_pp'])>=-1e-8 and float(x[f'{role}_selected_tail_delta_pp'])>=-1e-8 for x in stages)
    A=[mean('B_selected_macro_delta_pp')>=.5-1e-8,mean('B_selected_tail_delta_pp')>=-1e-8,both('B')>=4,totals['B']['rescue']>0]
    B=[mean('F_selected_macro_delta_pp')>=.3-1e-8,mean('F_selected_tail_delta_pp')>=-1e-8,both('F')>=4,totals['F']['tiny_harm']==0]
    B += [mean(f'F_selected_{metric}_delta_pp')-mean(f'F_{ref}_{metric}_delta_pp')>=-1e-8 for ref in ['fixed','uniform'] for metric in ['macro','tail']]
    assert A==list(summary['gate_A'].values()) and B==list(summary['gate_B'].values())
    assert (all(A) and all(B))==summary['passed']
    write(r/'SAVED_CLOSEOUT_VERIFICATION.json',{'passed':True,'contexts':24,'stage_constructions':6,'all_gate_booleans_recomputed':True,'all_selected_metrics_rescue_harm_recomputed':True,'B_selection_no_F_selection_verified':True,'constructions_finite_unit_norm':True,'totals':totals,'optimizer_encoder_new_score_forward_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'scope':'Saved prediction tensors only; repeated stage/block exposures are not independent cases.'})
