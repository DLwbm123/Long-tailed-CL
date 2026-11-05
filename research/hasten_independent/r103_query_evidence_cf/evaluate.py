"""Only after both streams seal: nine fixed deterministic readouts and oracle."""
import csv,json,gc
from pathlib import Path
import torch
from common import write,save,guard
from router import Router,ARMS,features
from fusion import Fusion,scores
from teacher_base import new_tail
from teacher_vision import load_delta
from evidence_views import encode_views

MODES=['T','CENTER','MULTI','UNIFORM']+ARMS
def csvout(path,rows):
    if not rows:return
    with Path(path).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def run(c,trainrows,trainy,store,counts):
    root=Path(c['root_output']);seal=json.loads((root/'GLOBAL_LEARNING_SEAL.json').read_text());assert seal['router_endpoints']==50 and seal['fusion_endpoints']==60
    r=Path(c['output']);rows=list(csv.DictReader((Path(c['manifest'])/'val.csv').open()));assert len(rows)==1698
    labels=torch.tensor([c['order'].index(int(v['original_label'])) for v in rows]);stages=[];classrows=[];groups=[];harmrows=[];routes=[];oracles=[]
    for s in range(5):
        guard(c);write(r/'STATUS.json',{'phase':'VAL','stage':s,'counts':counts});folder=r/f's{s}';C=sum(map(len,c['stages'][:s+1]));ids=[i for i in range(len(rows)) if labels[i]<C];y=labels[ids].cuda();birth=torch.tensor([j for j,cl in enumerate(c['stages'][:s+1]) for _ in cl],device='cuda');head=torch.load(Path(c['reuse_root'])/'strong_heads'/f'head{s}.private.pt',map_location='cpu',weights_only=False)
        m,_=new_tail(store.bundle,c['seed']);load_delta(m,head['delta']);m.eval().requires_grad_(False)
        z=encode_views(c,rows,ids,store,{'FULL':m},counts,'VAL')['FULL'];models=[]
        for action in [1,2]:
            fm=Fusion().cuda();fm.load_state_dict(torch.load(folder/'FULL'/f'fusion{action}_step512.private.pt',weights_only=False)['model']);fm.eval().requires_grad_(False);models.append(fm)
        ss=scores(z,head['prototypes'],models);p=ss.double().softmax(-1);phi=features(ss,z.cuda(),head['train_counts'].cuda(),birth,s);candidate=ss.argmax(-1);pred={name:candidate[:,a] for a,name in enumerate(['T','CENTER','MULTI'])};pred['UNIFORM']=p.mean(1).argmax(-1);soft={};policies={};actions={}
        with torch.no_grad():
            for arm in ARMS:
                rm=Router().cuda();rm.load_state_dict(torch.load(folder/f'{arm}_step512.private.pt',weights_only=False)['model']);rm.eval().requires_grad_(False);policy=rm(phi).double().softmax(-1);a=policy.argmax(-1);pred[arm]=candidate.gather(1,a[:,None]).squeeze(1);soft[arm]=(policy*(candidate==y[:,None]).double()).sum(-1);policies[arm]=policy;actions[arm]=a
        den=torch.bincount(y,minlength=C);assert (den>0).all();tail=torch.tensor(c['train_counts'][:C],device='cuda')<250;old=birth<s;teacher_good=pred['T']==y;record={}
        for mode in MODES:
            good=pred[mode]==y;cc=torch.bincount(y[good],minlength=C);rec=cc.double()/den
            record[mode]={'micro':float(good.double().mean()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100) if tail.any() else None,'old_tail':float(rec[old&tail].mean()*100) if (old&tail).any() else None}
            for cl in range(C):
                mask=y==cl;harm=int((mask&teacher_good&~good).sum());rescue=int((mask&~teacher_good&good).sum())
                classrows.append({'stage':s,'mode':mode,'class':cl,'legacy_index':c['legacy_indices'][cl],'original_label':c['order'][cl],'tiny':cl in c['tiny_indices'],'count':int(den[cl]),'correct':int(cc[cl]),'recall':float(rec[cl]*100)})
                harmrows.append({'stage':s,'mode':mode,'class':cl,'legacy_index':c['legacy_indices'][cl],'original_label':c['order'][cl],'count':int(den[cl]),'harm':harm,'rescue':rescue,'final_macro_contribution_pp':(rescue-harm)/int(den[cl])/C*100,'old_tail':bool(old[cl]&tail[cl])})
            for age,am in [('old',old),('new',~old),('all',torch.ones_like(old))]:
                for freq,fm in [('head',~tail),('tail',tail),('all',torch.ones_like(tail))]:
                    mask=am&fm;groups.append({'stage':s,'mode':mode,'group':age+'_'+freq,'class_count':int(mask.sum()),'macro':float(rec[mask].mean()*100) if mask.any() else None})
            if mode in ARMS:
                a=actions[mode];policy=policies[mode];correct=(candidate==y[:,None]).double();h=correct[:,0,None]*(1-correct);soft_harm=(policy*h).sum(-1)
                # Class-balanced analytic harm uses all class queries as denominator.
                byclass=torch.stack([soft_harm[y==cl].mean() for cl in range(C)])
                for cl in [-1]+list(range(C)):
                    mask=torch.ones_like(y,dtype=torch.bool) if cl==-1 else y==cl;n=int(mask.sum());local=mask&(a>0)
                    routes.append({'stage':s,'mode':mode,'class':cl,'count':n,'action0_fraction':float((a[mask]==0).double().mean()),'action1_fraction':float((a[mask]==1).double().mean()),'action2_fraction':float((a[mask]==2).double().mean()),'local_selected_count':int(local.sum()),'local_rescue':int((local&~teacher_good&good).sum()),'local_harm':int((local&teacher_good&~good).sum()),'soft_expected_micro':float(soft[mode][mask].mean()*100),'hard_micro':float(good[mask].double().mean()*100),'KL_to_qref':float((policy[mask]*(policy[mask].log()-torch.tensor([.8,.1,.1],device='cuda').log())).sum(-1).mean()),'entropy':float(-(policy[mask]*policy[mask].log()).sum(-1).mean()),'analytic_old_tail_harm_macro':float(byclass[old&tail].mean()) if (old&tail).any() else 0.})
        oracle=(candidate==y[:,None]).any(-1)
        for cl in range(C):
            mask=y==cl;oracles.append({'stage':s,'class':cl,'legacy_index':c['legacy_indices'][cl],'count':int(den[cl]),'teacher_correct':int((mask&teacher_good).sum()),'any_candidate_correct':int((mask&oracle).sum()),'teacher_wrong_candidate_rescue':int((mask&~teacher_good&oracle).sum()),'diagnostic_not_deployable':True})
        # FULL V0 identity to R102 was already checked on TRAIN; no historical VAL errors used.
        save(r/f'VAL_predictions_s{s}.private.pt',{'labels':y.cpu(),'ids':ids,'predictions':{k:v.cpu() for k,v in pred.items()},'actions':{k:v.cpu() for k,v in actions.items()},'candidate_correct':(candidate==y[:,None]).cpu(),'global_seal_before_read':True})
        stages.append({'stage':s,'modes':record});del m,models,z,ss,p,phi,pred,policies,actions;gc.collect();torch.cuda.empty_cache()
    summary={'stream':c['stream_name'],'primary':'GRPO_CF','mean':{},'mean_valid_stages':{},'final':stages[-1]['modes'],'comparisons':{},'TEST':0,'conclusion':'COMPLETE_FIXED_MECHANISM_TRIAL'}
    for mode in MODES:
        summary['mean'][mode]={};summary['mean_valid_stages'][mode]={}
        for key in ['micro','macro','tail','old_tail']:
            vv=[x['modes'][mode][key] for x in stages if x['modes'][mode][key] is not None];summary['mean'][mode][key]=sum(vv)/len(vv) if vv else None;summary['mean_valid_stages'][mode][key]=len(vv)
    for ref in MODES:
        if ref=='GRPO_CF':continue
        summary['comparisons'][ref]={phase:{key:summary[phase]['GRPO_CF'][key]-summary[phase][ref][key] if summary[phase]['GRPO_CF'][key] is not None and summary[phase][ref][key] is not None else None for key in ['micro','macro','tail','old_tail']} for phase in ['mean','final']}
    write(r/'RESULT_SUMMARY.json',summary);write(r/'STAGE_METRICS.json',stages)
    csvout(r/'STAGE_METRICS.csv',[dict(stage=x['stage'],mode=mode,**v) for x in stages for mode,v in x['modes'].items()]);csvout(r/'CLASS_STAGE.csv',classrows);csvout(r/'GROUP_STAGE.csv',groups);csvout(r/'RESCUE_HARM.csv',harmrows);csvout(r/'ROUTING_DIAGNOSTICS.csv',routes);csvout(r/'ORACLE_DIAGNOSTICS.csv',oracles)
    from verify import run as verify
    verify(c)
