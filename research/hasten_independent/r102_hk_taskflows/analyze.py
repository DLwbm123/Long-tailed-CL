"""Five-stage paired readouts; empty groups are null and no imported success gate."""
import csv,json
from pathlib import Path
import torch
from common import write
from evaluate import MODES
from model import ARMS

def csvout(path,header,rows):
    with path.open('w') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)

def metrics(c,p,s):
    y=p['labels'];C=sum(map(len,c['stages'][:s+1]));den=torch.bincount(y,minlength=C);assert (den>0).all()
    tail=torch.tensor(c['train_counts'][:C])<250;old=torch.arange(C)<sum(map(len,c['stages'][:s]));out={}
    for a,pr in p['predictions'].items():
        correct=torch.bincount(y[pr==y],minlength=C);rec=correct.double()/den;groups={}
        for age,am in [('old',old),('new',~old),('all',torch.ones(C,dtype=torch.bool))]:
            for freq,fm in [('head',~tail),('tail',tail),('all',torch.ones(C,dtype=torch.bool))]:
                m=am&fm;groups[age+'_'+freq]=float(rec[m].mean()*100) if m.any() else None
        out[a]={'micro':float(correct.double().sum()/den.sum()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100) if tail.any() else None,'count':den.tolist(),'correct':correct.tolist(),'recall':(rec*100).tolist(),'groups':groups}
    return {'stage':s,'arms':out}

def summarize(c,records,packs):
    result={'primary':'PROTO_GRPO','stream':c['stream_name'],'mean':{},'final':{},'comparisons':{},'mean_valid_stages':{}}
    for a in MODES:
        result['mean'][a]={};result['mean_valid_stages'][a]={}
        for k in ['micro','macro','tail']:
            values=[v['arms'][a][k] for v in records if v['arms'][a][k] is not None]
            result['mean'][a][k]=sum(values)/len(values) if values else None;result['mean_valid_stages'][a][k]=len(values)
        result['final'][a]={k:records[-1]['arms'][a][k] for k in ['micro','macro','tail']}
    for a in MODES:
        result['comparisons'][a]={b:{phase:{k:result[phase][a][k]-result[phase][b][k] if result[phase][a][k] is not None and result[phase][b][k] is not None else None for k in ['micro','macro','tail']} for phase in ['mean','final']} for b in MODES if b!=a}
    result['tiny']={}
    for a in MODES:
        result['tiny'][a]=[{'stage':s,'legacy_index':c['legacy_indices'][cl],'original_label':c['order'][cl],'new_classifier_index':cl,'correct':records[s]['arms'][a]['correct'][cl],'count':records[s]['arms'][a]['count'][cl]} for s in range(5) for cl in c['tiny_indices'] if cl<len(records[s]['arms'][a]['count'])]
    result.update(conclusion='COMPLETE_EXPLORATORY_TASKFLOW_TRIAL',no_R101_six_stage_gate_applied=True,not_strict_bounded_memory=True,no_significance_or_order_causality_claim=True,TEST=0)
    return result

def run(c):
    r=Path(c['output']);packs=[torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False) for s in range(5)];records=[metrics(c,p,s) for s,p in enumerate(packs)];summary=summarize(c,records,packs)
    write(r/'RESULT_SUMMARY.json',summary);write(r/'STAGE_METRICS.json',records)
    csvout(r/'STAGE_METRICS.csv',['stage','arm','micro','macro','tail'],[[v['stage'],a,*[m[k] for k in ['micro','macro','tail']]] for v in records for a,m in v['arms'].items()])
    csvout(r/'CLASS_STAGE.csv',['stage','arm','classifier_index','legacy_index','original_label','tiny','count','correct','recall'],[[v['stage'],a,k,c['legacy_indices'][k],c['order'][k],k in c['tiny_indices'],n,m['correct'][k],m['recall'][k]] for v in records for a,m in v['arms'].items() for k,n in enumerate(m['count'])])
    csvout(r/'GROUP_STAGE.csv',['stage','arm','group','macro'],[[v['stage'],a,g,x] for v in records for a,m in v['arms'].items() for g,x in m['groups'].items()])
    harms=[];contrib=[]
    for s,(p,v) in enumerate(zip(packs,records)):
        y=p['labels'];C=len(v['arms']['T']['count']);nt=sum(x<250 for x in c['train_counts'][:C])
        for a in ARMS:
            for ref in ['T','UNIFORM','CE_WEIGHT','RLPO_STYLE']:
                if a==ref:continue
                for cl,n in enumerate(v['arms'][a]['count']):
                    mask=y==cl;good=p['predictions'][ref]==y;own=p['predictions'][a]==y;harms.append([s,a,ref,cl,c['legacy_indices'][cl],n,int((mask&good&~own).sum()),int((mask&~good&own).sum())])
                    delta=v['arms'][a]['recall'][cl]-v['arms'][ref]['recall'][cl];contrib.append([s,a,ref,cl,c['legacy_indices'][cl],'tiny' if cl in c['tiny_indices'] else 'other',delta/(5*C),delta/(summary['mean_valid_stages'][a]['tail']*nt) if nt and c['train_counts'][cl]<250 else None])
    csvout(r/'RESCUE_HARM.csv',['stage','arm','reference','classifier_index','legacy_index','count','harm','rescue'],harms)
    csvout(r/'CONTRIBUTION_DECOMPOSITION.csv',['stage','arm','reference','classifier_index','legacy_index','group','mean_macro_contribution_pp','mean_valid_tail_contribution_pp'],contrib)
    F=[metrics(c,torch.load(r/f'F_predictions_s{s}.private.pt',map_location='cpu',weights_only=False),s) for s in range(5)]
    write(r/'F_METRICS_AGGREGATE.json',[{'stage':v['stage'],'arms':{a:{k:m[k] for k in ['micro','macro','tail','groups']} for a,m in v['arms'].items()}} for v in F])
    diag=json.loads((r/'TRAIN_DIAGNOSTICS.json').read_text());keys=list(diag[0]);csvout(r/'POLICY_DIAGNOSTICS.csv',keys,[[x[k] for k in keys] for x in diag])
    lines=['# '+c['stream_name'],'','新任务流下的教师条件机制试跑，非严格有限记忆 LT-CIL。','', '|mode|mean macro|final macro|final tail|tail mean denominator|','|---|---:|---:|---:|---:|']
    for a in MODES:lines.append(f"|{a}|{summary['mean'][a]['macro']:.6f}|{summary['final'][a]['macro']:.6f}|{summary['final'][a]['tail']:.6f}|{summary['mean_valid_stages'][a]['tail']}|")
    lines+=['','Full teachers are freshly trained on arrived classes only; old HK LoRA/heads/features never reused. All40 planned learning endpoints across both streams sealed before F/VAL. No winner/epoch/order selection, no TEST. Single train seed/order trial, not independent patients or significance. Original six-stage66.84 is not subtracted as new-method gain. No inherited R101 success gate. Costs include teachers, feature extraction and module training, not free borrowed strong teachers. Complete paired deltas/class/old-new/head-tail/tiny counts are supplied; empty groups null.']
    (r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return summary
