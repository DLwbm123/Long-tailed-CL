"""All signs, matched mechanism gates and original teacher reference."""
import csv,json
from pathlib import Path
import torch
from common import write
from evaluate import MODES
from model import ARMS
MAIN='PROTO_GRPO'

def metrics(c,p,s):
    y=p['labels'];C=sum(map(len,c['stages'][:s+1]));den=torch.bincount(y,minlength=C);assert (den>0).all()
    tail=torch.tensor(c['train_counts'][:C])<250;old=torch.arange(C)<sum(map(len,c['stages'][:s]));out={}
    for a,pr in p['predictions'].items():
        correct=torch.bincount(y[pr==y],minlength=C);rec=correct.double()/den;groups={}
        for age,am in [('old',old),('new',~old),('all',torch.ones(C,dtype=torch.bool))]:
            for freq,fm in [('head',~tail),('tail',tail),('all',torch.ones(C,dtype=torch.bool))]:
                m=am&fm;groups[age+'_'+freq]=float(rec[m].mean()*100) if m.any() else None
        out[a]={'micro':float(correct.sum()/den.sum()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100),'count':den.tolist(),'correct':correct.tolist(),'recall':(rec*100).tolist(),'groups':groups}
    return {'stage':s,'arms':out}

def summarize(c,records,packs):
    result={'primary':MAIN,'mean':{},'final':{},'comparisons':{}}
    for a in MODES:
        result['mean'][a]={k:sum(x['arms'][a][k] for x in records)/6 for k in ['micro','macro','tail']}
        result['final'][a]={k:records[-1]['arms'][a][k] for k in ['micro','macro','tail']}
    for a in ARMS:
        result['comparisons'][a]={ref:{phase:{k:result[phase][a][k]-result[phase][ref][k] for k in ['micro','macro','tail']} for phase in ['mean','final']} for ref in MODES if ref!=a}
    harms={str(cl):sum(int(((p['labels']==cl)&(p['predictions']['T']==p['labels'])&(p['predictions'][MAIN]!=p['labels'])).sum()) for p in packs) for cl in [1,2,12]}
    d=result['comparisons'][MAIN]['T'];both=sum(x['arms'][MAIN]['macro']>=x['arms']['T']['macro']-1e-8 and x['arms'][MAIN]['tail']>=x['arms']['T']['tail']-1e-8 for x in records)
    teacher=d['mean']['macro']>=.5 and d['final']['macro']>=.5 and d['mean']['tail']>=-1e-8 and d['final']['tail']>=1 and both>=4 and all(v==0 for v in harms.values())
    matched={}
    for ref in ['UNIFORM','CE_WEIGHT']:
        d=result['comparisons'][MAIN][ref]
        matched[ref]=d['mean']['macro']>=.3 and d['final']['macro']>=.3 and d['mean']['tail']>=-1e-8 and d['final']['tail']>=-1e-8
    result.update(teacher_gate=bool(teacher),matched_gates={k:bool(v) for k,v in matched.items()},stage_coverage=both,tiny_teacher_correct_harm=harms,passed=bool(teacher and all(matched.values())))
    result['conclusion']='COMPLETE_JOINT_DEVELOPMENT_GATE_PASS' if result['passed'] else 'COMPLETE_JOINT_MECHANISM_NEGATIVE'
    return result

def csvout(path,header,rows):
    with path.open('w') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)

def run(c):
    r=Path(c['output']);packs=[torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False) for s in range(6)];records=[metrics(c,p,s) for s,p in enumerate(packs)];summary=summarize(c,records,packs)
    write(r/'RESULT_SUMMARY.json',summary);write(r/'STAGE_METRICS.json',records)
    csvout(r/'STAGE_METRICS.csv',['stage','arm','micro','macro','tail'],[[v['stage'],a,*[m[k] for k in ['micro','macro','tail']]] for v in records for a,m in v['arms'].items()])
    csvout(r/'CLASS_STAGE.csv',['stage','arm','class','count','correct','recall'],[[v['stage'],a,k,n,m['correct'][k],m['recall'][k]] for v in records for a,m in v['arms'].items() for k,n in enumerate(m['count'])])
    csvout(r/'GROUP_STAGE.csv',['stage','arm','group','macro'],[[v['stage'],a,g,x] for v in records for a,m in v['arms'].items() for g,x in m['groups'].items()])
    harms=[];contrib=[]
    for s,(p,v) in enumerate(zip(packs,records)):
        y=p['labels'];C=len(v['arms']['T']['count']);nt=sum(x<250 for x in c['train_counts'][:C])
        for a in ARMS:
            for ref in ['T','UNIFORM','CE_WEIGHT','RLPO_STYLE']:
                if a==ref:continue
                for cl,n in enumerate(v['arms'][a]['count']):
                    mask=y==cl;good=p['predictions'][ref]==y;own=p['predictions'][a]==y
                    harms.append([s,a,ref,cl,n,int((mask&good&~own).sum()),int((mask&~good&own).sum())])
                    delta=v['arms'][a]['recall'][cl]-v['arms'][ref]['recall'][cl]
                    contrib.append([s,a,ref,cl,'tiny' if cl in [1,2,12] else 'other',delta/(6*C),delta/(6*nt) if c['train_counts'][cl]<250 else 0])
    csvout(r/'RESCUE_HARM.csv',['stage','arm','reference','class','count','harm','rescue'],harms)
    csvout(r/'CONTRIBUTION_DECOMPOSITION.csv',['stage','arm','reference','class','group','mean_macro_contribution_pp','mean_tail_contribution_pp'],contrib)
    F=[metrics(c,torch.load(r/f'F_predictions_s{s}.private.pt',map_location='cpu',weights_only=False),s) for s in range(6)];write(r/'F_TRANSFER.json',F)
    diag=json.loads((r/'TRAIN_DIAGNOSTICS.json').read_text());keys=list(diag[0]);csvout(r/'POLICY_DIAGNOSTICS.csv',keys,[[x[k] for k in keys] for x in diag])
    lines=['# R101 joint representation / instance policy','',summary['conclusion'],'','|mode|mean macro|final macro|mean tail|final tail|','|---|---:|---:|---:|---:|']
    for a in MODES:lines.append('|'+a+'|'+'|'.join(f'{summary[phase][a][k]:.6f}' for phase,k in [('mean','macro'),('final','macro'),('mean','tail'),('final','tail')])+'|')
    lines += ['', 'All four arms share the adapter, teacher anchor and fixed S evidence. Original T is a full-history strong teacher; A0 is a fresh identity/uniform module readout. Not strict bounded-memory CL, independent patients, multiple seeds or proven top-tier novelty. No posthoc arm/epoch selection. RLPO_STYLE is a transferred geometric auxiliary, not author code or standard GRPO. PROTO_GRPO is a one-update Gaussian group-relative policy gradient with leave-one-out advantages and no group-standard-deviation normalization; sampling uniqueness is a separate stronger claim.', '', 'Main comparisons, all class-stage/group signs, rescue/harm and original-denominator contributions are included. Continuous reward, concentrated weights or training accuracy do not establish classification benefit. Native/profile costs are included in worker, not added twice; borrowed original teacher/feature costs are historical, not free. Preparation/SSH/publication CPU nonzero and not fully sampled. No monitor.']
    (r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return summary
