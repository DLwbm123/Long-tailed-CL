"""All 70 classifier endpoints sealed before one generic six-view VAL encoding per flow."""
import csv,json,gc
from pathlib import Path
import torch
from common import write,save,guard
from memory import from_views
from learner import Learner
from constants import ARMS

def csvout(path,rows):
    if rows:
        with Path(path).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def run(c,encoder,counter,z=None):
    r=Path(c['output']);seal=json.loads((Path(c['root_output'])/'GLOBAL_LEARNING_SEAL.json').read_text());assert seal['classifier_endpoints']==180
    rows=list(csv.DictReader((Path(c['manifest'])/'val.csv').open()));assert len(rows)==1698;y=torch.tensor([c['order'].index(int(x['original_label'])) for x in rows],device='cuda');ids=list(range(len(rows)));z=encoder.encode(c,rows,ids,counter,'VAL') if z is None else z;bank=from_views(z,torch.arange(1698,device='cuda'),y);X,_,_=bank.bags();del z,bank;stage_metrics=[];classrows=[];group=[];harm=[];birth=torch.tensor([t for t,cl in enumerate(c['stages']) for _ in cl],device='cuda')
    for t in range(5):
        C=sum(map(len,c['stages'][:t+1]));mask=y<C;yy=y[mask];xx=X[mask];den=torch.bincount(yy,minlength=C);assert (den>0).all();tail=torch.tensor(c['train_counts'][:C],device='cuda')<250;old=birth[:C]<t;pred={};metrics={}
        for arm in ARMS:
            guard(c);m=Learner(c['seed']).cuda();pack=torch.load(r/f's{t}'/f'{arm}_MODEL.private.pt',weights_only=False);m.load_state_dict(pack['model']);m.C=C;m.eval().requires_grad_(False)
            with torch.no_grad():pred[arm]=m(xx).argmax(-1)
            counter['VAL_classifier_rows']=counter.get('VAL_classifier_rows',0)+len(xx);good=pred[arm]==yy;cc=torch.bincount(yy[good],minlength=C);rec=cc.double()/den
            metrics[arm]={'micro':float(good.double().mean()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100) if tail.any() else None,'old_tail':float(rec[tail&old].mean()*100) if (tail&old).any() else None}
            for cl in range(C):classrows.append({'stage':t,'arm':arm,'class':cl,'original_label':c['order'][cl],'legacy_index':c['legacy_indices'][cl],'tiny':cl in c['tiny_indices'],'count':int(den[cl]),'correct':int(cc[cl]),'recall':float(rec[cl]*100)})
            for age,am in [('old',old),('new',~old),('all',torch.ones_like(old))]:
                for freq,fm in [('head',~tail),('tail',tail),('all',torch.ones_like(tail))]:
                    mm=am&fm;group.append({'stage':t,'arm':arm,'group':age+'_'+freq,'class_count':int(mm.sum()),'macro':float(rec[mm].mean()*100) if mm.any() else None})
            del m,pack
        for arm in ARMS:
            for cl in range(C):
                mm=yy==cl;baseline=pred['BASE_ZERO']==yy;correct=pred[arm]==yy;rescue=int((mm&~baseline&correct).sum());damage=int((mm&baseline&~correct).sum());harm.append({'stage':t,'arm':arm,'class':cl,'legacy_index':c['legacy_indices'][cl],'count':int(den[cl]),'rescue':rescue,'harm':damage,'macro_contribution_pp':(rescue-damage)/int(den[cl])/C*100})
        save(r/f'VAL_s{t}.private.pt',{'labels':yy.cpu(),'predictions':{k:v.cpu() for k,v in pred.items()},'global_seal_before_read':True});stage_metrics.append({'stage':t,'modes':metrics});gc.collect();torch.cuda.empty_cache()
    summary={'stream':c['stream_name'],'primary':'CGMR_MAIN','final':stage_metrics[-1]['modes'],'mean':{},'valid_stages':{},'comparisons':{},'TEST':0}
    for arm in ARMS:
        summary['mean'][arm]={};summary['valid_stages'][arm]={}
        for k in ['micro','macro','tail','old_tail']:
            a=[s['modes'][arm][k] for s in stage_metrics if s['modes'][arm][k] is not None];summary['mean'][arm][k]=sum(a)/len(a) if a else None;summary['valid_stages'][arm][k]=len(a)
    for ref in ARMS:
        if ref!='CGMR_MAIN':summary['comparisons'][ref]={ph:{k:summary[ph]['CGMR_MAIN'][k]-summary[ph][ref][k] if summary[ph]['CGMR_MAIN'][k] is not None and summary[ph][ref][k] is not None else None for k in ['micro','macro','tail','old_tail']} for ph in ['mean','final']}
    write(r/'RESULT_SUMMARY.json',summary);write(r/'STAGE_METRICS.json',stage_metrics);csvout(r/'STAGE_METRICS.csv',[dict(stage=s['stage'],arm=a,**v) for s in stage_metrics for a,v in s['modes'].items()]);csvout(r/'CLASS_STAGE.csv',classrows);csvout(r/'GROUP_STAGE.csv',group);csvout(r/'RESCUE_HARM.csv',harm)
    from verify import run as verify
    verify(c)
