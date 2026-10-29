"""TRAIN-metadata-only counterfactual action audit; never scores or trains a model."""
import os,json,time,resource
from pathlib import Path
import torch
from common import write
from data import train_rows
from folds import assign_folds
from decision import legal_actions,allocation
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);dc=cfg['domains']['HK'];start=time.time();torch.set_num_threads(4);rows,labels=train_rows(dc);ctx=json.loads((r/'HK/CONTEXT_RECORDS.json').read_text());results={a:dict(same=0,total=0,retained_same=0,retained_total=0,uniform_tail=0,actual_tail=0) for a in ['macro_controller','protected_controller','shuffled_controller']};ids=[];detail=[]
for stage in range(6):
 ids.extend(i for i,y in enumerate(labels.tolist()) if y in dc['stages'][stage]);yy=labels[ids];folds=assign_folds(yy.tolist(),[rows[i]['identity_component'] for i in ids]);parts=[[i for i,f in enumerate(folds) if f==b] for b in [0,1]]
 for context in range(8):
  support=parts[1-context%2];seed=74002+10000*stage+100*context;uniform=[]
  for branch in range(8):
   aa=[]
   for h in range(2):
    g=torch.Generator().manual_seed(seed+h);common=[support[i] for i in torch.randperm(len(support),generator=g).tolist()[:192]];mask=legal_actions(yy,support,common);p=mask.float()/mask.sum();cdf=p.cumsum(0);cdf/=cdf[-1].clone();u=torch.rand((),generator=torch.Generator().manual_seed(seed+700000+branch*10+h));a=int(torch.searchsorted(cdf,u));aa.append(a)
   uniform.append(aa)
  for arm in results:
   v=next(v for v in ctx if v['arm']==arm and v['stage']==stage and v['context']==context);q=results[arm];retained=(stage*8+context)%8
   for b,(actual,rand) in enumerate(zip(v['branches'][:8],uniform)):
    for a,z in zip(actual['actions'],rand):
     q['same']+=a==z;q['total']+=1;q['actual_tail']+=a>0 and dc['train_counts_ordered'][a-1]<250;q['uniform_tail']+=z>0 and dc['train_counts_ordered'][z-1]<250
     if b==retained:q['retained_same']+=a==z;q['retained_total']+=1
   detail.append(dict(arm=arm,stage=stage,context=context,uniform_actions=uniform,actual_actions=[v['actions'] for v in v['branches'][:8]],fixed_retained_index=retained))
for q in results.values():q['same_fraction']=q['same']/q['total'];q['retained_same_fraction']=q['retained_same']/q['retained_total']
x=dict(summary=results,scope='Same seeds, legal action masks and sampling uniforms reconstructed from TRAIN metadata only',not_uniform_controller_training_performance=True,new_model_scores=0,new_optimizer_steps=0,wall_seconds=time.time()-start,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)
write(r/'results/UNIFORM_ACTION_AUDIT.json',x);print(json.dumps(x))
