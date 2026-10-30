import json,os,time,torch
from pathlib import Path
import torch.nn.functional as F
from common import write,guard
from data import train_rows,get_cache,Tokens,store
from vision import NativeTail,adapt_native_tail,load_delta
from pooling import pool,channels
from boundary import fit,logits
from outcome import counts
from visibility import capture

def main(cfg):
 r=Path(cfg['output']);s=Path(cfg['feature_source_run']);dc=cfg['domains']['HK'];paths=[];sealed=[];encoded=0;max_reference_error=0.;iterations=evaluations=0
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert json.loads((s/'VERIFICATION.json').read_text())['passed']
 bundle=torch.load(r/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);m=NativeTail(bundle).cuda().eval();adapt_native_tail(m)
 for stage in range(6):
  guard(cfg);write(r/'STATUS.json',dict(phase='ENCODE',stage=stage,encoded_rows=encoded))
  paths.extend(get_cache(cfg,dc,stage=stage));data=Tokens(paths);n=sum(map(len,dc['stages'][:stage+1]));assert set(data.labels.tolist())==set(range(n))
  source=torch.load(Path(cfg['source_run'])/f'HK/uniform_controller/74002/head{stage}.private.pt',map_location='cpu',weights_only=False)
  assert source['seen']==n and all(int(v['step'])==24*(stage+1) for v in source['optimizer_state']['state'].values())
  for name,p in m.named_parameters():p.requires_grad_(name in source['delta'])
  load_delta(m,source['delta']);m.requires_grad_(False);cls=[];patch=[]
  with torch.no_grad():
   for lo in range(0,len(data),64):
    guard(cfg);ii=list(range(lo,min(lo+64,len(data))));tokens=store().batch(data,ii,True);a,b=pool(m,tokens.cuda());cls.append(a.cpu());patch.append(b.cpu());encoded+=len(ii)
    if stage==0 and lo==0:capture(r,'ENCODING')
  cls=torch.cat(cls);patch=torch.cat(patch)
  reference=torch.load(s/f'stage{stage}.private.pt',map_location='cpu',weights_only=False)
  assert data.ids==reference['ids'] and torch.equal(data.labels,reference['labels'])
  error=float((cls-reference['features']).abs().max());assert error<=2e-5;max_reference_error=max(max_reference_error,error)
  assert all(torch.equal(dict(m.named_parameters())[k].detach().cpu(),v) for k,v in source['delta'].items())
  assert all(p.grad is None and not p.requires_grad for p in m.parameters())
  torch.save(dict(cls=cls,patch=patch,labels=data.labels,ids=data.ids,source_delta=source['delta'],source_optimizer_state=source['optimizer_state'],CLS_reference_max_abs=error),r/f'stage{stage}.private.pt')
  zz=channels(cls,patch);torch.set_num_threads(4)
  for fold in [0,1]:
   src=torch.load(s/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False);assert src['seen']==n
   S,Q=src['S'],src['Q'];assert set(S).isdisjoint(Q) and len(S)+len(Q)==len(data)
   y=data.labels[S];nc=torch.bincount(y,minlength=n);assert (nc>0).all();heads={};prototypes={}
   for arm,z in zz.items():
    sums=torch.zeros(n,1024,dtype=torch.double);sums.index_add_(0,y,z[S].double());p=F.normalize((sums/nc[:,None]).float(),dim=1);prototypes[arm]=p
    guard(cfg);write(r/'STATUS.json',dict(phase='CPU_HEAD_FIT',stage=stage,fold=fold,arm=arm))
    heads[arm]=fit(z[S],y,p,True);iterations+=heads[arm]['iterations'];evaluations+=heads[arm]['closure_evaluations']
   torch.save(dict(stage=stage,fold=fold,seen=n,S=S,Q=Q,heads=heads,prototypes=prototypes,Qa_ids=src['Qa_ids'],Qb_ids=src['Qb_ids']),r/f's{stage}f{fold}.private.pt');sealed.append((stage,fold))
  assert all(torch.cuda.max_memory_reserved(i)<12*2**30 for i in [0])
 write(r/'ALL_HEADS_SEALED.json',dict(cells=12,fitted_heads=24,feature_stages=6,new_TRAIN_rows_encoded=encoded,CPU_head_iterations=iterations,CPU_head_closure_evaluations=evaluations,CLS_reference_max_abs=max_reference_error,visual_optimizer_steps=0,controller_updates=0,scientific_Q_rows_scored=0,VAL_TEST_reads=0,time=time.time()))
 source_records=json.loads((s/'RECORDS.private.json').read_text());output=[];new_rows=0
 for stage,fold in sealed:
  guard(cfg);feature=torch.load(r/f'stage{stage}.private.pt',map_location='cpu',weights_only=False);head=torch.load(r/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
  Q=head['Q'];y=feature['labels'][Q];ids=[feature['ids'][i] for i in Q];n=head['seen'];zz=channels(feature['cls'],feature['patch']);source=next(v for v in source_records if v['stage']==stage and v['fold']==fold)
  result={a:source['query'][old] for a,old in [('NCM','NCM'),('CLS','balanced'),('natural','natural')]};saved={}
  for arm,h in head['heads'].items():
   score=logits(zz[arm][Q],head['prototypes'][arm],h['residual']);pred=score.argmax(1);saved[arm]=dict(logits=score,prediction=pred);result[arm]=counts(pred,y,n,ids,head['Qa_ids'],head['Qb_ids']);new_rows+=len(Q)
  torch.save(dict(Q=Q,ids=ids,labels=y,arms=saved),r/f'scores_s{stage}f{fold}.private.pt')
  output.append(dict(stage=stage,fold=fold,seen=n,query=result,fit={a:{k:v for k,v in h.items() if k not in ['residual','optimizer_state']} for a,h in head['heads'].items()}))
 write(r/'RECORDS.private.json',output)
 engine=store();assert engine.spill_bytes==engine.remote_spill_bytes==0
 write(r/'REGENERABLE_CACHE_CLEANUP.json',dict(encoded_unique_prefix_rows=engine.encoded,RAM_bytes=engine.ram_bytes,prefix_max_abs=engine.max_abs,root_spill_bytes=engine.spill_bytes,remote_spill_bytes=engine.remote_spill_bytes,science_preserved=True));engine.index.clear();engine.ram_bytes=0
 write(r/'TRAIN_COMPLETE.json',dict(cells=12,fitted_heads=24,new_TRAIN_encoding_rows=encoded,new_classifier_TRAIN_query_rows_scored=new_rows,reference_refits=0,reference_scoring_rows=0,visual_optimizer_steps=0,controller_updates=0,VAL_TEST_reads=0,time=time.time()))

if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
