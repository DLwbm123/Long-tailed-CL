"""Encode exact fixed states, fit S-only heads, seal all, then score TRAIN Q."""
import json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from common import write,guard
from data import train_rows,get_cache,Tokens,store
from folds import assign_folds
from vision import NativeTail,adapt_native_tail,load_delta,delta_state
from boundary import fit,logits

def main(cfg):
 r=Path(cfg['output']);dc=cfg['domains']['HK'];paths=[];rows,labels=train_rows(dc)
 folds=assign_folds(labels.tolist(),[x['identity_component'] for x in rows])
 bundle=torch.load(r/'cache/0/tail.private.pt',map_location='cpu',weights_only=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=NativeTail(bundle).cuda().eval();adapt_native_tail(m);sealed=[];encoded=0
 feasibility=json.loads((r/'FEASIBILITY.private.json').read_text())
 for stage in range(6):
  guard(cfg);write(r/'STATUS.json',dict(phase='ENCODE_AND_FIT',stage=stage))
  paths.extend(get_cache(cfg,dc,stage=stage));data=Tokens(paths);n=sum(map(len,dc['stages'][:stage+1]))
  source=torch.load(Path(cfg['source_run'])/f'HK/uniform_controller/74002/head{stage}.private.pt',map_location='cpu',weights_only=False)
  assert source['seen']==n;assert all(int(v['step'])==24*(stage+1) for v in source['optimizer_state']['state'].values())
  load_delta(m,source['delta']);emb=[]
  with torch.no_grad():
   for lo in range(0,len(data),64):
    guard(cfg);ii=list(range(lo,min(lo+64,len(data))));tokens=store().batch(data,ii,True)
    emb.append(m(tokens.cuda()).cpu());encoded+=len(ii)
  z=torch.cat(emb);assert torch.isfinite(z).all() and torch.allclose(z.norm(dim=1),torch.ones(len(z)),atol=2e-6)
  assert all(torch.equal(v,source['delta'][k]) for k,v in delta_state(m).items())
  assert all(p.grad is None for p in m.parameters())
  torch.save(dict(features=z,labels=data.labels,ids=data.ids,source_delta=source['delta'],source_optimizer_state=source['optimizer_state']),r/f'stage{stage}.private.pt')
  for fold in [0,1]:
   S=[i for i,idx in enumerate(data.ids) if folds[idx]==1-fold];Q=[i for i,idx in enumerate(data.ids) if folds[idx]==fold]
   assert set(S).isdisjoint(Q) and len(S)+len(Q)==len(data)
   ys=data.labels[S];count=torch.bincount(ys,minlength=n);assert (count>0).all()
   sums=torch.zeros(n,512,dtype=torch.double);sums.index_add_(0,ys,z[S].double());proto=F.normalize((sums/count[:,None]).float(),dim=1)
   heads={}
   for arm in ['natural','balanced']:
    guard(cfg);write(r/'STATUS.json',dict(phase='CPU_HEAD_FIT',stage=stage,fold=fold,arm=arm))
    heads[arm]=fit(z[S],ys,proto,arm=='balanced')
   split=next(x for x in feasibility['blocks'] if x['stage']==stage and x['fold']==fold)
   rec=dict(stage=stage,fold=fold,seen=n,S=S,Q=Q,prototypes=proto,heads=heads,
            Qa_ids=split['Qa_ids'],Qb_ids=split['Qb_ids'])
   torch.save(rec,r/f's{stage}f{fold}.private.pt');sealed.append(dict(stage=stage,fold=fold,arms=['NCM','natural','balanced']))
  assert all(torch.cuda.max_memory_reserved(i)<12*2**30 for i in [0,1])
 write(r/'ALL_HEADS_SEALED.json',dict(cells=len(sealed),fitted_heads=24,feature_stages=6,
        new_TRAIN_rows_encoded=encoded,visual_optimizer_steps=0,controller_updates=0,Q_scores=0,VAL_reads=0,TEST_reads=0,time=time.time()))
 output=[];scored=0
 for stage in range(6):
  feature=torch.load(r/f'stage{stage}.private.pt',map_location='cpu',weights_only=False);z=feature['features'];y=feature['labels'];ids=feature['ids']
  for fold in [0,1]:
   guard(cfg);rec=torch.load(r/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False);Q=rec['Q'];n=rec['seen'];results={}
   saved_scores={}
   for arm in ['NCM','natural','balanced']:
    score=logits(z[Q],rec['prototypes'],None if arm=='NCM' else rec['heads'][arm]['residual']);prediction=score.argmax(1)
    saved_scores[arm]=dict(logits=score,prediction=prediction)
    result={}
    for tag,subset in [('full',None),('Qa',set(rec['Qa_ids'])),('Qb',set(rec['Qb_ids']))]:
     mask=torch.ones(len(Q),dtype=torch.bool) if subset is None else torch.tensor([ids[i] in subset for i in Q])
     yy=y[Q][mask];den=torch.bincount(yy,minlength=n);cc=torch.bincount(yy[prediction[mask]==yy],minlength=n)
     result[tag]=dict(counts=den.tolist(),correct=cc.tolist())
    results[arm]=result;scored+=len(Q)
   torch.save(dict(Q=Q,labels=y[Q],ids=[ids[i] for i in Q],arms=saved_scores),r/f'scores_s{stage}f{fold}.private.pt')
   output.append(dict(stage=stage,fold=fold,seen=n,query=results,fit={a:{k:v for k,v in rec['heads'][a].items() if k not in ['residual','optimizer_state']} for a in ['natural','balanced']}))
 write(r/'RECORDS.private.json',output)
 engine=store();write(r/'REGENERABLE_CACHE_CLEANUP.json',dict(encoded_rows=engine.encoded,RAM_bytes=engine.ram_bytes,native_prefix_max_abs=engine.max_abs,root_spill_bytes=engine.spill_bytes,remote_spill_bytes=engine.remote_spill_bytes,science_preserved=True))
 assert engine.spill_bytes==engine.remote_spill_bytes==0
 engine.index.clear();engine.ram_bytes=0
 write(r/'TRAIN_COMPLETE.json',dict(cells=12,fitted_heads=24,new_TRAIN_query_rows_scored=scored,
       visual_optimizer_steps=0,controller_updates=0,VAL_reads=0,TEST_reads=0,time=time.time()))

if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
