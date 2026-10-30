import json,os,time,torch
from pathlib import Path
from common import write,guard
from boundary import fit,logits
from structure import matrices

def counts(pred,y,n,ids,qa,qb):
 out={}
 for tag,subset in [('full',None),('Qa',set(qa)),('Qb',set(qb))]:
  mask=torch.ones(len(y),dtype=torch.bool) if subset is None else torch.tensor([i in subset for i in ids]);yy=y[mask];pp=pred[mask]
  out[tag]=dict(counts=torch.bincount(yy,minlength=n).tolist(),correct=torch.bincount(yy[yy==pp],minlength=n).tolist())
 return out

def main(cfg):
 r=Path(cfg['output']);s=Path(cfg['feature_source_run']);torch.set_num_threads(4);sealed=[];iterations=evaluations=0
 assert json.loads((s/'VERIFICATION.json').read_text())['passed']
 for stage in range(6):
  guard(cfg);feature=torch.load(s/f'stage{stage}.private.pt',map_location='cpu',weights_only=False);n=sum(map(len,cfg['domains']['HK']['stages'][:stage+1]))
  assert set(feature['labels'].tolist())==set(range(n));assert max(feature['labels'])<n
  ms,nodes=matrices(cfg['domains']['HK']['ordered_names'][:n])
  for fold in [0,1]:
   source=torch.load(s/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False);assert source['seen']==n
   S,Q=source['S'],source['Q'];assert not set(S)&set(Q);heads={}
   for arm,L in ms.items():
    guard(cfg);write(r/'STATUS.json',dict(phase='CPU_HEAD_FIT',stage=stage,fold=fold,arm=arm))
    heads[arm]=fit(feature['features'][S],feature['labels'][S],source['prototypes'],True,L)
    iterations+=heads[arm]['iterations'];evaluations+=heads[arm]['closure_evaluations']
   torch.save(dict(stage=stage,fold=fold,seen=n,S=S,Q=Q,source_feature_ids=feature['ids'],matrices=ms,
              nodes=nodes,heads=heads,prototypes=source['prototypes'],Qa_ids=source['Qa_ids'],Qb_ids=source['Qb_ids']),r/f's{stage}f{fold}.private.pt')
   sealed.append((stage,fold))
 write(r/'ALL_HEADS_SEALED.json',dict(cells=12,fitted_heads=36,source_feature_stages=6,
       CPU_head_iterations=iterations,CPU_head_closure_evaluations=evaluations,
       scientific_Q_rows_scored=0,visual_encoding_rows=0,visual_optimizer_steps=0,controller_updates=0,VAL_TEST_reads=0,time=time.time()))
 output=[];new_rows=0
 source_records=json.loads((s/'RECORDS.private.json').read_text())
 for stage,fold in sealed:
  guard(cfg);feature=torch.load(s/f'stage{stage}.private.pt',map_location='cpu',weights_only=False);head=torch.load(r/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
  Q=head['Q'];y=feature['labels'][Q];ids=[feature['ids'][i] for i in Q];n=head['seen'];source=next(v for v in source_records if v['stage']==stage and v['fold']==fold)
  result={a:source['query'][old] for a,old in [('NCM','NCM'),('unstructured','balanced'),('natural','natural')]};saved={}
  for arm,h in head['heads'].items():
   score=logits(feature['features'][Q],head['prototypes'],h['residual']);pred=score.argmax(1);saved[arm]=dict(logits=score,prediction=pred)
   result[arm]=counts(pred,y,n,ids,head['Qa_ids'],head['Qb_ids']);new_rows+=len(Q)
  torch.save(dict(Q=Q,ids=ids,labels=y,arms=saved),r/f'scores_s{stage}f{fold}.private.pt')
  output.append(dict(stage=stage,fold=fold,seen=n,query=result,fit={a:{k:v for k,v in h.items() if k not in ['residual','optimizer_state']} for a,h in head['heads'].items()}))
 write(r/'RECORDS.private.json',output)
 write(r/'TRAIN_COMPLETE.json',dict(cells=12,fitted_heads=36,new_classifier_TRAIN_query_rows_scored=new_rows,
       native_engineering_classifier_rows_scored=2,reference_scoring_rows=0,reference_refits=0,
       visual_encoding_rows=0,visual_optimizer_steps=0,controller_updates=0,VAL_TEST_reads=0,time=time.time()))

if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
