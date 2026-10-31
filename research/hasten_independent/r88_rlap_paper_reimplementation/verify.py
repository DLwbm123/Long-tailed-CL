import json,torch
from pathlib import Path
from common import write

def run(cfg):
 r=Path(cfg['output']);seal=json.loads((r/'ALL_HEADS_SEALED.json').read_text());assert seal['count']==18 and seal['VAL_reads']==0
 checks=0
 for s in range(6):
  n=sum(map(len,cfg['stages'][:s+1]));v=torch.load(r/f'VAL_scores_s{s}.private.pt',weights_only=False,map_location='cpu');y=v['labels'];cnt=torch.bincount(y,minlength=n);assert y.max()<n and len(v['ids'])==len(set(v['ids']))==len(y)
  for a,scores in v['scores'].items():assert scores.shape==(len(y),n) and torch.isfinite(scores).all()
  for a in ['FULL','CE_WEIGHT','UNIFORM']:
   p=torch.load(r/f's{s}_{a}.private.pt',weights_only=False,map_location='cpu');assert p['stage']==s and p['arm']==a;assert len(p['bank_ids'])==len(set(p['bank_ids']))==len(p['bank']);assert (torch.bincount(p['bank_labels'],minlength=n)>0).all();assert set(p['model']).isdisjoint({'visual.conv1.weight','token_embedding.weight'});assert p['optimizer']['state'];assert all(torch.isfinite(x).all() for x in p['model'].values());checks+=1
 write(r/'VERIFICATION.json',{'passed':True,'saved_endpoints':checks,'new_encoder_classifier_policy_forward':0,'new_optimizer_steps':0,'new_VAL_TEST_reads':0})
