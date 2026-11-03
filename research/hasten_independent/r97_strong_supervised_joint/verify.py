import json,torch
from pathlib import Path
from common import write

def run(c):
 r=Path(c['output']);seal=json.loads((r/'ALL_STATES_SEALED.json').read_text());ev=json.loads((r/'EVAL_COMPLETE.json').read_text());assert seal['VAL_TEST']==0 and ev['all_states_sealed_before_VAL'] and ev['counts']['TEST_rows']==0;expected=6468 if seal['RL_trained'] else 4620;assert seal['counts']['formal_optimizer_steps']==expected and seal['count']==(12 if seal['RL_trained'] else 6)
 for name in seal['snapshots']:
  z=torch.load(r/name,map_location='cpu',weights_only=False);assert z['optimizer']['state'] and all(torch.isfinite(v).all() for v in z['model'].values());assert z['text'].shape==(sum(map(len,c['stages'][:z['stage']+1])),512)
 for s in range(6):
  z=torch.load(r/f'VAL_scores_s{s}.private.pt',map_location='cpu',weights_only=False);n=sum(map(len,c['stages'][:s+1]));assert len(z['ids'])==len(set(z['ids']))==len(z['labels']);assert set(z['scores'])==set(ev['modes'])
  for ss in z['scores'].values():assert ss.shape==(len(z['labels']),n) and torch.isfinite(ss).all()
 write(r/'VERIFICATION.json',{'passed':True,'saved_endpoints':seal['count'],'formal_steps':expected,'all_stages_readout':6,'no_new_encoder_classifier_optimizer_VAL_TEST':True})
