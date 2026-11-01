import json
from pathlib import Path
import numpy as np
import torch
from common import write

def run(cfg):
 r=Path(cfg['output']);seal=json.loads((r/'ALL_STATES_SEALED.json').read_text());assert seal['policy_snapshots']==30 and seal['dictionary_epochs']==10 and seal['VAL_reads']==0 and seal['TEST_reads']==0
 for epoch in range(1,11):
  z=torch.load(r/f'dictionary_epoch{epoch}.private.pt',map_location='cpu',weights_only=False);assert z['epoch']==epoch and set(z['dictionary'])=={'atoms'};assert torch.isfinite(z['dictionary']['atoms']).all() and z['optimizer']['state']
 for stage in range(6):
  deploy=torch.load(r/f'deployment_s{stage}.private.pt',map_location='cpu',weights_only=False);scores=torch.load(r/f'VAL_scores_s{stage}.private.pt',map_location='cpu',weights_only=False);assert np.isfinite(scores['program_scores']).all();assert len(scores['ids'])==len(set(scores['ids']));assert np.array_equal(scores['program_correct'],np.stack([np.bincount(scores['labels'][s.argmax(1)==scores['labels']],minlength=deploy['proto'].shape[1]) for s in scores['program_scores']]))
  for arm in ['C','E','X','G','H']:
   z=torch.load(r/f's{stage}_{arm}.private.pt',map_location='cpu',weights_only=False);assert z['arm']==arm and z['stage']==stage and len(z['groups'])==32 and all(len(g['losses'])==4 for g in z['groups']);assert z['optimizer']['state'] and all(torch.isfinite(t).all() for t in z['policy'].values());p=np.asarray(deploy['distributions'][arm]);assert abs(p.sum()-1)<1e-12 and (p>=0).all()
 assert seal['counts']['formal_visual_optimizer_steps']==1230 and seal['counts']['formal_controller_optimizer_steps']==3840
 write(r/'VERIFICATION.json',{'passed':True,'dictionary_states':10,'policy_states':30,'new_encoder_classifier_policy_forward':0,'new_optimizer_steps':0,'new_VAL_TEST_reads':0})
