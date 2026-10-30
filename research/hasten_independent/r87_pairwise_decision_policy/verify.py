"""Saved tensors and arithmetic only; no policy/encoder/classifier forward or update."""
import os,json,torch,time
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);torch.set_num_threads(4);seal=json.loads((r/'ALL_POLICIES_SEALED.json').read_text());assert seal['policy_snapshots']==288 and seal['formal_controller_optimizer_steps']==36864 and seal['heldout_labels_scored']==0
snap=list(r.glob('policy_s*.private.pt'));assert len(snap)==288;groups=0;steps=0
for f in snap:
 v=torch.load(f,map_location='cpu',weights_only=False);fit=v['fit'];assert fit['steps']==128 and len(fit['history'])==32;steps+=fit['steps'];groups+=len(fit['history']);assert all(torch.isfinite(x).all() for x in v['model_state'].values())
 for state in v['optimizer_state']['state'].values():assert int(state['step'])==128*(v['stage']+1)
 if v['arm'] in ['G','H']:
  ctx=torch.load(r/v['training_context'],map_location='cpu',weights_only=False)
  for h in fit['history']:
   a=torch.tensor(h['a']);b=torch.tensor(h['b']);assert ctx['mask'][a,b].all();reward=ctx['outcome']['reward'][a,b];expected=reward.roll(1) if v['arm']=='H' else reward;assert torch.allclose(expected,torch.tensor(h['assigned_reward'],dtype=torch.double),atol=1e-12,rtol=0);adv=(expected-expected.mean())/(expected.std(unbiased=False)+1e-8);assert torch.allclose(adv,torch.tensor(h['advantage'],dtype=torch.double),atol=1e-12,rtol=0)
records=json.loads((r/'RECORDS.private.json').read_text());assert len(records)==144
for v in records:
 tag=v['heldout_tag'] if v['query']=='heldout' else 'full';d=torch.load(r/f"eval_s{v['stage']}f{v['fold']}d{v['direction']}_{tag}.private.pt",map_location='cpu',weights_only=False);dist=torch.load(r/f"distributions_s{v['stage']}f{v['fold']}d{v['direction']}r{v['stream']}_{tag}.private.pt",map_location='cpu',weights_only=False)
 if v['query']=='heldout':
  tr=torch.load(r/f"context_s{v['stage']}f{v['fold']}d{v['direction']}.private.pt",map_location='cpu',weights_only=False);assert set(tr['ids']).isdisjoint(d['ids'])
 o=d['outcome']
 for arm,p in dist.items():
  j=p['p1'][:,None]*p['p2'];assert torch.allclose(j,p['joint'],atol=1e-12,rtol=0) and abs(float(j.sum())-1)<1e-12;assert not j[~d['mask']].any();c=o['base_correct']+(j.sum(1)[:,None]*o['delta_correct']).sum(0)+(j.sum(0)[:,None]*o['delta_correct']).sum(0);assert (c>=-1e-7).all() and (c<=o['count']+1e-7).all();assert torch.allclose(c,torch.tensor(v['arms'][arm]['expected_correct'],dtype=torch.double),atol=1e-8,rtol=0)
  n=len(j);k=int(j.argmax());a,b=k//n,k%n;cc=o['base_correct']+o['delta_correct'][a]+o['delta_correct'][b];assert torch.equal(cc,torch.tensor(v['arms'][arm]['modal_correct'],dtype=torch.double));elig=o['eligible'];tail=torch.tensor([c for c in elig.tolist() if cfg['train_counts'][c]<250]);macro=float((c[elig]/o['count'][elig]).mean()*100);tv=float((c[tail]/o['count'][tail]).mean()*100);assert abs(macro-v['arms'][arm]['expected_macro'])<1e-8 and abs(tv-v['arms'][arm]['expected_tail'])<1e-8
assert steps==36864
write(r/'VERIFICATION.json',dict(passed=True,policy_snapshots=288,formal_controller_optimizer_steps=steps,group_records=groups,heldout_training_ids_disjoint=True,saved_trajectory_probabilities_rewards_and_counts_exact=True,optimizer_carry_steps_exact=True,new_policy_encoder_classifier_forward_rows=0,new_optimizer_steps=0,new_VAL_TEST_reads=0,time=time.time()))
