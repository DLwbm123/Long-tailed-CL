"""Independent saved-token reward audit; no model load or new generation."""
import csv,json,re,time
from pathlib import Path
import torch
from .core import write

def run(cfg):
 begin=time.time();cpu=time.process_time();p=Path(cfg['output']);z=torch.load(p/'PROBE_ROLLOUT.private.pt',map_location='cpu',weights_only=True);receipt=json.loads((p/'GRADIENT_PROBE.json').read_text());train=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));ys=[int(r['original_label']) for r in train];names={int(r['original_label']):r['official_class_name'].replace('-',' ') for r in train};normal=lambda x:re.sub(r'[\s_\-]+','',x.lower().strip());ids=[i for i,c in enumerate(ys) if c in range(5,10)];g=torch.Generator().manual_seed(12474006);batch=torch.tensor(ids)[torch.randint(len(ids),(32,8),generator=g)][0].tolist();assert z['labels'].tolist()==[ys[i] for i in batch]
 target=[normal(names[ys[i]]) for i in batch for _ in range(8)];b=torch.tensor([float(normal(t)==y and bool(d)) for t,y,d in zip(z['text'],target,z['done'])]).reshape(8,8);assert torch.equal(b,z['b']);m=z['m'];drift=(((z['old']-z['ref'])*m).sum(-1)/m.sum(-1).clamp_min(1)).clamp_min(0);h=torch.exp(-20*drift).reshape_as(b);reward=b+.5*h;sig=.999*z['sigma_before']+.001*float(reward.std(unbiased=False));assert torch.allclose(h,z['h'],atol=1e-6,rtol=1e-5) and abs(sig-z['sigma_after'])<1e-7
 a=(reward-reward.mean(-1,keepdim=True))/(sig+1e-6);ac=(b-b.mean(-1,keepdim=True))/(sig+1e-6);ar=.5*(h-h.mean(-1,keepdim=True))/(sig+1e-6);assert torch.allclose(a,z['a'],atol=2e-6) and torch.allclose(ac,z['ac'],atol=2e-6) and torch.allclose(ar,z['ar'],atol=2e-6)
 assert int(b.sum())==receipt['correct_responses']==40 and receipt['counterfactual_discarded_actor_anchor_restored'];assert int(((b.sum(-1)>0)&(b.sum(-1)<8)).sum())==4
 proof={'passed':True,'responses':64,'natural_query_count':8,'correct_responses':int(b.sum()),'mixed_queries':4,'saved_text_done_accuracy_reward_drift_and_CTAN_independently_recomputed':True,'new_generation_model_encoder_optimizer_calls':0,'measured_gradient_scalars_not_independently_recomputed_without_forward':True,'CPU_body_seconds':time.process_time()-cpu,'wall_body_seconds':time.time()-begin};write(p/'PROBE_SAVED_VERIFICATION.json',proof)
