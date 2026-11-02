import json,torch
from pathlib import Path
from common import write

def run(cfg,counter):
 r=Path(cfg['output']);assert counter['builder_optimizer_steps']==3072 and counter['actor_optimizer_steps']==9216
 assert len(list(r.glob('builder_*_s*.private.pt')))==12 and len(list(r.glob('actor_*_s*_u*.private.pt')))==108 and len(list(r.glob('deploy_s*_u*.private.pt')))==18
 for p in r.glob('builder_*_s*.private.pt'):
  x=torch.load(p,map_location='cpu',weights_only=False);assert x['optimizer']['state'] and all(torch.isfinite(t).all() for t in x['model'].values())
 for p in r.glob('actor_*_s*_u*.private.pt'):
  x=torch.load(p,map_location='cpu',weights_only=False);assert x['optimizer']['state'] and all(torch.isfinite(t).all() for t in x['model'].values())
 rows=json.loads((r/'METRICS.json').read_text());assert len(rows)==6*3*11
 for s in range(6):
  d=torch.load(r/f'deploy_s{s}_u256.private.pt',map_location='cpu',weights_only=False);assert d['p0_exact_SFT'] and all(torch.allclose(v.sum(1),torch.ones(len(v),dtype=torch.float64),atol=1e-12,rtol=0) for v in d['distributions'].values())
 write(r/'VERIFICATION.json',{'passed':True,'builder_states':12,'actor_states':108,'deployment_states':18,'metrics_records':len(rows),'new_model_forward':0,'new_optimizer_steps':0,'new_VAL_TEST_reads':0})
