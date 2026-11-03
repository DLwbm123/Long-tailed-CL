"""One pre-specified scale calibration at each borrowed SUP0, no fitting or efficacy selection."""
import torch,json,time
from pathlib import Path
from common import write
from prototype_policy import Policy,build_prototypes,gaussian_KL
from constraints import behavior_KL
from episodes import episode

def load_reference(c,s,device='cuda'):
 m=Policy().to(device);z=torch.load(Path(c['init_source'])/f's{s}_SUP0_512.private.pt',map_location='cpu',weights_only=False);assert z['stage']==s and z['step']==512 and all(torch.isfinite(v).all() for v in z['model'].values());m.load_state_dict(z['model'],strict=True);return m.requires_grad_(False)

@torch.no_grad()
def calibrate_episode(m,ep,stage,which):
 raw=m.net(ep['state']);mu=m(ep['state']);p=build_prototypes(mu[None],ep)[0];score=20*ep['query']@p.T;ka=[];kb=[]
 for j in range(32):
  gen=torch.Generator(device=mu.device).manual_seed(74002+9000001*stage+1009*which+7919*71+j);eps=torch.randn(mu.shape,generator=gen,device=mu.device);tilde=2*torch.tanh((raw+.01*eps*ep['active'][:,None])/2);pp=build_prototypes(tilde[None],ep)[0];ss=20*ep['query']@pp.T;ka.append(float(gaussian_KL(tilde,mu,ep['active'])));kb.append(float(behavior_KL(ss,score,ep['y'])))
 return {'episode':'half' if which else 'full','action_KL_sum':sum(ka),'behavior_KL_sum':sum(kb),'directions':32,'queries':len(ep['y']),'classes':len(mu),'class_prototypes':33*len(mu),'raw_score_rows':33*len(ep['y'])}

def run(c,meta):
 from data_contract import load_stage
 start=time.time();cpu=time.process_time();out=[]
 for s in range(6):
  d=load_stage(c,s,meta[1],meta[2]);m=load_reference(c,s);rows=[calibrate_episode(m,episode(d,s,0,half),s,int(half)) for half in [False,True]];a=sum(x['action_KL_sum'] for x in rows);b=sum(x['behavior_KL_sum'] for x in rows);assert b>1e-14 and a>0,'CALIBRATION_NOT_IDENTIFIABLE';k=a/b;assert torch.isfinite(torch.tensor(k)) and 0<k<=1e8,('CALIBRATION_NOT_IDENTIFIABLE',s,k);out.append({'stage':s,'kappa':k,'contexts':rows,'not_optimized_or_VAL_selected':True})
 write(Path(c['output'])/'CALIBRATION.json',{'passed':True,'stages':out,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'formal_optimizer':0,'F_VAL_TEST':0,'scope':'Fixed SUP0, two S/B contexts/stage,32 output perturbations/context; local output-space scale only.'});return [v['kappa'] for v in out]
