"""New constraint tests only; R98 sampling/CF mathematics remains unchanged."""
import torch,copy,time,io
from common import write
from constraints import behavior_KL,preservation
from prototype_policy import Policy,build_state
from calibration import calibrate_episode
from losses import components

def run(path):
 start=time.time();cpu=time.process_time();torch.manual_seed(991);x=torch.randn(4,3,dtype=torch.double,requires_grad=True);ref=torch.randn(4,3,dtype=torch.double,requires_grad=True);y=torch.tensor([0,1,2,2]);k=behavior_KL(x,ref,y);assert k>=0;k.backward();assert torch.isfinite(x.grad).all() and x.grad.norm()>0 and ref.grad is None
 shift=torch.tensor([2.,-3.,5.,1.])[:,None];err=float((behavior_KL(x.detach()+shift,ref.detach(),y)-k.detach()).abs());assert err<1e-12;assert abs(float(behavior_KL(x.detach(),x.detach(),y)))<1e-14
 ix=torch.tensor([0,1,2,3,2,3]);assert abs(float(behavior_KL(x.detach()[ix],ref.detach()[ix],y[ix])-k.detach()))<1e-12
 perm=torch.tensor([2,0,1]);inv=torch.argsort(perm);assert abs(float(behavior_KL(x.detach()[:,perm],ref.detach()[:,perm],inv[y])-k.detach()))<1e-12
 support=[torch.nn.functional.normalize(torch.randn(n,512),dim=-1) for n in [1,3,5]];ep=build_state(support,[3,6,10],[0]*3,0);ep.update(query=torch.nn.functional.normalize(torch.randn(6,512),dim=-1),y=torch.tensor([0,1,2,0,1,2]),nB=torch.ones(3)*2,protected=torch.tensor([True]*3),tail=torch.tensor([True]*3),kappa=3.);m=Policy();rm=copy.deepcopy(m).requires_grad_(False)
 with torch.no_grad():m.net[-1].bias.add_(.03)
 for kind in ['ACT','BHV','NONE']:
  m.zero_grad();v,_,_,_,st=components('CF_'+kind,m,rm,ep,torch.randn(8,3,4));v.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()) and m.net[-1].weight.grad.norm()>0
 ca=calibrate_episode(rm,ep,0,0);cb=calibrate_episode(rm,ep,0,0);assert ca==cb and ca['behavior_KL_sum']>1e-14 and 0<ca['action_KL_sum']/ca['behavior_KL_sum']<1e8
 before={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=0,weight_decay=0);opt.step();assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items());buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);mm=Policy();mm.load_state_dict(torch.load(buf,weights_only=False),strict=True);assert torch.equal(m(ep['state']),mm(ep['state']))
 write(path,{'passed':True,'behavior_KL_translation_maxabs':err,'zero_identity_positive_finite_gradients':True,'class_balanced_repetition_invariance':True,'class_relabeling_invariance':True,'reference_no_gradient':True,'all_three_constraint_objectives_finite':True,'fixed_calibration_reproducible':True,'LR0_restore':True,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'engineering_optimizer_steps':1,'image_encoder_rows':0,'VAL_TEST':0,'R98_closed_Gaussian_MC_not_repeated':True})
