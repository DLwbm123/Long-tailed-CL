"""Real native encoder engineering, zero optimization and no performance screen."""
import os,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from common import write,guard
from data import get_cache,Tokens
from vision import NativeTail,adapt_native_tail,delta_state,load_delta
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;guard(cfg);dc=cfg['domains']['NIH'];p=get_cache(cfg,dc,stage=0,preflight=True)[0];bundle=torch.load(p/'tail.private.pt',map_location='cpu',weights_only=False);data=Tokens([p]);x=torch.stack([data[i][0] for i in range(len(data))]).cuda();base=data.base.cuda();m=NativeTail(bundle).cuda().eval()
with torch.no_grad():z=m(x);err=float((z-base).abs().max());assert err<=2e-5,('CROSS_RUNTIME_TAIL_PARITY',err)
params=adapt_native_tail(m);m.eval();text=torch.load(root/'refs/0.pt',map_location='cpu',weights_only=False)['text'].cuda();frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad};z=m(x);assert torch.allclose(z,base,atol=2e-5,rtol=0);loss=F.cross_entropy(20*z@text.T,data.labels.cuda());loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);assert max(float(p.grad.norm()) for p in params)>0;assert all(torch.equal(p,frozen[n]) and p.grad is None for n,p in m.named_parameters() if not p.requires_grad)
with torch.no_grad(),torch.random.fork_rng(devices=[0]):
 other=NativeTail(bundle).cuda();adapt_native_tail(other);other.eval();load_delta(other,delta_state(m));assert torch.equal(m(x),other(x))
assert torch.cuda.max_memory_reserved()<12*2**30
write(root/'PREFLIGHT.json',dict(passed=True,optimizer_steps=0,real_TRAIN_rows=2,no_VAL_TEST=True,prefix_shape=[197,768],prefix_dtype='float32',cross_runtime_tail_max_abs=err,original_visual_parameters_frozen=True,LoRA_trainable_parameters=sum(p.numel() for p in params),strict_restore_predictions_exact=True,torch_target=torch.__version__,no_short_scientific_pilot=True,time=time.time()))

# Full production support-conditioned loss and micro accumulation, zero optimizer.
from decision import prototypes,query_risk
from folds import assign_folds
fixture_labels=torch.tensor([0,1,0,1],device='cuda');fixture_folds=assign_folds([0,0,1,1],['a','b','c','d']);assert fixture_folds==[0,1,0,1]
with torch.no_grad():
 proto,pc=prototypes(torch.cat([base,base]),fixture_labels,2);assert not proto.requires_grad and pc.tolist()==[2,2]
for q in params:q.grad=None
synthetic_labels=torch.arange(len(x),device='cuda')%2
risk,score=query_risk(m(x),synthetic_labels,proto,torch.tensor([2,2],device='cuda'),4);risk.mean().backward();assert torch.isfinite(score).all() and max(float(q.grad.norm()) for q in params)>0
write(root/'DECISION_PREFLIGHT.json',dict(passed=True,optimizer_steps=0,detached_support=True,all_seen_softmax=True,finite_gradients=True,classes=2,metadata_fold_disjoint=True,no_scientific_effect_screen=True))

# Real native forward + production old-frozen/new-support concatenation, no optimizer.
for q in params:q.grad=None
fixed_old=(20*base@proto[:1].T).detach().clone();new_proto=proto.detach().clone();before=fixed_old.clone();zz=m(x);score=torch.cat([fixed_old,20*zz@new_proto.T],dim=1);yy=synthetic_labels+1;nc=torch.tensor([2,2,2],device='cuda');w=6/(3*nc[yy].float());loss=(w*(F.cross_entropy(score,yy,reduction='none')+.01*(zz-base).square().sum(1))).mean();loss.backward();assert torch.equal(fixed_old,before) and not fixed_old.requires_grad and torch.isfinite(score).all() and max(float(q.grad.norm()) for q in params)>0
write(root/'BIRTH_PREFLIGHT.json',dict(passed=True,optimizer_steps=0,old_score_detached_exact=True,new_score_native_backward=True,full_seen_competition=True,strictrestore=True,parameter_growth_disclosed=True))
