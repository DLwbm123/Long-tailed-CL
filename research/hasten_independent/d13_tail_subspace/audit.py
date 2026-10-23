"""CPU-only audit; private inputs stay on host, exports contain class aggregates only."""
import os,json,time,math
from pathlib import Path
import torch
ROOT=Path(os.environ['AUDIT_ROOT']);OUT=Path(os.environ['AUDIT_OUT']);OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
START=time.time();SEEDS=[74002,74003,74004]
seal=json.loads((ROOT/'SPLIT_SEAL.json').read_text());counts=seal['train_counts'];tail=[i for i,n in enumerate(counts) if n<250]
order=seal['original_order'];types0=[0,2,0,0,1,0,0,3,0,0,0,1,0,1,3,3,3,2,2,4];types=[types0[c] for c in order]
names0=['atelectasis','cardiomegaly','consolidation','edema','effusion','emphysema','fibrosis','hernia','infiltration','mass','nodule','pleural thickening','pneumonia','pneumothorax','pneumoperitoneum','pneumomediastinum','subcutaneous emphysema','tortuous aorta','calcification of the aorta','no finding']
arrival={c:s for s,cs in enumerate(seal['stages']) for c in cs};seen=[sum(map(len,seal['stages'][:s+1])) for s in range(6)]
aggregate=json.loads((ROOT/'aggregate.json').read_text())['results'];expected={(r['seed'],r['arm'],r['stage']):r for r in aggregate}
def save(name,obj):
 assert time.time()-START<1770
 p=OUT/name;p.write_text(json.dumps(obj,indent=2,allow_nan=False))
 assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<200_000_000

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def affine(state):
 n=state['network'];a=torch.zeros(512,512,dtype=torch.double);b=torch.zeros(512,dtype=torch.double)
 for j in range(state['task']+1):
  key=f'Image_Adapter.{j}.';w=n[key+'fc.0.weight'].double();bias=n[key+'fc.0.bias'].double();alpha=n[key+'log_alpha'].double().clamp(max=0).exp()
  a+=alpha*w;b+=alpha*bias
 return a,b,n['learned_lorentz_exp.exp_map.0.weight'].double()

def selfcheck(state):
 a,b,w=affine(state);x=torch.arange(3*512,dtype=torch.double).reshape(3,512)/1536
 direct=[];n=state['network']
 for j in range(state['task']+1):
  k=f'Image_Adapter.{j}.';direct.append((x@n[k+'fc.0.weight'].double().T+n[k+'fc.0.bias'].double())*n[k+'log_alpha'].double().clamp(max=0).exp())
 error=float((x@a.T+b-torch.stack(direct).sum(0)).abs().max());assert error<1e-12
 return error

trajectories=[];branches=[];allresponses={}
for arm in ['H-raw','H-unit','H-unit-no-proj']:
 for seed in SEEDS:
  rs=[]
  for stage in range(6):
   r=load(ROOT/f'trajectories/{seed}/{arm}/stage{stage}_responses.private.pt');y=r['labels'].long();ss=r['scores'];n=seen[stage]
   assert len(y)==n*15 and all(v.shape==(len(y),n) and torch.isfinite(v).all() for v in ss.values())
   pred={k:v.argmax(1) for k,v in ss.items()};ok={k:v.eq(y) for k,v in pred.items()}
   for k in ss:
    cc=torch.bincount(y[ok[k]],minlength=n).tolist();assert cc==expected[(seed,arm,stage)]['metrics'][k]['correct']
   for c in range(n):
    mask=y==c;birthn=seen[arrival[c]]
    rec=dict(seed=seed,arm=arm,stage=stage,class_index=c,original_id=order[c],name=names0[order[c]],train_count=counts[c],tail=c in tail,arrival=arrival[c],denominator=15)
    for k in ss:
     rec[k+'_correct']=int(ok[k][mask].sum());rec[k+'_arrival_candidates_correct']=int((ss[k][mask,:birthn].argmax(1)==c).sum())
     if stage and c<seen[stage-1]:rec[k+'_previous_candidates_correct']=int((ss[k][mask,:seen[stage-1]].argmax(1)==c).sum())
    trajectories.append(rec)
    p,h,f=ok['prototype'][mask],ok['hyper'][mask],ok['final'][mask]
    branches.append(dict(seed=seed,arm=arm,stage=stage,class_index=c,tail=c in tail,prototype_correct=int(p.sum()),hyper_correct=int(h.sum()),final_correct=int(f.sum()),prototype_to_fusion_rescue=int((~p&f).sum()),prototype_to_fusion_harm=int((p&~f).sum()),hyper_to_fusion_rescue=int((~h&f).sum()),hyper_to_fusion_harm=int((h&~f).sum()),both_wrong=int((~p&~h).sum()),only_hyper_correct=int((~p&h).sum()),only_prototype_correct=int((p&~h).sum())))
   rs.append(r)
  if arm=='H-unit':allresponses[seed]=rs
save('TRAJECTORIES.json',trajectories);save('BRANCH_TRANSITIONS.json',branches)
spectra=[];coverage=[];drifts=[];checks=[]
for seed in SEEDS:
 prev=None
 for stage in range(6):
  s=load(ROOT/f'trajectories/{seed}/H-unit/stage{stage}.private.pt');n=s['total'];assert n==seen[stage]
  a,b,w=affine(s);checks.append(selfcheck(s));C=s['accumulated_cov'].double();B=s['null_space_basis'].double()
  _,sv,v=torch.svd(C);rank=512-B.shape[1];U=v[:,:rank]
  assert float((B.T@U).abs().max())<1e-5
  frac=sv.square()/sv.square().sum();lin=sv/sv.sum()
  spectra.append(dict(seed=seed,stage=stage,protected_rank=rank,update_rank=B.shape[1],singular_values=sv.tolist(),top_squared_fraction=float(frac[0]),top_linear_fraction=float(lin[0]),squared_95_rank=int(torch.searchsorted(frac.cumsum(0),.95))+1,linear_95_rank=int(torch.searchsorted(lin.cumsum(0),.95))+1,used_for_next_stage=stage<5))
  means=torch.stack([s['prototype'][c].double() for c in range(n)]);z=means@a.T+b
  for c in range(n):
   distances=(z-z[c]).square().sum(1);distances[c]=float('inf');nearest=int(distances.argmin())
   for d in range(n):
    if c==d:continue
    diff=z[c]-z[d];den=float(diff.square().sum());q=float((diff@U).square().sum())/den if den>1e-20 else None
    coverage.append(dict(seed=seed,stage=stage,class_index=c,other_class=d,tail=c in tail,other_tail=d in tail,same_parent=types[c]==types[d],nearest_train_mean=d==nearest,q=q))
  if prev is not None:
   ao,bo,wo=affine(prev);dw=w-wo
   # Row-vector affine maps for the two exact pre-Lorentz drift contributions.
   ma=dw@ao;ba=bo@dw.T;mb=w@(a-ao);bb=(b-bo)@w.T
   mt=w@a-wo@ao;bt=b@w.T-bo@wo.T
   err=max(float((mt-ma-mb).abs().max()),float((bt-ba-bb).abs().max()));assert err<1e-10
   for c in range(prev['total']):
    mu=prev['prototype'][c].double();cov=prev['cov'][c].double()-1e-5*torch.eye(512,dtype=torch.double);cov=cov*((counts[c]-1)/counts[c])
    ea=mu@ma.T+ba;eb=mu@mb.T+bb;et=mu@mt.T+bt
    va=float(((ma@cov)*ma).sum()+ea.square().sum());vb=float(((mb@cov)*mb).sum()+eb.square().sum());cross=float(((ma@cov)*mb).sum()+ea@eb);vt=float(((mt@cov)*mt).sum()+et.square().sum())
    identity=abs(vt-va-vb-2*cross);assert identity<1e-7*max(1,abs(vt))
    drifts.append(dict(seed=seed,stage=stage,class_index=c,tail=c in tail,train_count=counts[c],mapping_RMS=math.sqrt(max(0,va)/512),adapter_RMS=math.sqrt(max(0,vb)/512),total_RMS=math.sqrt(max(0,vt)/512),cross_per_dimension=2*cross/512,identity_error=identity,coordinate='TRAIN raw-unit moments through previous/current pre-TP affine adapters'))
  prev=s
  save('STATUS.json',dict(phase='RUNNING',seed=seed,stage=stage))
save('SPECTRA.json',spectra);save('COVERAGE.json',coverage);save('DRIFT.json',drifts)
save('VERIFICATION.json',dict(phase='COMPLETE',no_training=True,no_test=True,read_only_inputs=True,score_count_checks=54*3,checkpoint_count=18,affine_max_error=max(checks),elapsed_seconds=time.time()-START,output_bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),limitations=['development reused','TRAIN drift is descriptive, not causal validation attribution','historical covariance mixes task-time adapter coordinates','no validation features re-encoded']))
save('STATUS.json',dict(phase='COMPLETE'))
print(json.dumps(dict(phase='COMPLETE',seconds=time.time()-START)))
