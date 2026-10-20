import sys,os,json
from pathlib import Path
import torch
from space import check_space

def weights(y,groups):
 from collections import Counter
 count=Counter(groups);by={int(c):len({groups[i] for i in torch.where(y==c)[0].tolist()}) for c in y.unique()}
 return torch.tensor([1/(by[int(c)]*count[g]) for c,g in zip(y,groups)],dtype=torch.float64)
def geometry(x,y,groups):
 q=weights(y,groups);q=q/q.sum();mu=q@x;z=x-mu;cov=z.T@(q[:,None]*z);scatter=float(cov.trace());square=float(cov.square().sum());pr=scatter**2/square if square>0 else 0.;classes=[];centers=[]
 for c in y.unique().tolist():
  ix=torch.where(y==c)[0];v=x[ix];gg=[groups[i] for i in ix.tolist()];w=weights(y[ix],gg);w=w/w.sum();m=w@v;centers.append(m);keys=sorted(set(gg));gc=torch.stack([v[torch.tensor([g==k for g in gg])].mean(0) for k in keys]);d=gc[:,None]-gc[None,:];pd=d.square().sum(-1);sel=torch.triu_indices(len(gc),len(gc),1);pairs=pd[sel.unbind()]
  classes.append(dict(class_index=c,rows=len(ix),known_groups=len(keys),within_class_scatter=float((w[:,None]*(v-m).square()).sum()),group_centroid_scatter=float((gc-m).square().sum(1).mean()),cross_group_centroid_squared_distance_quantiles=torch.quantile(pairs,torch.tensor([0.,.25,.5,.75,1.],dtype=torch.float64)).tolist() if c==1 else None))
 centers=torch.stack(centers);return dict(global_scatter=scatter,covariance_participation_ratio=pr,class_center_squared_distances=(centers[:,None]-centers[None,:]).square().sum(-1).tolist(),classes=classes)
def main(cfg):
 out=Path(cfg['output']);check_space(cfg['window_root'],reserve=2**20);assert not torch.cuda.is_initialized()
 if cfg['mode']=='INJECT_FAILURE':raise RuntimeError('synthetic injected failure')
 if cfg['mode']=='PREFLIGHT':
  x=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.]],dtype=torch.float64);y=torch.tensor([0,0,1,1]);groups=['a','b','c','d'];a=geometry(x,y,groups);b=geometry(-x,y,groups);assert a==b;assert float((x-(-x)).norm(dim=1).min())>0;assert geometry(torch.ones_like(x),y,groups)['global_scatter']==0;assert 1<=a['covariance_participation_ratio']<=3
  (out/'aggregate.json').write_text(json.dumps(dict(pass_all=True,rigid_transform_geometry_invariant=True,nonzero_drift_not_collapse=True,constant_scatter_zero=True,method_success=False),indent=2));return
 reference=Path(cfg['reference']).resolve();feature=Path(cfg['feature']).resolve();allowed={reference,feature}
 protected=Path(cfg['protected_root']).resolve()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(protected) and p not in allowed and not p.is_relative_to(Path(cfg['root']).resolve()):raise PermissionError('outside frozen training-only allowlist')
 sys.addaudithook(guard)
 for bad in [protected/'depth_residual_20261018/run/inputs/1.private.pt',protected/'autonomy_20261019_12h/initial_bias/fit/predictions.private.pt']:
  try:guard('open',(str(bad),'rb'))
  except PermissionError:pass
  else:raise AssertionError('guard probe')
 obj=torch.load(reference,map_location='cpu',weights_only=False);mask=~obj['valid'];y=obj['y'][mask];groups=[obj['groups'][i] for i in torch.where(mask)[0].tolist()];old=obj['g'][mask].double();allnew=torch.load(feature,map_location='cpu',weights_only=False);assert allnew.shape==obj['g'].shape==(3236,1536);new=allnew[mask].double();del obj,allnew
 assert old.shape==new.shape==(2555,1536) and torch.bincount(y).tolist()==[307,20,170,2058]
 assert torch.isfinite(old).all() and torch.isfinite(new).all();assert max(float((v.norm(dim=1)-1).abs().max()) for v in [old,new])<1e-5
 a,b=geometry(old,y,groups),geometry(new,y,groups);assert min(a['global_scatter'],a['covariance_participation_ratio'])>1e-12;ratios=dict(scatter=b['global_scatter']/a['global_scatter'],participation=b['covariance_participation_ratio']/a['covariance_participation_ratio']);drift=(old-new).norm(dim=1)
 result=dict(source_commit=cfg['source_commit'],old=a,new=b,ratios=ratios,direction_L2_mean=float(drift.mean()),direction_L2_quantiles=torch.quantile(drift,torch.tensor([0.,.25,.5,.75,1.],dtype=torch.float64)).tolist(),old_halves_exact=torch.equal(old[:,:768],old[:,768:]),new_halves_exact=torch.equal(new[:,:768],new[:,768:]),training_rows=2555,validation_rows_used=0,test_rows_used=0,new_fit_units=0,new_predictions=0,new_image_forwards=0,GPU_initialized=False,guard_negative_probes_passed=True,decision='INITIAL_TRAIN_EXTREME_CONTRACTION_COMPATIBLE' if max(ratios.values())<=.1 else 'NARROW_GLOBAL_CONTRACTION_NOT_SUPPORTED',method_success=False,not_causal_risk_attribution=True)
 (out/'aggregate.json').write_text(json.dumps(result,indent=2,allow_nan=False))
