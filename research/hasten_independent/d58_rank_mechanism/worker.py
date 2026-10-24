"""Full TRAIN direct-gradient and saved-parameter accounting, zero optimizer updates."""
import os,json,time,math,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);source=Path(cfg['source']);torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False

def write(n,z):(root/n).write_text(json.dumps(z,indent=2,allow_nan=False))
def guard():
 assert time.time()<json.loads((root/'FIRST_ACTUAL.private.json').read_text())['deadline_unix']-25
 assert torch.cuda.max_memory_reserved()<8*2**30

def spectrum(z):
 s=torch.linalg.svdvals(z.double());p=s.square();p=p/p.sum().clamp_min(1e-30)
 return dict(singular_values=s[:16].tolist(),participation_rank=float(1/p.square().sum().clamp_min(1e-30)),leading_energy_fraction=float(p[0]))

def main():
 # Synthetic chain rule at a nonzero product, not an optimized scientific model.
 x=torch.arange(35,device='cuda',dtype=torch.float64).reshape(7,5)/17;u=torch.arange(10,device='cuda',dtype=torch.float64).reshape(5,2)/13;v=torch.flip(u,[0]).clone();u.requires_grad_();v.requires_grad_();g=torch.arange(25,device='cuda',dtype=torch.float64).reshape(5,5)/11
 (g*(u@v.T)).sum().backward();assert torch.allclose(u.grad,g@v) and torch.allclose(v.grad,g.T@u)
 records=[]
 for ds in ['NIH','ISIC','HK']:
  dc=json.loads((source/ds/'config.private.json').read_text());C=len(dc['order'])
  for seed in [74002,74003,74004]:
   for phase in ['initial','final']:
    guard();stage=0 if phase=='initial' else len(dc['stages'])-1;blocks=[torch.load(source/ds/'memory'/f'block{i}.private.pt',map_location='cpu',weights_only=False) for i in range(stage+1)];xx=torch.cat([b['x'] for b in blocks]).cuda();yy=torch.cat([b['y'] for b in blocks]).cuda();n=sum(len(v) for v in dc['stages'][:stage+1]);counts=torch.bincount(yy,minlength=n);assert counts.tolist()==dc['train_counts_ordered'][:n];means=torch.stack([xx[yy==c].mean(0) for c in range(n)]);N=len(xx)
    if phase=='initial':
     with torch.random.fork_rng(devices=[0]):
      torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);U=torch.randn(512,16,device='cuda')/math.sqrt(512);V=torch.zeros_like(U)
    else:
     h=torch.load(source/ds/str(seed)/f'head{stage}.private.pt',map_location='cuda',weights_only=False);U=h['state']['U'];V=h['state']['V'];assert torch.equal(means,h['means'])
    initialU=U.clone();initialV=V.clone();delta=U@V.T;M=(torch.eye(512,device='cuda')+delta).detach().requires_grad_();stats={};gradnorm=None
    correct=torch.zeros(n,device='cuda',dtype=torch.long);losses=torch.zeros(n,device='cuda',dtype=torch.double);margins=torch.zeros_like(losses)
    for ids in torch.arange(N,device='cuda').split(512):
     z=20*F.normalize(xx[ids]@M.T,dim=1)@F.normalize(means@M.T,dim=1).T;ce=F.cross_entropy(z,yy[ids],reduction='none');(ce/(n*counts[yy[ids]])).sum().backward();pred=z.detach().argmax(1);correct+=torch.bincount(yy[ids][pred==yy[ids]],minlength=n);losses.scatter_add_(0,yy[ids],ce.detach().double());other=z.detach().clone();other[torch.arange(len(ids),device='cuda'),yy[ids]]=-torch.inf;mar=z.detach()[torch.arange(len(ids),device='cuda'),yy[ids]]-other.max(1).values;margins.scatter_add_(0,yy[ids],mar.double())
    G=M.grad.detach();normG=G.norm();assert torch.isfinite(G).all();gu=G@V;gv=G.T@U;firstV=-.001*gv/(gv.abs()+1e-8);firstDelta=U@firstV.T
    if phase=='initial':assert gu.norm()==0 and gv.norm()>0
    dot=(G*delta).sum();dn=delta.norm();radial=G*0 if dn==0 else dot/(dn*dn)*delta;tangent=G-radial
    with torch.no_grad():
     unitmeans=F.normalize(means,dim=1);tmeans=F.normalize(means@M.T,dim=1);rawwithin=0.;afterwithin=0.
     for ids in torch.arange(N,device='cuda').split(2048):
      rawwithin+=float(((xx[ids]-unitmeans[yy[ids]])**2).sum(1).div(n*counts[yy[ids]]).sum());afterwithin+=float(((F.normalize(xx[ids]@M.T,dim=1)-tmeans[yy[ids]])**2).sum(1).div(n*counts[yy[ids]]).sum())
    tail=counts<250;row=dict(dataset=ds,seed=seed,phase=phase,stage=stage,rows=N,counts=counts.tolist(),correct=correct.tolist(),class_mean_margin=(margins/counts).tolist(),macro_CE=float((losses/counts).mean()),macro_BA=float((correct/counts).mean()),tail_BA=float((correct[tail]/counts[tail]).mean()),U_F_norm=float(U.norm()),V_F_norm=float(V.norm()),product_F_norm=float(dn),factor_norm_ratio=float(U.norm()/V.norm()) if V.norm()>0 else None,U_spectrum=spectrum(U),V_spectrum=spectrum(V) if V.norm()>0 else None,product_spectrum=spectrum(delta) if dn>0 else None,direct_gradient_norm=float(normG),direct_gradient_spectrum=spectrum(G),initial_factor_U_gradient_norm=float(gu.norm()),factor_V_gradient_spectrum=spectrum(gv),algebraic_Adam_CE_product_direction=spectrum(firstDelta),gradient_scale_invariance_relative=float((G*M.detach()).sum()/(normG*M.detach().norm()).clamp_min(1e-20)),gradient_rotation_invariance_relative=float((G@M.detach().T-M.detach()@G.T).norm()/(2*normG*M.detach().norm()).clamp_min(1e-20)),gradient_trace_relative=float(G.trace()/normG.clamp_min(1e-20)),gradient_antisymmetric_relative=float((G-G.T).norm()/(2*normG.clamp_min(1e-20))),radial_alignment=float(dot/(normG*dn).clamp_min(1e-20)) if dn>0 else None,nonradial_gradient_fraction=float(tangent.norm()/normG.clamp_min(1e-20)),raw_equalclass_within_squared_distance=rawwithin,transformed_equalclass_within_squared_distance=afterwithin,raw_between_center_scatter=float((unitmeans-unitmeans.mean(0)).square().sum(1).mean()),transformed_between_center_scatter=float((tmeans-tmeans.mean(0)).square().sum(1).mean()))
    assert torch.equal(U,initialU) and torch.equal(V,initialV);records.append(row);write('RESULTS.json',dict(records=records,optimizer_steps=0,new_learned_parameters=0,VAL_reads=0,TEST_reads=0,synthetic_chain_rule_passed=True,all_original_parameters_unchanged=True));write('STATUS.json',dict(phase='AUDIT',dataset=ds,seed=seed,cell=phase,time=time.time()));del xx,yy,M,U,V,blocks;torch.cuda.empty_cache()
 assert len(records)==18;write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:(root/'FAILURE.private.txt').write_text(traceback.format_exc());raise
