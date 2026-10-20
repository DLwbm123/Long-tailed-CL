"""Head-score span and dimension-matched training PCA in original augmented coordinates."""
import torch


def bases(j0,cls_head,factors,gram,mean,rank_rtol):
 d,k=j0.shape;assert cls_head.shape==(769,k) and d==1537
 cls=torch.zeros_like(j0);cls[:768]=cls_head[:768]/factors[:768,None];cls[-1]=cls_head[-1]/factors[-1]
 h=torch.eye(k,dtype=j0.dtype)-torch.ones(k,k,dtype=j0.dtype)/k;features=torch.cat([j0[:-1]@h,cls[:-1]@h],1);norms=torch.linalg.vector_norm(features,dim=0);features=features[:,norms>0]/norms[norms>0]
 u,s,_=torch.linalg.svd(features,full_matrices=False);m=int((s>s.max()*rank_rtol).sum());assert 0<m<=2*(k-1)
 qb=torch.zeros(d,m+1,dtype=j0.dtype);qb[-1,0]=1;qb[:-1,1:]=u[:,:m]
 cov=gram[:-1,:-1]-mean[:-1,None]*mean[None,:-1];ev,vec=torch.linalg.eigh((cov+cov.T)/2);assert float(ev.min())>=-max(1.,float(ev.abs().max()))*1e-10
 qp=torch.zeros_like(qb);qp[-1,0]=1;qp[:-1,1:]=vec[:,-m:]
 assert max(float((q.T@q-torch.eye(m+1,dtype=q.dtype)).abs().max()) for q in [qb,qp])<1e-10
 span=torch.cat([j0@h,cls@h],1);span_error=float((span-qb@(qb.T@span)).abs().max());assert span_error<1e-9
 return qb,qp,dict(rank=m+1,nominal_parameters=(m+1)*k,head_singular_values=s.tolist(),rank_relative_tolerance=rank_rtol,head_span_max_error=span_error,PCA_selected_eigenvalues=ev[-m:].tolist(),PCA_boundary_eigenvalue_gap=float(ev[-m]-ev[-m-1]),PCA_center_only_for_selection=True)


def diagnostic(z,y,w,j0,residual,q):
 # Fix the full model's hardest wrong class; independent maxima do not decompose linearly.
 base=z@j0;parallel=q@(q.T@residual);perp=residual-parallel;ps=z@parallel;ns=z@perp;full=base+ps+ns;other=full.clone();other[torch.arange(len(y)),y]=-torch.inf;competitor=other.argmax(1);iy=torch.arange(len(y));delta_p=ps[iy,y]-ps[iy,competitor];delta_n=ns[iy,y]-ns[iy,competitor];base_fixed=base[iy,y]-base[iy,competitor]
 base_wrong=base.clone();base_wrong[iy,y]=-torch.inf;base_margin=base[iy,y]-base_wrong.max(1).values;full_margin=full[iy,y]-other.max(1).values;switch=base_fixed-base_margin;err=float((full_margin-base_margin-delta_p-delta_n-switch).abs().max());assert err<1e-9
 good0=base.argmax(1)==y;good1=full.argmax(1)==y;out=[]
 for c in range(20):
  mask=y==c;row=dict(class_index=c,images=int(mask.sum()),anchor_correct=int(good0[mask].sum()),plain_correct=int(good1[mask].sum()),anchor_accuracy=float(good0[mask].double().mean()),plain_accuracy=float(good1[mask].double().mean()),anchor_PP_accuracy=float((w[mask]*good0[mask]).sum()/w[mask].sum()),plain_PP_accuracy=float((w[mask]*good1[mask]).sum()/w[mask].sum()),anchor_margin_quantiles=torch.quantile(base_margin[mask],torch.tensor([0.,.25,.5,.75,1.],dtype=z.dtype)).tolist(),plain_margin_quantiles=torch.quantile(full_margin[mask],torch.tensor([0.,.25,.5,.75,1.],dtype=z.dtype)).tolist(),groups={})
  for name,selected in [('rescue',mask&~good0&good1),('harm',mask&good0&~good1),('all',mask)]:
   n=int(selected.sum())
   row['groups'][name]=dict(images=n,PP_mass=float(w[selected].sum()),parallel_mean=float(delta_p[selected].mean()) if n else None,perpendicular_mean=float(delta_n[selected].mean()) if n else None,competitor_switch_mean=float(switch[selected].mean()) if n else None,parallel_PP_mean=float((w[selected]*delta_p[selected]).sum()/w[selected].sum()) if n else None,perpendicular_PP_mean=float((w[selected]*delta_n[selected]).sum()/w[selected].sum()) if n else None,parallel_negative=int((delta_p[selected]<0).sum()),perpendicular_negative=int((delta_n[selected]<0).sum()))
  out.append(row)
 return dict(classes=out,additive_decomposition_max_error=err,residual_norm=float(torch.linalg.vector_norm(residual)),parallel_residual_norm=float(torch.linalg.vector_norm(parallel)),perpendicular_residual_norm=float(torch.linalg.vector_norm(perp)),fixed_competitor='full_plain_model_max_wrong',margin_units='parent_joint_temperature')
