"""Fixed count/context ablations and deterministic cost rules."""
import numpy as np
from scipy.special import logsumexp


def patient_counts(y,g,ids):
 return np.array([len(np.unique(g[ids][y[ids]==c])) for c in range(20)],dtype=int)


def inputs(features,b,f,c,counts,tail_indices):
 bp=b.argmax(1);fp=f.argmax(1);cp=c.argmax(1)
 if counts.ndim==1:counts=np.broadcast_to(counts,(len(b),20))
 r=np.arange(len(b));support=np.log1p(counts[r,fp])-np.log1p(counts[r,bp]);context=np.column_stack([np.isin(bp,tail_indices),np.isin(fp,tail_indices),cp==bp,cp==fp]).astype(float)
 return np.column_stack([features[:,:4],support]),np.column_stack([features,context])


def policy(model,x,disagree,lt,lc,tail_factor):
 if model['fallback']:return np.zeros(len(x),dtype=bool),np.zeros(len(x))
 z=np.column_stack([(x-model['mean'])/model['scale'],np.ones(len(x))]);s=z@model['coef'];a=1+lt*tail_factor*x[:,6]+lc*x[:,8];b=1+lt*tail_factor*x[:,5]+lc*x[:,7];log_odds=s[:,0]-s[:,1]+np.log(a)-np.log(b)
 return disagree&(log_odds>0),log_odds


def oracle_diagnostics(scores,y,g,fold,order,decisions):
 result=[];B=scores['df_matched_joint'].argmax(1);F=scores['plain_correction'].argmax(1);C=scores['cls768_ridge'].argmax(1);bc=B==y;fc=F==y;cc=C==y;dis=B!=F
 for outer in [None,0,1,2]:
  scope=np.ones(len(y),dtype=bool) if outer is None else fold==outer
  for label in [None,*range(20)]:
   mask=scope if label is None else scope&(y==label);ids=np.flatnonzero(mask)
   if not len(ids):continue
   # Aggregate scope is BA; per-class rows use conditional recall/H_C contribution.
   if label is None:w=1/(20*np.bincount(y[ids],minlength=20)[y[ids]])
   else:w=np.full(len(ids),1/len(ids))
   def mass(v):return float(w@v[ids])
   rescue=fc&~bc;harm=bc&~fc;neither=~bc&~fc;choices={}
   for name,select in decisions.items():
    correct=np.where(select,fc,bc);choices[name]=dict(BA_or_recall=mass(correct),CLS_harm=mass(cc&~correct),accepted_rescue=mass(select&rescue),rejected_rescue=mass(~select&rescue),accepted_harm=mass(select&harm),rejected_harm=mass(~select&harm),CLS_harm_delta_vs_B=mass(cc*(bc.astype(int)-correct.astype(int))),chosen_F_images=int((select&mask).sum()))
   result.append(dict(scope='pooled' if outer is None else f'outer_{outer}',original_class=None if label is None else order[label],images=len(ids),patients=len(np.unique(g[ids])),disagreement_images=int((dis&mask).sum()),disagreement_patients=len(np.unique(g[mask&dis])),oracle=mass(bc|fc),unavoidable_CLS_harm=mass(cc&neither),B_CLS_harm=mass(cc&~bc),F_CLS_harm=mass(cc&~fc),rescue=mass(rescue),harm=mass(harm),both_wrong=mass(neither),disagreement_both_wrong=mass(neither&dis),choices=choices))
 return result
