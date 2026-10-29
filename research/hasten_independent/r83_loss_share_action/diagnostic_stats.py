"""Prespecified numerical ties, rank and exact-uniform selection diagnostics."""
import math

def kendall(x,y,tol=1e-7):
 c=d=tx=ty=0
 for i in range(len(x)):
  for j in range(i):
   u=x[i]-x[j];v=y[i]-y[j];a=0 if abs(u)<=tol else (1 if u>0 else -1);b=0 if abs(v)<=tol else (1 if v>0 else -1)
   if a and b:
    if a==b:c+=1
    else:d+=1
   elif a:ty+=1
   elif b:tx+=1
 den=math.sqrt((c+d+tx)*(c+d+ty));return None if not den else (c-d)/den

def selection(branches,source='Qa',target='Qb'):
 v=[b for b in branches if not b['duplicate'] and not b.get('fixed_mean',False)];scores=[b['stats'][source]['utility'] for b in v];best=[i for i,x in enumerate(scores) if max(scores)-x<=1e-7];out={'best_actions':[v[i]['action'] for i in best]}
 for metric in ['BA','tail_BA']:
  chosen=sum(v[i]['stats'][target][metric] for i in best)/len(best);uniform=sum(b['stats'][target][metric] for b in v)/len(v);baseline=v[0]['stats'][target][metric];out[metric]=dict(chosen=chosen,uniform=uniform,action0=baseline,gain_vs_uniform=chosen-uniform,gain_vs_action0=chosen-baseline)
 mean=next((b for b in branches if b.get('fixed_mean',False)),None)
 if mean is not None:
  for metric in ['BA','tail_BA']:out[metric]['fixed_mean']=mean['stats'][target][metric];out[metric]['gain_vs_fixed_mean']=out[metric]['chosen']-mean['stats'][target][metric]
 return out

def check():
 assert kendall([1,2,3],[1,2,3])==1 and kendall([1,2,3],[3,2,1])==-1 and kendall([1,1],[1,1]) is None
 b=[dict(action=a,duplicate=False,stats={'Qa':dict(utility=u),'Qb':dict(BA=z,tail_BA=z)}) for a,u,z in [(0,1,10),(1,2,20),(2,2,40)]];b.append(dict(action=0,duplicate=True,stats={'Qa':dict(utility=999),'Qb':dict(BA=999,tail_BA=999)}));x=selection(b);assert x['best_actions']==[1,2] and x['BA']['chosen']==30 and abs(x['BA']['gain_vs_uniform']-(30-70/3))<1e-12
 return dict(rank_ties=True,uniform_excludes_duplicate=True,mean_top_ties=True)
