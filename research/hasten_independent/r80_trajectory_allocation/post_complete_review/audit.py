"""Three saved-only within-context diagnostics requested by the Pro evidence review."""
import json,math,statistics,time,resource,sys
from pathlib import Path
r=Path(__file__).parent;start=time.time();rows=[v for v in json.loads((r/'CONTEXT_RECORDS.private.json').read_text()) if v['arm']=='protected_controller'];assert len(rows)==48
counts=[626,3,6,85,288,403,736,488,252,167,595,261,6,34,83,25,23,635,127,662,18,654,644];tail=set(i for i,n in enumerate(counts) if n<250)
def mean(v):return sum(v)/len(v)
def norm(v):return math.sqrt(sum(x*x for x in v))
def center(v):m=mean(v);return [x-m for x in v]
def pearson(x,y):
 x=center(x);y=center(y);den=norm(x)*norm(y);return None if den<1e-18 else sum(a*b for a,b in zip(x,y))/den
def adv(v):
 z=center(v);s=math.sqrt(mean([x*x for x in z]));return [x/(s+1e-8) for x in z]
def kendall(x,y,xtol=1e-7,ytol=1e-10):
 C=D=TX=TY=both=0
 for i in range(len(x)):
  for j in range(i):
   a=x[i]-x[j];b=y[i]-y[j];sx=0 if abs(a)<=xtol else (1 if a>0 else -1);sy=0 if abs(b)<=ytol else (1 if b>0 else -1)
   if not sx and not sy:both+=1
   elif not sx:TX+=1
   elif not sy:TY+=1
   elif sx==sy:C+=1
   else:D+=1
 den=math.sqrt((C+D+TX)*(C+D+TY));return dict(tau_b=None if not den else (C-D)/den,concordant=C,discordant=D,tied_reward=TX,tied_other=TY,tied_both=both)
def summarize(v):
 vv=[x for x in v if x is not None];return dict(defined_groups=len(vv),undefined_groups=len(v)-len(vv),mean=None if not vv else mean(vv),minimum=None if not vv else min(vv),maximum=None if not vv else max(vv))
assert abs(kendall([1,2,3],[1,2,3])['tau_b']-1)<1e-12
assert abs(kendall([1,2,3],[3,2,1])['tau_b']+1)<1e-12
assert kendall([1,2,3],[1,1,1])['tau_b'] is None
protection=[];alignment=[];increments=[];byclass={c:[] for c in range(23)};actiontypes={'tail':[],'head':[],'balanced':[]};examples=[]
for v in rows:
 n=len(v['query_counts']);den=v['query_counts'];P=v['protected_classes'];u=v['utilities'][:8];D=v['damage'][:8];M=[a+b for a,b in zip(u,D)];am=adv(M);ap=adv(u)
 assert all(abs(M[i]+mean(br['class_brier']))<1e-6 for i,br in enumerate(v['branches'][:8]))
 xc=center(M);yc=center(u);slope=sum(a*b for a,b in zip(xc,yc))/sum(a*a for a in xc);res=[b-slope*a for a,b in zip(xc,yc)];active=[sum(p and br['class_brier'][c]>v['anchor_brier'][c] for c,p in enumerate(P)) for br in v['branches'][:8]];positive=sum(P)
 protection.append(dict(stage=v['stage'],context=v['context'],advantage_cosine=pearson(am,ap),sign_flips=sum(a*b<0 for a,b in zip(am,ap)),top_rank_overlap=bool(set(i for i,x in enumerate(M) if max(M)-x<=1e-7)&set(i for i,x in enumerate(u) if max(u)-x<=1e-7)),rank=kendall(M,u,ytol=1e-7),affine_slope=slope,affine_residual_relative=norm(res)/norm(yc),all_classes_protected=all(P),all_protected_damage_active_all_branches=all(x==positive for x in active),mean_protected_damage_activation_fraction=mean(active)/positive))
 ba=[100*mean([k/z for k,z in zip(br['query_correct'],den)]) for br in v['branches']];tba=[100*mean([k/z for c,(k,z) in enumerate(zip(br['query_correct'],den)) if c in tail]) for br in v['branches']];rr=v['absolute_returns'];macro=kendall(u,ba[:8]);tails=kendall(u,tba[:8]);alignment.append(dict(stage=v['stage'],context=v['context'],macro_rank=macro,tail_rank=tails,macro_pearson=pearson(u,ba[:8]),tail_pearson=pearson(u,tba[:8]),macro_BA_constant=max(ba[:8])-min(ba[:8])<1e-10,tail_BA_constant=max(tba[:8])-min(tba[:8])<1e-10,positive_R_macro_loss=sum(x>1e-7 and ba[i]<ba[8]-1e-10 for i,x in enumerate(rr)),positive_R_tail_loss=sum(x>1e-7 and tba[i]<tba[8]-1e-10 for i,x in enumerate(rr)),positive_R_macro_tie=sum(x>1e-7 and abs(ba[i]-ba[8])<1e-10 for i,x in enumerate(rr)),positive_R_tail_tie=sum(x>1e-7 and abs(tba[i]-tba[8])<1e-10 for i,x in enumerate(rr)),positive_R_count=sum(x>1e-7 for x in rr)))
 for branch,br in enumerate(v['branches'][:8]):
  for h,a in enumerate(br['actions']):
   cur=br['allocation'][h]['controlled_counts'];base=v['branches'][8]['allocation'][h]['controlled_counts'];delta=[x-y for x,y in zip(cur,base)];assert sum(delta)==0
   own=0 if a==0 else delta[a-1];item=dict(stage=v['stage'],context=v['context'],branch=branch,decision=h,action=a,priority_class=None if a==0 else a-1,own_class_increment=own,tail_total_increment=sum(delta[c] for c in range(n) if c in tail),changed_class_counts=sum(x!=0 for x in delta),absolute_class_count_change=sum(abs(x) for x in delta))
   increments.append(item);key='balanced' if not a else ('tail' if a-1 in tail else 'head');actiontypes[key].append(item)
   if a:byclass[a-1].append(own)
class_summary=[dict(class_index=c,TRAIN_count=counts[c],priority_decisions=len(v),mean_own_class_increment=None if not v else mean(v),zero_increment_decisions=sum(x==0 for x in v),negative_increment_decisions=sum(x<0 for x in v)) for c,v in byclass.items()]
summary=dict(scope='R80 C only, same context/initial state/common stream/endpoints; saved TRAIN aggregates only',new_model_scores=0,new_optimizer_steps=0,contexts=48,candidate_trajectories=384,decisions=768,action_position_control_upper_bound=1/6,action_types={k:dict(decisions=len(v),mean_own_class_increment=mean([x['own_class_increment'] for x in v]),zero_own_class_increment=sum(x['own_class_increment']==0 for x in v),mean_total_tail_increment=mean([x['tail_total_increment'] for x in v]),mean_L1_class_count_change=mean([x['absolute_class_count_change'] for x in v])) for k,v in actiontypes.items()},class_priority_increment=class_summary,protection=dict(advantage_cosine=summarize([v['advantage_cosine'] for v in protection]),rank_tau_b=summarize([v['rank']['tau_b'] for v in protection]),sign_flips=sum(v['sign_flips'] for v in protection),total_advantages=384,top_rank_overlap_groups=sum(v['top_rank_overlap'] for v in protection),all_classes_protected_groups=sum(v['all_classes_protected'] for v in protection),all_protected_damage_active_groups=sum(v['all_protected_damage_active_all_branches'] for v in protection),affine_slope=summarize([v['affine_slope'] for v in protection]),relative_affine_residual=summarize([v['affine_residual_relative'] for v in protection])),reward_identification=dict(macro_rank_tau_b=summarize([v['macro_rank']['tau_b'] for v in alignment]),tail_rank_tau_b=summarize([v['tail_rank']['tau_b'] for v in alignment]),macro_BA_constant_groups=sum(v['macro_BA_constant'] for v in alignment),tail_BA_constant_groups=sum(v['tail_BA_constant'] for v in alignment),macro_pearson=summarize([v['macro_pearson'] for v in alignment]),tail_pearson=summarize([v['tail_pearson'] for v in alignment]),positive_R_count=sum(v['positive_R_count'] for v in alignment),positive_R_macro_loss=sum(v['positive_R_macro_loss'] for v in alignment),positive_R_tail_loss=sum(v['positive_R_tail_loss'] for v in alignment),positive_R_macro_tie=sum(v['positive_R_macro_tie'] for v in alignment),positive_R_tail_tie=sum(v['positive_R_tail_tie'] for v in alignment)),independent_query_ranking_reliability='unavailable: alternate legal-query endpoint scores/component statistics not saved; nonretained branch weights discarded; do not compare different-context folds as independent rankings',not_independent_generalization_or_causal_failure_proof=True,ranking_utility_tolerance=1e-7,ranking_BA_tolerance=1e-10,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),wall_seconds=time.time()-start,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)
(r/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(r/'CONTEXT_DIAGNOSTICS.json').write_text(json.dumps(dict(protection=protection,reward_alignment=alignment),indent=2));(r/'ACTION_INCREMENT_COUNTS.json').write_text(json.dumps(increments,indent=2));print(json.dumps(summary,indent=2))
