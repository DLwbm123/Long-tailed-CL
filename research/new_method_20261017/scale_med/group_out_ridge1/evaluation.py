import numpy as np
import torch
# Same metrics as sealed historical reports; no threshold selection.
def metrics(s,y,groups,k):
 pred=s.argmax(1).numpy();yy=y.numpy();rec=[float((pred[yy==c]==c).mean()) for c in range(k)];gr=[]
 for c in range(k):
  ids=np.where(yy==c)[0];keys=sorted({groups[i] for i in ids});gr.append(float(np.mean([np.mean(pred[[i for i in ids if groups[i]==g]]==c) for g in keys])))
 return dict(BA=float(np.mean(rec)),recall=rec,correct=[int(((pred==c)&(yy==c)).sum()) for c in range(k)],counts=[int((yy==c).sum()) for c in range(k)],group_equal_BA=float(np.mean(gr)),group_recall=gr)
def evaluate(scores,y,groups,complete):
 required={(n,s) for n in ['O','I','E','G_Matched','P_Matched','C_Matched'] for s in range(3)}
 if not required.issubset(scores):return dict(decision='UNRESOLVED_GROUP_OUT_RIDGE1',gates={},method_success=False)
 names=sorted({n for n,s in scores});summary={}
 for (n,s),v in scores.items():
  mask=y<s+4;summary[n,s]=metrics(v[mask],y[mask],[groups[i] for i in torch.where(mask)[0].tolist()],s+4)
 yy=y.numpy();rng=np.random.default_rng(74221);rec=np.zeros((len(names),3,10000,6))
 for c in range(6):
  ids=np.where(yy==c)[0];keys=sorted({groups[i] for i in ids});members=[[i for i in ids if groups[i]==g] for g in keys];sizes=np.array([len(a) for a in members]);draw=rng.integers(0,len(keys),(10000,len(keys)));den=sizes[draw].sum(1)
  for ni,n in enumerate(names):
   for st in range(3):
    if c>=st+4:continue
    ok=scores[n,st].argmax(1).numpy()==yy;hits=np.array([ok[ix].sum() for ix in members]);rec[ni,st,:,c]=hits[draw].sum(1)/den
 jboot=(rec[:,1,:,:5].mean(-1)+rec[:,2].mean(-1))/2*100;J={n:100*(summary[n,1]['BA']+summary[n,2]['BA'])/2 for n in names};ci=lambda b:dict(CI95_pp=np.quantile(b,[.025,.975]).tolist(),CI6_adjusted_pp=np.quantile(b,[.05/12,1-.05/12]).tolist());contrasts={};ni=names.index('O');bi=names.index('G_Matched')
 for a in ['G','P','C']:
  n=a+'_Matched';ds=jboot[ni]-jboot[names.index(n)];contrasts['J_vs_'+a]=dict(gain_pp=J['O']-J[n],**ci(ds))
 for st in range(3):
  ds=100*(rec[ni,st,:,1]-rec[bi,st,:,1]);contrasts['class1_stage'+str(st)]=dict(gain_pp=100*(summary['O',st]['recall'][1]-summary['G_Matched',st]['recall'][1]),**ci(ds))
 own='G_Matched';n='O';stages=[100*(summary[n,s]['BA']-summary[own,s]['BA']) for s in [1,2]];birth=[100*(summary[n,s]['recall'][c]-summary[own,s]['recall'][c]) for s,c in [(1,4),(2,4),(2,5)]];old=[100*np.mean(np.array(summary[n,s]['recall'][:4])-summary[own,s]['recall'][:4]) for s in [1,2]]+[100*np.mean(np.array(summary[n,2]['recall'][:5])-summary[own,2]['recall'][:5])];gates=dict(J_vs_G_at_least1=contrasts['J_vs_G']['gain_pp']>=1,J_vs_P_C_positive=all(contrasts['J_vs_'+a]['gain_pp']>0 for a in ['P','C']),all6_adjusted_lower_positive=all(v['CI6_adjusted_pp'][0]>0 for v in contrasts.values()),class1_each_stage_at_least5=all(contrasts['class1_stage'+str(s)]['gain_pp']>=5-1e-12 for s in range(3)),both_future_BA_nonnegative=min(stages)>=-1e-12,future_birth_retired_nonnegative=min(birth)>=-1e-12,old_macro_cost_within1=min(old)>=-1-1e-12,class0_cost_each_stage_within1=min(100*(summary[n,s]['recall'][0]-summary[own,s]['recall'][0]) for s in range(3))>=-1-1e-12);gates={k:bool(v) for k,v in gates.items()};gates['pass_all']=all(gates.values())
 mechanism=dict(gain_pp=J['O']-J['I'],CI95_pp=np.quantile(jboot[ni]-jboot[names.index('I')],[.025,.975]).tolist());mechanism['gatekeeping_pass']=bool(gates['pass_all'] and mechanism['gain_pp']>=.5 and mechanism['CI95_pp'][0]>0)
 transitions={}
 for name in ['E','I','O']:
  transitions[name]={}
  for st in range(3):
   pairs=[]
   for c in range(st+4):
    mask=y==c;oldok=scores['G_Matched',st][mask].argmax(1)==y[mask];newok=scores[name,st][mask].argmax(1)==y[mask];pairs.append(dict(class_index=c,count=int(mask.sum()),rescued=int((~oldok & newok).sum()),harmed=int((oldok & ~newok).sum())))
   delta=sum((v['rescued']-v['harmed'])/v['count'] for v in pairs)/(st+4);assert abs(delta-(summary[name,st]['BA']-summary['G_Matched',st]['BA']))<1e-12;transitions[name][str(st)]=pairs
 return dict(mechanism_O_minus_I=mechanism,rescue_harm=transitions,summary={n:{str(s):summary[n,s] for s in range(3)} for n in names},J_pp=J,contrasts=contrasts,gates=gates,future_BA_gains_pp=stages,birth_retired_gains_pp=birth,old_macro_gains_pp=old,decision='GROUP_OUT_RIDGE1_PERFORMANCE_SIGNAL' if gates['pass_all'] else 'STOP_GROUP_OUT_RIDGE1_NO_EXTENSION',method_success=False,bootstrap=dict(seed=74221,draws=10000,primary_family=6),frozen_reference_scores_reused_not_independent=True)
