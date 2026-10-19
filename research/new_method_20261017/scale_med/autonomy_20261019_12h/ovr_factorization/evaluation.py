import numpy as np
import torch
ARMS=['G','P','C']
def metrics(s,y,groups,k):
    pred=s.argmax(1).numpy();yy=y.numpy();rec=[float((pred[yy==c]==c).mean()) for c in range(k)]
    grec=[]
    for c in range(k):
        gs=sorted({groups[i] for i in np.where(yy==c)[0]});grec.append(float(np.mean([np.mean(pred[[i for i in np.where(yy==c)[0] if groups[i]==g]]==c) for g in gs])))
    return dict(BA=sum(rec)/k,recall=rec,correct=[int(((pred==c)&(yy==c)).sum()) for c in range(k)],counts=[int((yy==c).sum()) for c in range(k)],group_equal_BA=sum(grec)/k,group_recall=grec)

def evaluate(scores,ty,gg,complete):
    modelseal=dict(complete_arms=complete);summary={}
    for (name,st),s in scores.items():
        mask=ty<st+4;groups=[gg[i] for i in torch.where(mask)[0].tolist()];summary[name,st]=metrics(s[mask],ty[mask],groups,st+4)
    def check(phase):pass
    names=[a+'_ridge' for a in ARMS]+[a+'_CE' for a in ARMS]+modelseal['complete_arms'];truth=ty.numpy();rng=np.random.default_rng(74155);rec=np.zeros((len(names),2,10000,6));check('paired_bootstrap')
    for c in range(6):
        ids=np.where(truth==c)[0];keys=sorted({gg[i] for i in ids});members=[[i for i in ids if gg[i]==g] for g in keys];sizes=np.array([len(v) for v in members]);draw=rng.integers(0,len(keys),(10000,len(keys)));den=sizes[draw].sum(1)
        for ni,name in enumerate(names):
            for si,st in enumerate([1,2]):
                if c>=st+4:continue
                ok=(scores[name,st].argmax(1).numpy()==truth);hits=np.array([ok[ix].sum() for ix in members]);rec[ni,si,:,c]=hits[draw].sum(1)/den
    jboot=(rec[:,0,:,:5].mean(-1)+rec[:,1].mean(-1))/2*100
    J={n:100*(summary[n,1]['BA']+summary[n,2]['BA'])/2 for n in names};comparisons={};gates={}
    for a in [n for n in modelseal['complete_arms'] if n.endswith('_B') and n[:-1]+'S' in modelseal['complete_arms']]:
        n=a;representation=a.split('_')[0];diffs={}
        for r in ARMS:
            b=jboot[names.index(n)]-jboot[names.index(r+'_ridge')];diffs[r]=dict(J_gain_pp=J[n]-J[r+'_ridge'],CI95_pp=np.quantile(b,[.025,.975]).tolist(),CI12_adjusted_pp=np.quantile(b,[.05/24,1-.05/24]).tolist())
        b=jboot[names.index(n)]-jboot[names.index(a[:-1]+'S')];diffs['S']=dict(J_gain_pp=J[n]-J[a[:-1]+'S'],CI95_pp=np.quantile(b,[.025,.975]).tolist(),CI12_adjusted_pp=np.quantile(b,[.05/24,1-.05/24]).tolist())
        own=representation+'_ridge';stagegains=[100*(summary[n,s]['BA']-summary[own,s]['BA']) for s in [1,2]]
        birth=[100*(summary[n,1]['recall'][4]-summary[own,1]['recall'][4]),100*(summary[n,2]['recall'][5]-summary[own,2]['recall'][5]),100*(summary[n,2]['recall'][4]-summary[own,2]['recall'][4])]
        oldg=[100*np.mean(np.array(summary[n,s]['recall'][:4])-summary[own,s]['recall'][:4]) for s in [1,2]]+[100*np.mean(np.array(summary[n,2]['recall'][:5])-summary[own,2]['recall'][:5])]
        gate=dict(J_vs_standard_positive=diffs['S']['J_gain_pp']>0,class0_class1_each_stage_cost_within1pp=min(100*(summary[n,s]['recall'][c]-summary[own,s]['recall'][c]) for s in range(3) for c in [0,1])>=-1-1e-12,J_vs_G_at_least1pp=diffs['G']['J_gain_pp']>=1,J_vs_P_C_positive=all(diffs[r]['J_gain_pp']>0 for r in ['P','C']),all4_adjusted_lower_positive=all(z['CI12_adjusted_pp'][0]>0 for z in diffs.values()),both_stage_gain_nonnegative=min(stagegains)>=-1e-12,future_birth_and_retirement_nonnegative=min(birth)>=-1e-12,old_macro_cost_within1pp=min(oldg)>=-1-1e-12)
        comparisons[a]=dict(vs_ridge=diffs,stage_gain_own_pp=stagegains,class4_birth_class5_birth_class4_retired_gain_own_pp=birth,old_macro_gains_pp=oldg);gates[a]=dict(**gate,pass_all=all(gate.values()))
    passed=[a for a,v in gates.items() if v['pass_all']];decision='PRIVILEGED_OVR_HEADROOM' if passed else ('STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION' if len(modelseal['complete_arms'])==6 else 'UNRESOLVED_OPTIMIZATION_NO_EXTENSION')
    return dict(summary={n:{str(s):summary[n,s] for s in range(3) if (n,s) in summary} for n in sorted({name for name,st in scores})},J_pp=J,comparisons=comparisons,gates=gates,passing_arms=passed,decision=decision,method_success=False,bootstrap=dict(draws=10000,seed=74155,primary_family=12))
