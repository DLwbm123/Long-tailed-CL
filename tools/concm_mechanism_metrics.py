"""Class aggregates only; fixed competitor geometry and all signed outcomes."""
import numpy as np
from concm_statistical_metrics import stage,summary

def diagnose(off,scores,x,head,heads,labels,d,task):
    seen=d['order'];y=np.array([seen.index(int(c)) for c in labels]);old=len(seen)-d['task_sizes'][-1]
    competitor=off.copy();competitor[np.arange(len(y)),y]=-np.inf;j=competitor.argmax(1)
    def geometric(w):
        diff=(w[:,y]-w[:,j]).T.astype(np.float64)
        return (x*diff).sum(1)/np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(diff,axis=1),1e-30)
    baseline=geometric(head);po=off.argmax(1);reports={}
    for v,on in scores.items():
        a=stage(off,on,on,labels,d,task);pn=on.argmax(1);g=geometric(heads[v]);groups=[('all',np.ones(len(y),bool)),('old',y<old),('new',y>=old)]+[(str(c),labels==c) for c in seen]
        a['normalized_fixed_competitor_margin']={n:dict(off=summary(baseline[m],n not in ('all','old','new') and m.sum()<5),on=summary(g[m],n not in ('all','old','new') and m.sum()<5),delta=summary((g-baseline)[m],n not in ('all','old','new') and m.sum()<5)) for n,m in groups}
        a['class_confusion']=[dict(label=c,competitor=q,off=int(((labels==c)&(po==ix)).sum()),on=int(((labels==c)&(pn==ix)).sum()),rescued_from=int(((labels==c)&(po==ix)&(pn==ci)&(po!=ci)).sum()),harmed_to=int(((labels==c)&(po==ci)&(pn==ix)&(pn!=ci)).sum())) for ci,c in enumerate(seen) for ix,q in enumerate(seen)]
        if v=='IDENTITY' and not np.array_equal(off,on):raise ValueError('Identity float readout differs')
        reports[v]=a
    return reports
