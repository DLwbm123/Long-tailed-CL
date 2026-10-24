import numpy as np

def report(stages,cfg,train):
    from collections import Counter
    counts=Counter(r['label'] for r in train);arms=cfg['arms'];a=cfg['primary_arm']
    groups={'tail4':[7,14,15,16],'tail7':[c for c,n in counts.items() if n<250],'initial10':cfg['order'][:10],'later10':cfg['order'][10:]}
    mean={b:float(np.mean([s['metrics'][b]['BA_percent'] for s in stages])) for b in arms}
    freq={b:{g:sum(stages[-1]['metrics'][b]['correct'][str(c)] for c in cs) for g,cs in groups.items()} for b in arms}
    final={b:stages[-1]['metrics'][b]['total_correct'] for b in arms}
    gates={'final':final[a]>=91,'mean':mean[a]>=36.005291005291,'tail7':freq[a]['tail7']>=25,'initial':freq[a]['initial10']>=44,'later':freq[a]['later10']>=41}
    for b in arms[:-1]:gates.update({b+'_final':final[a]>=final[b]+4,b+'_mean':mean[a]>=mean[b]+1,b+'_tail4':freq[a]['tail4']>=freq[b]['tail4']+2,b+'_later':freq[a]['later10']>=freq[b]['later10']})
    return dict(stages=stages,mean_BA_percent=mean,frequency=freq),dict(gates=gates,proceed=all(gates.values()),mechanism_LOCAL_over_GLOBAL=all(v for k,v in gates.items() if k.startswith('GLOBAL_REDUNDANCY_')),no_parameter_rescue=True)
