import csv,json,os,time
from pathlib import Path
import numpy as np
import torch

def config():
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    torch.set_num_threads(4)
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    return cfg

def write(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def records(cfg,classes):
    result=[]
    for c in classes:
        r=torch.load(Path(cfg['source_root'])/f'class_{c}.private.pt',map_location='cpu',weights_only=False)
        s=r['FROZEN_NATIVE'];m=np.sqrt(2)*s['mean'].numpy()[:-1]
        S=2*s['gram'].numpy()[:-1,:-1]-np.outer(m,m)
        assert r['label']==c and s['online_independent_error']<1e-9
        assert abs(float(s['gram'][-1,-1])-.5)<1e-9
        result.append(dict(label=c,patients=r['patients'],images=r['images'],mean=m,covariance=(S+S.T)/2,
                           proof=s['proof_features'].numpy()))
    return result

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def queries(cfg):
    # Called only after all new readouts have been sealed.
    q=torch.load(Path(cfg['source_root'])/'query_features.private.pt',map_location='cpu',weights_only=False)
    return unit(q['FROZEN_NATIVE'].numpy())

def metadata(cfg):
    rows={}
    for split,n in [('train',68058),('val',300)]:
        with (Path(cfg['manifest'])/(split+'.csv')).open() as f:rr=list(csv.DictReader(f))
        with (Path(cfg['author_labels'])/(split+'.csv')).open() as f:author={r['id']:r for r in csv.DictReader(f)}
        assert len(rr)==n
        for r in rr:
            name=Path(r['relative_path']).name;r['label']=int(r['original_label']);r['patient']=str(int(author[name]['subject_id']))
            assert r['patient']==str(int(name.split('_')[0])) and r['split']==split
        rows[split]=rr
    assert not {r['patient'] for r in rows['train']}&{r['patient'] for r in rows['val']}
    return rows

def metrics(score,seen,y):
    mask=np.isin(y,seen);pred=np.array(seen)[score.argmax(1)]
    correct={str(c):int(((pred==c)&(y[mask]==c)).sum()) for c in seen}
    return dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/int(mask.sum()))

def report(stages,cfg,train):
    from collections import Counter
    counts=Counter(r['label'] for r in train);arms=cfg['arms'];a=cfg['primary_arm']
    groups={'tail4':[7,14,15,16],'tail7':[c for c,n in counts.items() if n<250],'initial10':cfg['order'][:10],'later10':cfg['order'][10:]}
    mean={b:float(np.mean([s['metrics'][b]['BA_percent'] for s in stages])) for b in arms}
    freq={b:{g:sum(stages[-1]['metrics'][b]['correct'][str(c)] for c in cs) for g,cs in groups.items()} for b in arms}
    final={b:stages[-1]['metrics'][b]['total_correct'] for b in arms}
    gates={'final':final[a]>=91,'mean':mean[a]>=36.005291005291,'tail7':freq[a]['tail7']>=25,'initial':freq[a]['initial10']>=44,'later':freq[a]['later10']>=41}
    for b in arms[:-1]:gates.update({b+'_final':final[a]>=final[b]+4,b+'_mean':mean[a]>=mean[b]+1,b+'_tail4':freq[a]['tail4']>=freq[b]['tail4']+2,b+'_later':freq[a]['later10']>=freq[b]['later10']})
    return dict(stages=stages,mean_BA_percent=mean,frequency=freq),dict(gates=gates,proceed=all(gates.values()),mechanism_FULL_over_DIAGONAL=all(v for k,v in gates.items() if k.startswith('DIAGONAL_')),no_parameter_rescue=True)
