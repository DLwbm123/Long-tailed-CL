"""Complete the prespecified paired metric analysis using locked scores only."""
import os,json
from pathlib import Path
import numpy as np
from run_ct4f import read,sha,metrics,csvwrite,bootstrap_weights,boot_unit
from threadpoolctl import threadpool_limits


def main():
    cfg=read(os.environ['P23_CONFIG']);root=Path(cfg['root']);pub=root/'output/public';ref=Path(cfg['ct3p_root'])/'output';data={};point={};rows=[];pcs=[]
    for directory,filename,methods in [(root/'output','PREDICTIONS_LOCK.json',{'F1'}),(ref,'ONLINE_PREDICTIONS_LOCK.json',{'P','C0','C2','C3'})]:
        for e in read(directory/'public'/filename)['units']:
            if e['method'] not in methods:continue
            path=directory/'private/sealed'/e['file'];assert sha(path)==e['sha256']
            with np.load(path) as f:p={k:f[k].copy() for k in f.files}
            key=(e['dataset'],e['seed'],e['task'],e['method']);data[key]=p
            m,pc,_=metrics(p['raw'],p['y'],p['order'],e['seen']-2,cfg['frequency_groups'][e['dataset']],dict(zip(('dataset','seed','task','method'),key)),p['component']);point[key]=m;rows.append(m);pcs+=pc
    out=[]
    for ds in ('HK','ISIC'):
        identity={}
        for key,p in data.items():
            if key[0]!=ds:continue
            for sid,label,comp in zip(p['ids'],p['original'],p['component']):
                val=(int(label),str(comp));assert sid not in identity or identity[sid]==val;identity[sid]=val
        ids=sorted(identity);ix={x:i for i,x in enumerate(ids)};canon=dict(y=np.zeros(len(ids),int),original=np.array([identity[x][0] for x in ids]),component=np.array([identity[x][1] for x in ids]))
        w=bootstrap_weights(canon,2000,48001,sorted(set(canon['original'])));boot={}
        for key,p in data.items():
            if key[0]!=ds:continue
            rec=boot_unit(p,w[:,[ix[x] for x in p['ids']]],True);C=len(p['order']);old=rec[:,:C-2].mean(1) if C>2 else None;cur=rec[:,C-2:].mean(1);tail=np.isin(p['order'],cfg['frequency_groups'][ds]['tail'])
            boot[key]=dict(BA=rec.mean(1),old=old,current=cur,tail=rec[:,tail].mean(1) if tail.any() else None,HM=None if old is None else np.divide(2*old*cur,old+cur,out=np.zeros(2000),where=old+cur!=0))
        for task in (1,2,3):
            for a,b in [('C3','F1'),('C2','F1'),('F1','P'),('F1','C0')]:
                for metric,col in [('BA','balanced_accuracy'),('old','old_macro_recall'),('current','current_macro_recall'),('HM','HM'),('tail','tail_recall')]:
                    samples=[];values=[]
                    for seed in (1993,1994,1995):
                        ka=(ds,seed,task,a);kb=(ds,seed,task,b)
                        for k in ('ids','y','order','component'):assert np.array_equal(data[ka][k],data[kb][k])
                        va=point[ka][col];vb=point[kb][col]
                        if va is None or vb is None:delta=lo=hi=None
                        else:
                            delta=va-vb;s=boot[ka][metric]-boot[kb][metric];lo,hi=map(float,np.quantile(s,[.025,.975]));samples.append(s);values.append(delta)
                        out.append(dict(dataset=ds,seed=seed,task=task,contrast=a+'-'+b,metric=metric,difference_pp=delta,low=lo,high=hi))
                    if len(values)==3:
                        lo,hi=map(float,np.quantile(np.mean(samples,axis=0),[.025,.975]));out.append(dict(dataset=ds,seed='fixed_three_mean',task=task,contrast=a+'-'+b,metric=metric,difference_pp=float(np.mean(values)),low=lo,high=hi))
    # Reproduce the original BA analysis exactly before adding the remaining metrics.
    import csv
    prior=list(csv.DictReader((pub/'paired_BA_intervals.csv').open()))
    for r in prior:
        q=next(q for q in out if q['metric']=='BA' and all(str(q[k])==r[k] for k in ('dataset','seed','task','contrast')))
        for k in ('difference_pp','low','high'):assert abs(q[k]-float(r[k]))<1e-10
    csvwrite(pub/'comparison_metrics.csv',rows);csvwrite(pub/'comparison_per_class.csv',pcs);csvwrite(pub/'paired_all_metrics.csv',out)
    (pub/'ANALYSIS_AUDIT.json').write_text(json.dumps(dict(status='PASS',saved_scores_only=True,new_forwards=0,new_fits=0,BA_reproduction_rows=len(prior),bootstrap_n=2000,seed=48001),indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
