"""Saved-score-only Task4 beta perturbation comparisons; no model imports."""
import os,json,time,sys,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ct2d_math import metrics
from report_locked_holdout_r1 import csvwrite,bootstrap_weights,boot_unit
from report_ct6f import recall_groups,stage_order

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    start=time.monotonic();cfg=read(os.environ['P30_CONFIG']);root=Path(cfg['root']);pub=root/'output/public'
    lock=read(pub/'DIAGNOSTIC_LOCK.json');assert sha(__file__)==lock['report_sha256']
    for path,digest in lock['references'].items():assert sha(path)==digest
    new=read(pub/'PREDICTIONS_LOCK.json')['units'];assert len(new)==12
    sets=[(root/'output',new)]
    for key,methods in [('ct6f_root',('C2','C3')),('ct9p_root',('B1A','B1T'))]:
        d=Path(cfg[key])/'output';units=[e for e in read(d/'public/PREDICTIONS_LOCK.json')['units'] if e['task']==4 and e['method'] in methods]
        assert len(units)==3*2*len(methods);sets.append((d,units))
    allowed={str((d/'private/sealed'/e['file']).resolve()) for d,units in sets for e in units}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
        if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):assert p.is_relative_to(pub.resolve())
        elif p.suffix.lower() in ('.npz','.npy','.pt','.jpg','.jpeg','.png','.csv'):assert str(p) in allowed
    sys.addaudithook(guard)
    data={};entries={};point={};rows=[];pcs=[];errors=[]
    for d,units in sets:
        for e in units:
            path=d/'private/sealed'/e['file'];assert sha(path)==e['sha256']
            with np.load(path,allow_pickle=False) as f:p={k:f[k].copy() for k in f.files}
            assert e['seen']==8 and e['known']==6 and e['task']==4
            p['order']=stage_order(p,8);key=(e['dataset'],e['seed'],e['method']);assert key not in data
            m,pc,er=metrics(p['raw'],p['y'],p['order'],6,cfg['frequency_groups'][e['dataset']],dict(dataset=e['dataset'],seed=e['seed'],task=4,method=e['method']),p['component'])
            data[key]=p;entries[key]=e;point[key]=m;rows.append(m);pcs+=pc;errors+=er
    assert len(rows)==36 and len(pcs)==288
    contrasts=[('N10','C3'),('N1','B1T'),('N10','C2'),('N1','B1A'),('N1','N10')];paired=[]
    for ds in ('HK','ISIC'):
        identity={}
        for key,p in data.items():
            if key[0]!=ds:continue
            for sid,label,component in zip(p['ids'],p['original'],p['component']):
                v=(int(label),str(component));assert sid not in identity or identity[sid]==v;identity[sid]=v
        ids=sorted(identity);ix={v:i for i,v in enumerate(ids)}
        canon=dict(y=np.zeros(len(ids),int),original=np.array([identity[v][0] for v in ids]),component=np.array([identity[v][1] for v in ids]))
        weights=bootstrap_weights(canon,2000,54001,sorted(set(canon['original'])));boot={}
        for key,p in data.items():
            if key[0]==ds:
                rec=boot_unit(p,weights[:,[ix[v] for v in p['ids']]],True)
                boot[key]=recall_groups(rec,6,p['order'],cfg['frequency_groups'][ds]['tail'])
        for a,b in contrasts:
            for metric,col in [('BA','balanced_accuracy'),('old','old_macro_recall'),('current','current_macro_recall'),('tail','tail_recall'),('HM','HM')]:
                values=[];samples=[]
                for seed in (1993,1994,1995):
                    ka=(ds,seed,a);kb=(ds,seed,b)
                    for k in ('ids','order','original','y','component'):assert np.array_equal(data[ka][k],data[kb][k])
                    va,vb=point[ka][col],point[kb][col]
                    if va is None or vb is None:delta=lo=hi=None
                    else:
                        delta=va-vb;s=boot[ka][metric]-boot[kb][metric];lo,hi=map(float,np.quantile(s,[.025,.975]));samples.append(s);values.append(delta)
                    paired.append(dict(dataset=ds,seed=seed,task=4,contrast=a+'-'+b,metric=metric,difference_pp=delta,low=lo,high=hi))
                if len(values)==3:
                    lo,hi=map(float,np.quantile(np.mean(samples,axis=0),[.025,.975]))
                    paired.append(dict(dataset=ds,seed='fixed_three_mean',task=4,contrast=a+'-'+b,metric=metric,difference_pp=float(np.mean(values)),low=lo,high=hi))
    for name,rs in [('metrics',rows),('per_class',pcs),('paired',paired),('errors',errors)]:csvwrite(pub/(name+'.csv'),rs)
    report=['# CT11-H Task4固定读出诊断','', '本轮新增训练为0，固定CT6 beta10与CT9 Task4 beta1末状态，比较同一pointwise eval probe的神经main+few与既有解析读出。HK仅8/23前缀；ISIC8类。仅val，test访问0，不用更好的head替换正式结果。','', '|数据|方法|Task4 BA|最差seed|old|current|tail|','|---|---|---:|---:|---:|---:|---:|']
    for ds in ('HK','ISIC'):
        for method in ('N10','N1','C2','C3','B1A','B1T'):
            q=[x for x in rows if x['dataset']==ds and x['method']==method]
            vals=[(np.mean([x[k] for x in q]) if all(x[k] is not None for x in q) else float('nan')) for k in ['balanced_accuracy','old_macro_recall','current_macro_recall','tail_recall']]
            report.append('|'+ '|'.join([ds,method,f'{vals[0]:.3f}',f'{min(x["balanced_accuracy"] for x in q):.3f}']+[f'{v:.3f}' for v in vals[1:]])+'|')
    for r in paired:
        if r['seed']=='fixed_three_mean' and r['metric']=='BA':report.append(f"- {r['dataset']} {r['contrast']}: {r['difference_pp']:+.3f}pp [{r['low']:+.3f}, {r['high']:+.3f}]。")
    report+=['','2000次seed54001按类component配对区间，固定模型，不重抽seed，多重比较未校正。HK各seed前8类不同，配对只在同seed同布局，不能当同一八类的三次训练；不把结果外推完整23类或其他FD强度。NEXT_DECISION=STOP。']
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(report));(pub/'NEXT_DECISION.md').write_text('STOP\n固定十二状态读出诊断完成，不训练、改路由或选最佳head。\n')
    ledger=read(pub/'RESOURCE_LEDGER.json');assert ledger['fit_image_reads']==0 and ledger['val_image_reads']==5464 and not ledger['calls'].get('formal_optimizer_steps',0)
    (pub/'ANALYSIS_AUDIT.json').write_text(json.dumps(dict(status='PASS',rows=36,per_class=288,new_forwards=0,new_fits=0,bootstrap_n=2000,bootstrap_seed=54001,CPU_seconds=time.monotonic()-start),indent=2))
    (pub/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE_CT11H',NEXT_DECISION='STOP',neural_epochs=0,steps=0,new_checkpoints=0,new_prediction_units=12,test_predictions=0),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
