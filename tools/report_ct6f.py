"""Saved-val-score-only full horizon comparison; no learner or image access."""
import os,json,time,hashlib,sys
from pathlib import Path
import numpy as np
from ct2d_math import metrics
from report_locked_holdout_r1 import csvwrite,bootstrap_weights,boot_unit
from threadpoolctl import threadpool_limits

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def recall_groups(rec,known,order,tail):
    old=rec[:,:known].mean(1) if known else None;cur=rec[:,known:].mean(1);ix=np.isin(order,tail)
    return dict(BA=rec.mean(1),old=old,current=cur,tail=rec[:,ix].mean(1) if ix.any() else None,HM=None if old is None else np.divide(2*old*cur,old+cur,out=np.zeros(len(rec)),where=old+cur!=0))

def stage_order(p,seen):
    # CT1 serializes the full arrival order; CT4/5 serialize its seen prefix.
    assert p['raw'].shape==(len(p['y']),seen) and len(p['order'])>=seen
    assert len(np.unique(p['order']))==len(p['order'])
    order=p['order'][:seen]
    assert np.all((p['y']>=0)&(p['y']<seen))
    assert np.array_equal(order[p['y']],p['original']),'BLOCKED_LABEL_MAPPING'
    return order

def main():
    start=time.monotonic();cfg=read(os.environ['P25_CONFIG']);root=Path(cfg['root']);pub=root/'output/public';old=Path(cfg['ct1_root'])/'output';data={};point={};rows=[];pcs=[];errors=[];entries={}
    lock=read(pub/'PROTOCOL_LOCK.json');assert sha(__file__)==lock['report_sha256']
    assert read(pub/'INFERENCE_COMPLETE.json')['units']==90
    # This isolated reporter may only read locked predictions, never images or fitting assets.
    allowed={str(p.resolve()) for p in (root/'output/private/sealed').glob('*.npz')}
    references=read(old/'public/TRAJECTORIES_LOCK.json')['sealed_units']
    references=[dict(e,method={'PT-CB':'P','CT-J-CB':'C0'}[e['method']],source_method=e['method']) for e in references if e['method'] in ('PT-CB','CT-J-CB')]
    frozen=Path(cfg['ct5f_root'])/'output';frozen_units=read(frozen/'public/PREDICTIONS_LOCK.json')['units']
    allowed.update(str((old/'private/sealed'/e['file']).resolve()) for e in references)
    allowed.update(str((frozen/'private/sealed'/e['file']).resolve()) for e in frozen_units)
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0]));s=str(p.resolve());flags=args[2] if len(args)>2 else 0
        if p.suffix.lower() in ('.npz','.npy','.pt','.jpg','.jpeg','.png'):assert s in allowed,'BLOCKED_REPORT_ASSET'
        if any(x.lower().startswith(('test','reserved')) for x in p.parts):raise AssertionError('BLOCKED_TEST')
        if flags and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):assert p.resolve().is_relative_to(pub.resolve()),'BLOCKED_REPORT_WRITE'
    sys.addaudithook(guard)
    for directory,units in [(root/'output',read(pub/'PREDICTIONS_LOCK.json')['units']),(old,references),(frozen,frozen_units)]:
        for e in units:
            path=directory/'private/sealed'/e['file'];assert sha(path)==e['sha256']
            with np.load(path) as f:p={k:f[k].copy() for k in f.files}
            key=(e['dataset'],e['seed'],e['task'],e['method']);assert key not in data
            p['order']=stage_order(p,e['seen']);data[key]=p;entries[key]=e
            m,pc,er=metrics(p['raw'],p['y'],p['order'],e['known'],cfg['frequency_groups'][e['dataset']],dict(zip(('dataset','seed','task','method'),key)),p['component'])
            point[key]=m;rows.append(m);pcs+=pc;errors+=er
    assert len(rows)==225 and len(pcs)==2295
    assert sum(x['method'] in ('C2','C3') for x in rows)==90 and sum(x['method'] in ('C2','C3') for x in pcs)==918
    out=[];audit=[]
    for ds,last in [('HK',11),('ISIC',4)]:
        identity={}
        for key,p in data.items():
            if key[0]!=ds:continue
            for sid,label,comp in zip(p['ids'],p['original'],p['component']):
                val=(int(label),str(comp));assert sid not in identity or identity[sid]==val;identity[sid]=val
        ids=sorted(identity);ix={x:i for i,x in enumerate(ids)};canon=dict(y=np.zeros(len(ids),int),original=np.array([identity[x][0] for x in ids]),component=np.array([identity[x][1] for x in ids]))
        w=bootstrap_weights(canon,2000,50001,sorted(set(canon['original'])));boot={}
        for key,p in data.items():
            if key[0]!=ds:continue
            rec=boot_unit(p,w[:,[ix[x] for x in p['ids']]],True);assert np.isfinite(rec).all()
            boot[key]=recall_groups(rec,entries[key]['known'],p['order'],cfg['frequency_groups'][ds]['tail'])
        # Full PT-CB is one deterministic reference despite three arrival orders.
        ref=[data[ds,s,last,'P'] for s in (1993,1994,1995)]
        from ct2d_math import classify
        preds=[p['order'][classify(p['raw'],p['order'])] for p in ref]
        assert all(np.array_equal(preds[0],p) for p in preds[1:])
        audit.append(dict(dataset=ds,PT_final_prediction_order_invariant=True,n_val=len(ids)))
        for task in range(1,last+1):
            for a,b in [('C3','F1'),('C2','F1'),('C3','C2'),('C3','P'),('C3','C0')]:
                for metric,col in [('BA','balanced_accuracy'),('old','old_macro_recall'),('current','current_macro_recall'),('HM','HM'),('tail','tail_recall')]:
                    samples=[];values=[]
                    for seed in (1993,1994,1995):
                        ka=(ds,seed,task,a);kb=(ds,seed,task,b)
                        for k in ('ids','y','order','original','component'):assert np.array_equal(data[ka][k],data[kb][k])
                        assert entries[ka]['known']==entries[kb]['known']
                        va=point[ka][col];vb=point[kb][col]
                        if va is None or vb is None:delta=lo=hi=None
                        else:
                            delta=va-vb;s=boot[ka][metric]-boot[kb][metric];lo,hi=map(float,np.quantile(s,[.025,.975]));samples.append(s);values.append(delta)
                        out.append(dict(dataset=ds,seed=seed,task=task,contrast=a+'-'+b,metric=metric,difference_pp=delta,low=lo,high=hi))
                    if len(values)==3:
                        lo,hi=map(float,np.quantile(np.mean(samples,axis=0),[.025,.975]));out.append(dict(dataset=ds,seed='fixed_three_mean',task=task,contrast=a+'-'+b,metric=metric,difference_pp=float(np.mean(values)),low=lo,high=hi))
    final=[];forget=[]
    for ds,last in [('HK',11),('ISIC',4)]:
        for seed in (1993,1994,1995):
            for method in ('C2','C3','F1','P','C0'):
                m=dict(point[ds,seed,last,method]);m['AvgBA_all_tasks']=float(np.mean([point[ds,seed,t,method]['balanced_accuracy'] for t in range(1,last+1)]));final.append(m)
                labels=sorted({r['original_label'] for r in pcs if (r['dataset'],r['seed'],r['method'])==(ds,seed,method)})
                for label in labels:
                    rs=sorted([r for r in pcs if (r['dataset'],r['seed'],r['method'],r['original_label'])==(ds,seed,method,label)],key=lambda r:r['task']);eligible=len(rs)>1
                    forget.append(dict(dataset=ds,seed=seed,method=method,original_label=label,eligible=eligible,first_task=rs[0]['task'],first_recall=rs[0]['recall'],final_recall=rs[-1]['recall'],peak_minus_final=max(r['recall'] for r in rs)-rs[-1]['recall'] if eligible else None,first_minus_final=rs[0]['recall']-rs[-1]['recall'] if eligible else None))
    for filename,table in [('metrics',[r for r in rows if r['method'] in ('C2','C3')]),('per_class',[r for r in pcs if r['method'] in ('C2','C3')]),('comparison_metrics',rows),('comparison_per_class',pcs),('paired_all_metrics',out),('final_metrics',final),('forgetting',forget),('errors',errors)]:csvwrite(pub/(filename+'.csv'),table)
    text=['# CT6-F 完整任务范围FD续训','', '沿用三种顺序，HK为完整23类11任务，ISIC为完整8类4任务。继承六个CT3-P Task3父状态，新增270task-epochs/5570steps；仅val，test/reserved访问=0。','', '| 数据 | 方法 | Final BA均值 | 最差seed | 全任务平均BA |','|---|---|---:|---:|---:|']
    for ds in ('HK','ISIC'):
        for method in ('C2','C3','F1','P','C0'):
            ms=[r for r in final if r['dataset']==ds and r['method']==method];text.append(f"|{ds}|{method}|{np.mean([r['balanced_accuracy'] for r in ms]):.3f}|{min(r['balanced_accuracy'] for r in ms):.3f}|{np.mean([r['AvgBA_all_tasks'] for r in ms]):.3f}|")
    for r in out:
        if r['seed']=='fixed_three_mean' and r['metric']=='BA' and r['task']==(11 if r['dataset']=='HK' else 4):text.append(f"- {r['dataset']} {r['contrast']}: {r['difference_pp']:+.3f}pp [{r['low']:+.3f}, {r['high']:+.3f}]。")
    text+=['','以上为完整固定FD观察结果，不预设取得正结果。区间只条件于现有模型和val component；未重采样seed，P终端是确定性参照，singleton和反复使用val的限制仍在，多重比较未校正。36个C2/C3前缀预测原样复用，54个新阶段预测来自27个新任务末checkpoint，未重训Task1–3；不存在旧fit回放。','', '所有逐阶段/逐类、old/current/HM/tail、零召回、误分及遗忘见配套CSV。NEXT_DECISION=STOP。']
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(text))
    elapsed=time.monotonic()-start;assert elapsed<1800
    (pub/'ANALYSIS_AUDIT.json').write_text(json.dumps(dict(status='PASS',saved_scores_only=True,new_forwards=0,new_fits=0,rows=225,per_class=2295,bootstrap_n=2000,seed=50001,CPU_report_seconds=elapsed,checks=audit),indent=2))
    (pub/'NEXT_DECISION.md').write_text('STOP\n完整固定矩阵完成；不自动扩展本轮。\n')
    (pub/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE_CT6F',NEXT_DECISION='STOP',units=90,per_class=918,neural_epochs=270,optimizer_steps=5570,test_predictions=0),indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
