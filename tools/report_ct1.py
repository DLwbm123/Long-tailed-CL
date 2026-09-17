"""CT1 CPU-only unsealing, fixed paired component bootstrap and final report."""
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from report_locked_holdout_r1 import (read,write,csvwrite,predict_columns,
                                     bootstrap_weights,boot_unit)
from isic_g1_controlled_mixup import sha

METHODS=('CT-J-CB','CT-Risk','CT-ConCM','PT-CB','CT-J-Stale','CT-ConCM-CB')
PAIRS=(('CT-Risk','CT-J-CB'),('CT-ConCM','CT-J-CB'),('CT-ConCM-CB','CT-J-CB'),
       ('CT-J-CB','CT-J-Stale'),('CT-J-CB','PT-CB'),('CT-Risk','PT-CB'),('CT-ConCM','PT-CB'))


def mean(x):return float(np.mean(x)) if len(x) else None
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def analyze(p,entry,groups):
    y=p['y'];order=p['order'];scores=p['raw'];n=entry['seen'];known=entry['known']
    assert scores.shape==(len(y),n) and np.isfinite(scores).all()
    pred=predict_columns(scores,order);correct=pred==y
    prefix={k:entry[k] for k in ('dataset','seed','task','method','seen','known')}
    pc=[]
    for c in range(n):
        mask=y==c;assert mask.any()
        alternatives=scores[mask].copy();alternatives[:,c]=-np.inf
        margin=scores[mask,c]-alternatives.max(1)
        pc.append(dict(**prefix,original_label=int(order[c]),n_images=int(mask.sum()),
            n_components=len(np.unique(p['component'][mask])),n_correct=int(correct[mask].sum()),
            recall=100*float(correct[mask].mean()),margin_mean=float(margin.mean()),
            current_class=c>=known,first_task_class=c<2))
    recalls=np.array([x['recall'] for x in pc]);old=mean(recalls[:known]);current=mean(recalls[known:])
    m=dict(**prefix,balanced_accuracy=float(recalls.mean()),accuracy=100*float(correct.mean()),
           old_macro_recall=old,current_macro_recall=current,HM=None if old is None else (2*old*current/(old+current) if old+current else 0.),
           worst_class_recall=float(recalls.min()),zero_recall_classes=[int(order[c]) for c in range(n) if recalls[c]==0])
    for group,labels in groups.items():m[group+'_recall']=mean([recalls[c] for c in range(n) if order[c] in labels])
    errors=[]
    for scope,cs in [('old',list(range(known))),('current',list(range(known,n)))]:
        mask=np.isin(y,cs);same=np.isin(pred,cs)
        er=dict(**prefix,scope=scope,n_images=int(mask.sum()))
        for key,event in [('correct',correct),('to_other',~same),('wrong_within',same&~correct)]:
            er[key+'_macro']=100*mean([event[y==c].mean() for c in cs]) if cs else None
            er[key+'_sample']=100*float(event[mask].mean()) if cs else None
        if cs:
            assert abs(sum(er[k+'_macro'] for k in ('correct','to_other','wrong_within'))-100)<1e-9
        errors.append(er)
    return m,pc,errors


def main():
    cfg=read(os.environ['P20_CONFIG']);root=Path(cfg['root']);pub=root/'output/public';private=root/'output/private'
    started=time.monotonic();lock=read(pub/'TRAJECTORIES_LOCK.json');assert len(lock['checkpoints'])==90
    data={};metrics=[];classes=[];errors=[];groups=cfg['frequency_groups']
    for e in lock['sealed_units']:
        p=private/'sealed'/e['file'];assert sha(p)==e['sha256'],'BLOCKED_SEALED_SCORE_DRIFT'
        with np.load(p) as z:d={k:z[k].copy() for k in z.files}
        key=(e['dataset'],e['method'],e['seed'],e['task']);assert key not in data
        data[key]=d;m,pc,er=analyze(d,e,groups[e['dataset']]);metrics.append(m);classes+=pc;errors+=er
    complete=not lock['risk_failures']
    if complete:
        assert len(metrics)==270 and len(classes)==2754
        assert len([m for m in metrics if m['method'] in METHODS[:3]])==135
        assert len([m for m in classes if m['method'] in METHODS[:3]])==1377
    write(pub/'VAL_RELEASE_CT1.json',dict(status='RELEASED',source_commit=cfg['source_commit'],
         locked_units=len(metrics),complete_matrix=complete,new_test_predictions=0))
    csvwrite(pub/'validation_all_metrics.csv',metrics);csvwrite(pub/'validation_all_per_class.csv',classes)
    csvwrite(pub/'validation_main_metrics.csv',[m for m in metrics if m['method'] in METHODS[:3]])
    csvwrite(pub/'validation_main_per_class.csv',[m for m in classes if m['method'] in METHODS[:3]])
    csvwrite(pub/'error_decomposition.csv',errors)
    train=lines(pub/'train_epoch_metrics.jsonl');assert len(train)==900
    csvwrite(pub/'train_epoch_metrics.csv',train)
    for name in ('transport_audit','risk_solver_audit'):csvwrite(pub/(name+'.csv'),lines(pub/(name+'.jsonl')))
    summaries=[];forget=[];paired=[];intervals=[];means=[];boots={};point={}
    for dataset,last in [('HK',11),('ISIC',4)]:
        canonical=data[dataset,'PT-CB',1993,last];idmap={s:i for i,s in enumerate(canonical['ids'])}
        weights=bootstrap_weights(canonical,resamples=2000,seed=45001,labels=range(23 if dataset=='HK' else 8))
        for method in METHODS:
            method_rows=[m for m in metrics if m['dataset']==dataset and m['method']==method]
            for seed in (1993,1994,1995):
                ms=sorted([m for m in method_rows if m['seed']==seed],key=lambda x:x['task'])
                if len(ms)!=last:continue
                f=ms[-1];summary=dict(dataset=dataset,method=method,seed=seed,Final_BA=f['balanced_accuracy'],
                    Average_BA=mean([m['balanced_accuracy'] for m in ms]),Final_tail=f['tail_recall'],
                    Final_old=f['old_macro_recall'],Final_current=f['current_macro_recall'],Final_HM=f['HM'],
                    zero_recall_classes=f['zero_recall_classes'])
                summaries.append(summary)
                for m in ms:
                    k=(dataset,method,seed,m['task']);p=data[k];ix=np.array([idmap[s] for s in p['ids']])
                    assert np.array_equal(canonical['original'][ix],p['original'])
                    assert np.array_equal(canonical['component'][ix],p['component'])
                    rr=boot_unit(p,weights[:,ix],tie_original_label=True)
                    tail=[j for j in range(m['seen']) if p['order'][j] in groups[dataset]['tail']]
                    old=rr[:,:m['known']].mean(1) if m['known'] else np.full(2000,np.nan)
                    cur=rr[:,m['known']:].mean(1)
                    boots[k]=dict(BA=rr.mean(1),tail=rr[:,tail].mean(1) if tail else np.full(2000,np.nan),old=old,current=cur,
                                  HM=np.divide(2*old*cur,old+cur,out=np.zeros(2000),where=old+cur!=0))
                    point[k]=dict(BA=m['balanced_accuracy'],tail=m['tail_recall'],old=m['old_macro_recall'],current=m['current_macro_recall'],HM=m['HM'])
                pcs=[c for c in classes if c['dataset']==dataset and c['method']==method and c['seed']==seed]
                for c in range(23 if dataset=='HK' else 8):
                    history=sorted([q for q in pcs if q['original_label']==c],key=lambda q:q['task'])
                    values=[q['recall'] for q in history]
                    forget.append(dict(dataset=dataset,method=method,seed=seed,original_label=c,
                        arrival_task=history[0]['task'],first_quality=values[0],final_quality=values[-1],
                        signed_first_to_final=None if len(values)==1 else values[0]-values[-1],
                        max_observed_to_final=None if len(values)==1 else max(values)-values[-1]))
            rows=[s for s in summaries if s['dataset']==dataset and s['method']==method]
            if len(rows)==3:
                row=dict(dataset=dataset,method=method,deterministic_final_reference=method=='PT-CB',worst_run_BA=min(s['Final_BA'] for s in rows))
                for k in ('Final_BA','Average_BA','Final_tail','Final_old','Final_current','Final_HM'):
                    row[k+'_mean']=mean([s[k] for s in rows]);row[k+'_sd']=float(np.std([s[k] for s in rows],ddof=1))
                means.append(row)
        for a,b in PAIRS:
            for stage in range(1,last+1):
                if not all((dataset,m,seed,stage) in boots for m in (a,b) for seed in (1993,1994,1995)):continue
                for metric in ('BA','tail','old','current','HM'):
                    values=[];rep=[]
                    for seed in (1993,1994,1995):
                        ka=(dataset,a,seed,stage);kb=(dataset,b,seed,stage)
                        if point[ka][metric] is None or point[kb][metric] is None:break
                        delta=point[ka][metric]-point[kb][metric];values.append(delta);rep.append(boots[ka][metric]-boots[kb][metric])
                        za=set(next(m for m in metrics if (m['dataset'],m['method'],m['seed'],m['task'])==ka)['zero_recall_classes'])
                        zb=set(next(m for m in metrics if (m['dataset'],m['method'],m['seed'],m['task'])==kb)['zero_recall_classes'])
                        paired.append(dict(dataset=dataset,comparison=a+' minus '+b,seed=seed,task=stage,metric=metric,
                                           difference_pp=delta,new_zero_recall_classes=sorted(za-zb)))
                    if len(values)==3:
                        d=np.mean(rep,axis=0);lo,hi=np.quantile(d,[.025,.975])
                        intervals.append(dict(dataset=dataset,comparison=a+' minus '+b,task=stage,metric=metric,
                            mean_difference_pp=mean(values),paired_sd=float(np.std(values,ddof=1)),low=float(lo),high=float(hi),
                            resamples=2000,bootstrap_seed=45001,fixed_models_not_resampled=True))
            if all((dataset,m,seed,t) in boots for m in (a,b) for seed in (1993,1994,1995) for t in range(1,last+1)):
                d=np.mean([np.mean([boots[dataset,a,s,t]['BA']-boots[dataset,b,s,t]['BA'] for t in range(1,last+1)],axis=0)
                           for s in (1993,1994,1995)],axis=0)
                pv=[mean([point[dataset,a,s,t]['BA']-point[dataset,b,s,t]['BA'] for t in range(1,last+1)]) for s in (1993,1994,1995)]
                lo,hi=np.quantile(d,[.025,.975]);intervals.append(dict(dataset=dataset,comparison=a+' minus '+b,
                    task='all',metric='Average_BA',mean_difference_pp=mean(pv),paired_sd=float(np.std(pv,ddof=1)),
                    low=float(lo),high=float(hi),resamples=2000,bootstrap_seed=45001,fixed_models_not_resampled=True))
        # Same PT feature set/target, not three independently trained references.
        finals=[data[dataset,'PT-CB',s,last] for s in (1993,1994,1995)]
        predictions=[p['order'][predict_columns(p['raw'],p['order'])] for p in finals]
        assert all(np.array_equal(predictions[0],x) for x in predictions[1:]),'BLOCKED_PT_FINAL_ORDER_PARITY'
        assert time.monotonic()-started<21600,'BLOCKED_CPU_BUDGET'
    for file,rows in [('summary_by_run.csv',summaries),('summary_mean.csv',means),('forgetting.csv',forget),
                      ('paired_differences.csv',paired),('bootstrap_intervals.csv',intervals)]:csvwrite(pub/file,rows)
    risks=[]
    tasks=read(pub/'DATA_AND_TASK_LOCK.json')
    for r in lines(pub/'risk_solver_audit.jsonl'):
        if r['status']!='optimal':continue
        order=tasks[r['dataset']]['orders'][str(r['seed'])]
        for j,xi in enumerate(r['xi']):
            pc=next(c for c in classes if c['dataset']==r['dataset'] and c['seed']==r['seed'] and
                    c['task']==r['task'] and c['method']=='CT-Risk' and c['original_label']==order[j])
            risks.append(dict(dataset=r['dataset'],seed=r['seed'],task=r['task'],original_label=order[j],
                         slack=xi,variance_proxy=r['v_tilde_mean_by_class'][j],uncertainty_proxy=r['u_mean_by_class'][j],
                         error_rate=100-pc['recall'],n_val=pc['n_images'],n_components=pc['n_components']))
    if risks:csvwrite(pub/'risk_proxy_and_errors.csv',risks)
    resource=read(pub/'RESOURCE_LEDGER.json');access=read(pub/'ACCESS_formal.json')
    audit=dict(status='COMPLETE_CT1' if complete else 'PARTIAL_SOLVER',formal_trajectories=12,task_checkpoints=90,
               neural_task_epochs=900,neural_optimizer_steps=access['formal_optimizer_steps'],
               main_metric_rows=len([m for m in metrics if m['method'] in METHODS[:3]]),
               all_metric_rows=len(metrics),all_per_class_rows=len(classes),separate_S0=False,
               new_test_image_reads=0,new_test_model_forwards=0,new_test_feature_reads=0,new_test_predictions=0,
               old_train_image_reads=0,future_train_image_reads=0,validation_adaptively_reused=True,
               independent_confirmation=False,risk_failures=lock['risk_failures'],CPU_report_seconds=time.monotonic()-started,
               resource=resource,access=access)
    write(pub/'MEMORY_ACCESS_AND_RESOURCE_AUDIT.json',audit)
    write(pub/'training_step_audit.json',dict(expected=32340,actual=access['formal_optimizer_steps'],
                                           epoch_rows=900,all_task_steps_checked=True))
    assert access['formal_optimizer_steps']==32340
    text=['# CT1 无独立基础阶段的持续适配结果','',f"状态：{audit['status']}。12条神经轨迹、90个任务状态、900个task-epoch已完成。",
          '每个任务均更新适配器；CT-Risk复用U轨迹，仅干预任务末读出。无额外S0。','',
          '| 数据 | 方法 | Final BA | Average BA | Tail | 最差run BA |','|---|---|---:|---:|---:|---:|']
    for m in means:text.append(f"| {m['dataset']} | {m['method']} | {m['Final_BA_mean']:.3f} ± {m['Final_BA_sd']:.3f} | {m['Average_BA_mean']:.3f} | {m['Final_tail_mean']:.3f} | {m['worst_run_BA']:.3f} |")
    text+=['','主比较固定为CT-Risk−CT-J-CB；全部配对差、区间和逐类结果见CSV。未挑选方法、epoch或seed。',
           '已观察事实：下列主比较为三个固定训练/顺序配对的平均差，不代表训练总体。']
    for q in intervals:
        if q['comparison']=='CT-Risk minus CT-J-CB' and q['task']==(11 if q['dataset']=='HK' else 4) and q['metric'] in ('BA','tail'):
            text.append(f"- {q['dataset']} Final {q['metric']}: {q['mean_difference_pp']:+.3f} pp，条件性95%区间 [{q['low']:+.3f}, {q['high']:+.3f}]。")
    text+=['','机制解释只限所定义的虚拟输运统计目标。对角仿射更新的代数等价不证明真实旧类漂移已准确恢复；',
           '高斯回放仅直接更新head，Risk不反传adapter；e和收缩方差不是临床置信度或覆盖保证。',
           '数据已参与开发，患者隔离未知。按类分层component bootstrap固定模型，singleton component没有病例间不确定性。',
           '不把本协议分数减旧大基础阶段结果作为配对提升。历史A1/HK1及负结果保留。',
           '旧训练图像与未来训练图像没有用于当前任务；新增test图像/特征/模型前向/预测均为0。',
           '模型状态、统计、封存逐样本分数和解析权重私有；公开仅源码、锁和聚合结果。','',
           f"GPU-process residence：{resource['GPU_process_residence_seconds']/3600:.3f}小时；CPU报告：{audit['CPU_report_seconds']:.1f}秒。",
           'NEXT_DECISION=STOP。不自动扩展、释放test或恢复定时监测。']
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(text)+'\n')
    write(pub/'NEXT_DECISION.json',dict(action='STOP',status=audit['status'],further_experiments_started=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
