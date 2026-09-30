"""Frozen four-condition terminal analysis, using the original paired bootstrap."""
import json,time
from pathlib import Path
import numpy as np
from ct2d_math import metrics
from report_locked_holdout_r1 import csvwrite,bootstrap_weights,boot_unit
from report_ct6f import recall_groups

def report(r):
    from run_dfd_m1 import PUB,PRIVATE,REC,read,save,sha
    start=time.monotonic();groups=read(PUB/'PROTOCOL_LOCK.json')['frequency_groups']
    new=read(PUB/'PREDICTIONS_LOCK.json')['units'];old=read(REC/'public/CONTROL_REPRODUCTION_AUDIT.json')['units']
    data={};point={};rows=[];pcs=[];errors=[]
    for root,units in [(PRIVATE,new),(REC/'private',old)]:
        for e in units:
            pth=root/'sealed'/e['file'];assert sha(pth)==e['sha256']
            with np.load(pth) as f:p={k:f[k].copy() for k in f.files}
            method={'C3':'FD10_T','B1T':'FD1_T'}.get(e['method'],e['method']);key=e['dataset'],e['seed'],method
            assert key not in data;data[key]=p;meta=dict(dataset=key[0],seed=key[1],task=4,method=method)
            m,pc,er=metrics(p['raw'],p['y'],p['order'],6,groups[key[0]],meta,p['component'])
            for group,labels in groups[key[0]].items():
                ix=np.flatnonzero(np.isin(p['order'],labels));m[group+'_n_classes']=len(ix);m[group+'_n_images']=int(np.isin(p['y'],ix).sum())
            for scope,indices in [('old_tail',range(6)),('current_tail',range(6,8))]:
                ix=[k for k in indices if int(p['order'][k]) in groups[key[0]]['tail']]
                m[scope+'_recall']=float(np.mean([pc[k]['recall'] for k in ix])) if ix else None
                m[scope+'_n_classes']=len(ix);m[scope+'_n_images']=sum(pc[k]['n_images'] for k in ix)
            m['source']='REBUILT_FIXED_CONTROL' if method.startswith('FD') else 'NEW_FIXED_TRAJECTORY'
            m['historical_per_sample_equality']='UNKNOWN' if method.startswith('FD') else 'NOT_APPLICABLE'
            point[key]=m;rows.append(m);pcs+=pc;errors+=er
    assert len(rows)==24 and len(pcs)==192
    contrasts=[('D_T','R_T'),('D_T','FD1_T'),('D_T','FD10_T'),('R_T','FD1_T')];paired=[];gates=[];summary=[]
    for ds in ('ISIC','HK'):
        identity={}
        for key,p in data.items():
            if key[0]!=ds:continue
            for sid,lab,comp in zip(p['ids'],p['original'],p['component']):
                v=int(lab),str(comp);assert sid not in identity or identity[sid]==v;identity[sid]=v
        ids=sorted(identity);ix={v:i for i,v in enumerate(ids)}
        canon=dict(y=np.zeros(len(ids),int),original=np.array([identity[v][0] for v in ids]),component=np.array([identity[v][1] for v in ids]))
        weights=bootstrap_weights(canon,2000,64201,sorted(set(canon['original'])));boot={}
        for key,p in data.items():
            if key[0]==ds:boot[key]=recall_groups(boot_unit(p,weights[:,[ix[v] for v in p['ids']]],True),6,p['order'],groups[ds]['tail'])
        for a,b in contrasts:
            for metric,col in [('BA','balanced_accuracy'),('old','old_macro_recall'),('current','current_macro_recall'),('tail','tail_recall'),('HM','HM')]:
                vals=[];samples=[]
                for seed in (1993,1994,1995):
                    ka,kb=(ds,seed,a),(ds,seed,b)
                    for k in ('ids','order','original','y','component'):np.testing.assert_array_equal(data[ka][k],data[kb][k])
                    va,vb=point[ka][col],point[kb][col]
                    if va is None or vb is None:delta=lo=hi=None
                    else:
                        delta=va-vb;s=boot[ka][metric]-boot[kb][metric];lo,hi=map(float,np.quantile(s,[.025,.975]));samples.append(s);vals.append(delta)
                    paired.append(dict(dataset=ds,seed=seed,contrast=a+'-'+b,metric=metric,difference_pp=delta,low=lo,high=hi))
                lo=hi=delta=None
                if len(vals)==3:
                    delta=float(np.mean(vals));lo,hi=map(float,np.quantile(np.mean(samples,axis=0),[.025,.975]))
                paired.append(dict(dataset=ds,seed='fixed_three_mean',contrast=a+'-'+b,metric=metric,difference_pp=delta,low=lo,high=hi))
        def contrast(c,m):return next(x for x in paired if x['dataset']==ds and x['seed']=='fixed_three_mean' and x['contrast']==c and x['metric']==m)
        def diff(c,m):return contrast(c,m)['difference_pp']
        positive=sum(point[ds,s,'D_T']['balanced_accuracy']>point[ds,s,'R_T']['balanced_accuracy'] for s in (1993,1994,1995))
        M=diff('D_T-R_T','BA')>=1 and positive>=2
        U1=diff('D_T-FD1_T','BA')>=.5 and diff('D_T-FD1_T','old')>=0 and diff('D_T-FD1_T','current')>=-1
        U2=diff('D_T-FD10_T','current')>=2 and diff('D_T-FD10_T','old')>=-1 and diff('D_T-FD10_T','BA')>=0
        zero=[]
        for pc in pcs:
            if pc['dataset']==ds and pc['method']=='D_T' and pc['recall']==0:
                control=next(q for q in pcs if (q['dataset'],q['seed'],q['method'],q['head_index'])==(ds,pc['seed'],'FD10_T',pc['head_index']))
                if pc['n_images']>=10 and pc['n_components']>=5 and control['recall']>0:zero.append(dict(seed=pc['seed'],original_label=pc['original_label']))
        tail=diff('D_T-FD10_T','tail')
        S=all(point[ds,s,'D_T']['balanced_accuracy']-point[ds,s,'FD1_T']['balanced_accuracy']>=-2 for s in (1993,1994,1995)) and (tail is None or tail>=-1) and not zero
        mechanism='MECHANISM_TENTATIVE' if M and contrast('D_T-R_T','BA')['low']<=0 else ('DEVELOPMENTAL_SUPPORT' if M else 'FAIL')
        conclusion='LOCAL_DIRECTIONAL_SIGNAL_PENDING_FZ1' if all((M,U1,U2,S)) else ('MECHANISM_SIGNAL_NO_PROMOTION' if M else ('REGULARIZATION_EFFECT_ONLY' if U1 and U2 and S else 'CLOSE_DFD_T4_PILOT'))
        gates.append(dict(dataset=ds,M='PASS' if M else 'FAIL',positive_pairs=positive,mechanism_status=mechanism,U1='PASS' if U1 else 'FAIL',U2='PASS' if U2 else 'FAIL',S='PASS' if S else 'FAIL',new_qualified_zero_recall=zero,
            U3='NOT_EVALUABLE_MISSING_FZ1',full_method_acceptance='PENDING_FZ1',conclusion=conclusion,full_method_promotion=False,frozen_superiority='NOT_ESTABLISHED',NEXT_DECISION='STOP'))
        for method in ('D_T','R_T','FD10_T','FD1_T'):
            rs=[point[ds,s,method] for s in (1993,1994,1995)]
            summary.append(dict(dataset=ds,method=method,mean_BA=float(np.mean([v['balanced_accuracy'] for v in rs])),worst_seed_BA=min(v['balanced_accuracy'] for v in rs)))
    availability=[dict(dataset=d,seed=s,method='FZ1',status='DEFERRED_MISSING_TASK4_READOUT',BA=None,HM=None) for d in ('ISIC','HK') for s in (1993,1994,1995)]
    for name,rs in [('task4_metrics',rows),('task4_per_class',pcs),('paired_differences',paired),('conditional_intervals',paired),('error_flows',errors),('FZ1_availability',availability),('dataset_summary',summary),('zero_recall_classes',[p for p in pcs if p['recall']==0])]:
        if rs:csvwrite(PUB/(name+'.csv'),rs)
    epochs=[json.loads(x) for x in (PUB/'epoch_records.jsonl').read_text().splitlines()];probes=[json.loads(x) for x in (PUB/'gradient_probes.jsonl').read_text().splitlines()]
    assert len(epochs)==len(probes)==120 and sum(v['optimizer_steps'] for v in epochs)==5500
    assert len({(v['dataset'],v['seed'],v['method'],v['epoch']) for v in epochs})==120
    save('decision_gates.json',dict(datasets=gates,original_five_condition_plan_complete=False,full_method_promotion=False))
    save('ANALYSIS_AUDIT.json',dict(rows=24,per_class=192,FZ1_availability_rows=6,epochs=120,gradient_probes=120,bootstrap_replicates=2000,bootstrap_seed=64201,shared_component_draws_across_methods_and_orders=True,CPU_seconds=time.monotonic()-start,historical_longitudinal_fields='NOT_AVAILABLE'))
    lines=['# DFD-T4-P-v1-M1：固定 Task4 方向机制实验','',
      '12 条 D/R 轨迹完成，新增 5500 步、120 epoch；四条件 24 行阶段指标、192 行逐类结果。FZ1 六项后置；原五条件计划未完整完成。',
      '控制为 REBUILT_FIXED_CONTROL，历史逐样本一致性 UNKNOWN；Task3 W 为 REBUILT_FROM_VERIFIED_PARENT，独立历史 W 匹配 NOT_AVAILABLE。','',
      '|数据|方法|平均 BA|最差 seed BA|','|---|---|---:|---:|']
    for x in summary:lines.append(f"|{x['dataset']}|{x['method']}|{x['mean_BA']:.3f}|{x['worst_seed_BA']:.3f}|")
    for g in gates:
        lines+=['',f"## {g['dataset']}",f"M={g['M']}；机制={g['mechanism_status']}；U1={g['U1']}；U2={g['U2']}；S={g['S']}。",f"结论：{g['conclusion']}。"]
        for x in paired:
            if x['dataset']==g['dataset'] and x['seed']=='fixed_three_mean' and x['metric']=='BA':lines.append(f"- {x['contrast']}: {x['difference_pp']:+.3f}pp [{x['low']:+.3f}, {x['high']:+.3f}]。")
    lines+=['','U3=NOT_EVALUABLE_MISSING_FZ1；完整方法验收=PENDING_FZ1；frozen_superiority=NOT_ESTABLISHED；full_method_promotion=false。',
      '区间为 class/component 共享配对 bootstrap 2000 次、seed64201，条件于固定模型和既定顺序；不重抽训练 seed，不是独立确认研究，未校正多重比较。HK 是每顺序前 8/23 类。控制恢复不是独立验证。',
      '全部零召回、分母、restricted 诊断与错误流向见配套表。缺失历史纵向字段为 NOT_AVAILABLE。当前输入漂移不代表旧图像真实漂移。',
      '私有终态、Q 来源、W、预测和锁通过独立备份回执验证；最终交付状态见 COMPLETE.json 与 DELIVERY_AUDIT.json。','NEXT_DECISION=STOP。']
    (PUB/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
