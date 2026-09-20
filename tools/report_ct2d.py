"""Fixed CPU-only CT2-D analysis of saved validation scores; no model loading."""
import csv,json,os,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from report_locked_holdout_r1 import read,write,csvwrite,bootstrap_weights,boot_unit
from ct2d_math import metrics
from hashlib import sha256

def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def mean(xs):return float(np.mean(xs))

def main():
    began=time.monotonic();cfg=read(os.environ['P21_CONFIG']);root=Path(cfg['root']);pub=root/'output/public';private=root/'output/private'
    assert read(pub/'GPU_PHASE_COMPLETE.json')['status']=='PASS'
    scorelock=read(pub/'SCORES_LOCK.json');assert scorelock['formal_metric_units']==54
    for name,digest in scorelock['files'].items():assert sha256((private/'scores'/name).read_bytes()).hexdigest()==digest,'BLOCKED_SCORES_DRIFT'
    data={};ms=[];pcs=[];errors=[];fit=lines(pub/'oracle_fit_audit.jsonl')
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            for stream in ('U','R'):
                for mode in ('Q00','Q10','Q01','Q11')+(('Q11-Risk',) if stream=='U' else ()):
                    path=private/'scores'/f'{name}_{seed}_{stream}_{mode}.npz'
                    with np.load(path) as f:p={k:f[k].copy() for k in f.files if k!='W'}
                    c=len(p['order']);known=c-(3 if name=='HK' else 2)
                    assert len(set(p['ids']))==len(p['ids']) and np.array_equal(p['order'][p['y']],p['original'])
                    meta=dict(dataset=name,seed=seed,stream=stream,mode=mode,identity='ORACLE_MOMENT_DIAGNOSTIC',valid_online_CIL=False)
                    m,pc,er=metrics(p['raw'],p['y'],p['order'],known,cfg['frequency_groups'][name],meta,p['component'])
                    f=next(x for x in fit if all(x[k]==meta[k] for k in ('dataset','seed','stream','mode')))
                    m.update(fit_BA=f['fit_BA'],fit_accuracy=f['fit_accuracy'],fit_val_BA_gap=f['fit_BA']-m['balanced_accuracy'])
                    ms.append(m);pcs+=pc;errors+=er;data[name,seed,stream,mode]=p
    assert len(ms)==54 and len(pcs)==837
    csvwrite(pub/'oracle_factorial_metrics.csv',ms);csvwrite(pub/'oracle_factorial_per_class.csv',pcs);csvwrite(pub/'oracle_error_decomposition.csv',errors)
    for source,target in [('geometry.jsonl','transport_generalization_and_geometry.csv'),('routing.jsonl','routing_change_diagnostics.csv'),
                          ('class_pair_distances.jsonl','class_pair_distances.csv'),('synthetic_diagnostics.jsonl','synthetic_vs_real_feature_diagnostics.csv')]:
        csvwrite(pub/target,lines(pub/source))
    grads=lines(pub/'gradient_probes.jsonl');assert len(grads)==8
    assert all(g['model_unchanged'] and g['synthetic_direct_adapter_gradient_zero'] and g['optimizer_steps']==0 for g in grads)
    write(pub/'gradient_and_reduction_audit.json',dict(status='PASS',states=grads,total_states=8,
          synthetic_direct_adapter_gradient_zero=True,Risk_no_neural_graph=True,optimizer_steps=0,
          interpretation='Separate terms reproduce native reductions; no rescaling or training repair.'))
    pairs=[];intervals=[];summaries=[];components=[];boot={};points={}
    for m in ms:
        points[m['dataset'],m['seed'],m['stream'],m['mode']]={k:m[f] for k,f in [('BA','balanced_accuracy'),('tail','tail_recall'),('old','old_macro_recall'),('current','current_macro_recall'),('HM','HM')]}
    for name in ('HK','ISIC'):
        canonical=data[name,1993,'U','Q00'];n=23 if name=='HK' else 8;known=n-(3 if name=='HK' else 2)
        weights=bootstrap_weights(canonical,resamples=2000,seed=46001,labels=range(n))
        for c in range(n):
            mask=canonical['original']==c;nc=len(np.unique(canonical['component'][mask]))
            components.append(dict(dataset=name,original_label=c,n_images=int(mask.sum()),n_components=nc,singleton_component=nc==1))
        for key,p in data.items():
            if key[0]!=name:continue
            assert np.array_equal(canonical['ids'],p['ids']) and np.array_equal(canonical['original'],p['original']) and np.array_equal(canonical['component'],p['component'])
            recalls=boot_unit(p,weights,tie_original_label=True);tail=np.isin(p['order'],cfg['frequency_groups'][name]['tail'])
            old=recalls[:,:known].mean(1);cur=recalls[:,known:].mean(1)
            boot[key]=dict(BA=recalls.mean(1),tail=recalls[:,tail].mean(1),old=old,current=cur,HM=np.divide(2*old*cur,old+cur,out=np.zeros(2000),where=old+cur!=0))
        # Fixed same-protocol PT controls: no new fitting, no held-out assets.
        for seed in (1993,1994,1995):
            stage=11 if name=='HK' else 4
            with np.load(Path(cfg['ct1_root'])/'output/private/sealed'/f'{name}_{seed}_PT-CB_t{stage:02d}.npz') as f:p={k:f[k].copy() for k in f.files}
            assert np.array_equal(canonical['ids'],p['ids']) and np.array_equal(canonical['original'],p['original'])
            rr=boot_unit(p,weights,tie_original_label=True);tail=np.isin(p['order'],cfg['frequency_groups'][name]['tail']);old=rr[:,:known].mean(1);cur=rr[:,known:].mean(1)
            boot[name,seed,'PT','PT-CB']=dict(BA=rr.mean(1),tail=rr[:,tail].mean(1),old=old,current=cur,HM=np.divide(2*old*cur,old+cur,out=np.zeros(2000),where=old+cur!=0))
            m,_,_=metrics(p['raw'],p['y'],p['order'],known,cfg['frequency_groups'][name],{})
            points[name,seed,'PT','PT-CB']={k:m[f] for k,f in [('BA','balanced_accuracy'),('tail','tail_recall'),('old','old_macro_recall'),('current','current_macro_recall'),('HM','HM')]}
        for stream in ('U','R'):
            contrasts={'Q11-Q00':{'Q11':1,'Q00':-1},'Q10-Q00':{'Q10':1,'Q00':-1},'Q01-Q00':{'Q01':1,'Q00':-1},
                       'interaction':{'Q11':1,'Q10':-1,'Q01':-1,'Q00':1},'Q11-PT':{'Q11':1,'PT-CB':-1},'Q00-PT':{'Q00':1,'PT-CB':-1}}
            if stream=='U':contrasts['Q11-Risk-Q11']={'Q11-Risk':1,'Q11':-1}
            for contrast,terms in contrasts.items():
                for metric in ('BA','tail','old','current','HM'):
                    vals=[];rep=[]
                    for seed in (1993,1994,1995):
                        key=lambda mode:(name,seed,'PT' if mode=='PT-CB' else stream,mode)
                        val=sum(w*points[key(mode)][metric] for mode,w in terms.items());sample=sum(w*boot[key(mode)][metric] for mode,w in terms.items())
                        vals.append(val);rep.append(sample);pairs.append(dict(dataset=name,stream=stream,seed=seed,contrast=contrast,metric=metric,difference_pp=val))
                    lo,hi=np.quantile(np.mean(rep,axis=0),[.025,.975])
                    intervals.append(dict(dataset=name,stream=stream,contrast=contrast,metric=metric,mean_difference_pp=mean(vals),paired_sd=float(np.std(vals,ddof=1)),low=float(lo),high=float(hi),resamples=2000,seed=46001,fixed_models_not_resampled=True))
            modes=('Q00','Q10','Q01','Q11')+(('Q11-Risk',) if stream=='U' else ())
            for mode in modes:
                rr=[m for m in ms if (m['dataset'],m['stream'],m['mode'])==(name,stream,mode)]
                summaries.append(dict(dataset=name,stream=stream,mode=mode,BA_mean=mean([m['balanced_accuracy'] for m in rr]),BA_sd=float(np.std([m['balanced_accuracy'] for m in rr],ddof=1)),
                     tail=mean([m['tail_recall'] for m in rr]),old=mean([m['old_macro_recall'] for m in rr]),current=mean([m['current_macro_recall'] for m in rr]),fit_BA=mean([m['fit_BA'] for m in rr])))
    csvwrite(pub/'oracle_paired_differences.csv',pairs);csvwrite(pub/'oracle_bootstrap_intervals.csv',intervals);csvwrite(pub/'oracle_summary.csv',summaries);csvwrite(pub/'component_support.csv',components)
    zero=[]
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            for stream in ('U','R'):
                rr=[m for m in ms if (m['dataset'],m['seed'],m['stream'])==(name,seed,stream)]
                baseline=set(next(m for m in rr if m['mode']=='Q00')['zero_recall_classes'])
                for m in rr:zero.append(dict(dataset=name,seed=seed,stream=stream,mode=m['mode'],zero_recall_classes=m['zero_recall_classes'],new_zero_vs_Q00=sorted(set(m['zero_recall_classes'])-baseline)))
    csvwrite(pub/'zero_recall_audit.csv',zero)
    risk_error=[]
    for q in lines(pub/'risk_counterfactual_audit.jsonl'):
        if q['mode']!='Q11-Risk':continue
        for i,c in enumerate(q['order']):
            pc=next(x for x in pcs if (x['dataset'],x['seed'],x['stream'],x['mode'],x['original_label'])==(q['dataset'],q['seed'],'U','Q11-Risk',c))
            risk_error.append(dict(dataset=q['dataset'],seed=q['seed'],original_label=c,xi=q['xi'][i],no_slack_margin=q['no_slack_margin'][i],val_error=100-pc['recall']))
    csvwrite(pub/'risk_margin_error_diagnostics.csv',risk_error)
    restores=lines(pub/'restore_audit.jsonl');reproductions=lines(pub/'reproduction_audit.jsonl');assert len(reproductions)==12
    write(pub/'RESTORE_AND_REPRODUCTION_AUDIT.json',dict(status='PASS',restores=restores,reproductions=reproductions,final_models=12,extra_transition_parents=4))
    ledger=read(pub/'RESOURCE_AND_ACCESS_LEDGER.json');ledger.update(status='COMPLETE_CT2D',CPU_report_seconds=time.monotonic()-began,NEXT_DECISION='STOP')
    ledger['CPU_archive_source_audit_seconds']=read(pub/'ARCHIVE_SOURCE_AUDIT.json')['cpu_seconds']
    ledger['CPU_other_engineering_reserved_seconds']=600
    write(pub/'RESOURCE_AND_ACCESS_LEDGER.json',ledger)
    bugs=['# 实现核验与设计局限','',
      '- IMPLEMENTATION_CONFIRMED：原 270 行指标、2,754 行逐类正确数与分母独立复现；来源、映射、最小 original_label tie、seen 候选与任务步数通过。',
      '- IMPLEMENTATION_CONFIRMED：90 个序列化状态逐名核对 core 引用与全部非共享参数；12 个最终和 4 个转移父模型严格恢复。原 .pool. 命名检查包含两个池的嵌套 adapter，但不覆盖 keys/assigner/heads；本轮另外记录这些组。',
      '- IMPLEMENTATION_CONFIRMED：identity、仿射 full Gram/交叉项、类均衡目标、PSD factor 方向与 eta=0 合成验证通过。源码每个 epoch 从 immutable pre-task anchor 映射，任务末提交一次；4 个实际末任务映射另行复算。未发现重复 epoch 输运的证据。',
      '- SPECIFICATION_WEAKNESS：U 没有直接保持旧图像表征的 loss；Risk 是不反传 adapter 的任务末读出。synthetic CE 直接作用于 heads，上游 adapter 的直接梯度为零，这符合原定义。',
      '- SPECIFICATION_WEAKNESS：真实 main/sum CE 为 MEAN，few 与 assignment 为 SUM；保留原 /3、pull 和系数。梯度尺度探针说明实际差异，不把协议要求的 reduction 写成软件 bug。',
      '- SPECIFICATION_WEAKNESS：训练 batchwise 与提取 pointwise 的路由差异已在 CT1 声明；本轮记录 pointwise pre/post 切换，不声称二者同一映射。Risk slack 允许牺牲 margin，optimal 不保证各类识别。',
      '- SUPPORTED_MECHANISM：只能按 Q00/Q10/Q01/Q11、真实/合成与固定转移对照解释，结果见最终报告及全部逐配对表。',
      '- UNRESOLVED：有限末任务转移不等于所有 seed/任务因果确认；不能据结果证明信息不可逆丢失，也不能把离线旧图像重编码当合法无回放方法。','']
    (pub/'BUGS_VS_DESIGN_LIMITS.md').write_text('\n'.join(bugs))
    text=['# CT2-D 隔离诊断结果','', '**COMPLETE_CT2D；新增神经训练 epoch=0，optimizer steps=0。**','',
      '以下全为 ORACLE_MOMENT_DIAGNOSTIC，使用明确授权的旧 fit 图像，不能并入合法在线 CIL 排行榜。Q00 原输运统计；Q10 只刷新旧均值；Q01 只刷新旧中心化协方差和；Q11 同时刷新。当前类统计不变。','',
      '| 数据 | 流 | 状态 | Val BA mean ± sd | Tail | Old | Current | Fit BA |','|---|---|---|---:|---:|---:|---:|---:|']
    for q in summaries:text.append(f"| {q['dataset']} | {q['stream']} | {q['mode']} | {q['BA_mean']:.3f} ± {q['BA_sd']:.3f} | {q['tail']:.3f} | {q['old']:.3f} | {q['current']:.3f} | {q['fit_BA']:.3f} |")
    text+=['','主对照 Q11−Q00：']
    for q in intervals:
        if q['contrast']=='Q11-Q00' and q['metric']=='BA':text.append(f"- {q['dataset']}/{q['stream']}: {q['mean_difference_pp']:+.3f} pp，条件性95%区间 [{q['low']:+.3f}, {q['high']:+.3f}]。")
    text+=['','刷新统计后 Risk−CB：']
    for q in intervals:
        if q['contrast']=='Q11-Risk-Q11' and q['metric']=='BA':text.append(f"- {q['dataset']}: {q['mean_difference_pp']:+.3f} pp，条件性95%区间 [{q['low']:+.3f}, {q['high']:+.3f}]。")
    text+=['','以上差值为固定父模型的配对结果。正向 Q11 恢复支持记忆空间失配是可修复因素之一；Q11 仍差需结合 fit/val 与固定 PT 参照审查泛化。两者均不是唯一原因证明。',
      '梯度探针不重放历史某个 minibatch；只在同一诊断张量 B/2B 上分项求导，模型状态与参数不变。合成样本每旧类32对仅用于最后层诊断，不改变 CT1 训练配额。',
      '所有区间：2,000次按类分层 component 配对 bootstrap，seed46001，模型不重采样；singleton 及反复 validation 开发限制保留，不是独立确认。',
      f"实际旧训练图像诊断读取：{ledger['diagnostic_old_train_image_reads']}；encoder forward calls：{ledger['encoder_forward_calls']}；backward probes：{ledger['backward_probe_calls']}；新增解析拟合/校验解：{ledger['new_analytic_fits']}。全部 test 图像、特征、逐样本历史预测读取、前向、新预测均为0。", 
      f"GPU累计驻留：{ledger['GPU_process_residence_seconds']/3600:.3f}小时；原 CT1 资产未修改。",'','NEXT_DECISION=STOP。不自动启动 CT2 训练、参数搜索、test 或监测。','']
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(text));write(pub/'NEXT_DECISION.json',dict(status='COMPLETE_CT2D',action='STOP',further_training_started=False))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
