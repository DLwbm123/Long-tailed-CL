"""G1 aggregate analysis: saved validation scores only, no fitting or test I/O."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from report_locked_holdout_r1 import read, write, csvwrite, bootstrap_weights, boot_unit

METHODS=('R0','B0','B1','B2','B3','B4')
ORDERS=(1993,1994,1995)
MIX=(41001,41002,41003)
CONTRASTS=('B4-B0','B3-B2','B4-B3','B0-R0','B3-B0','B2-B0','B1-B0')
METRICS=('balanced_accuracy','tail_rank2','old_macro_recall','current_macro_recall','label_6_recall','label_7_recall')


def seeds(method): return MIX if method in ('B2','B3','B4') else (0,)


def records(path):
    result=[]
    for row in csv.DictReader(Path(path).open()):
        r={}
        for k,v in row.items():
            try: r[k]=json.loads(v)
            except (ValueError,TypeError): r[k]=v if v!='' else None
        result.append(r)
    return result


def avg(values): return float(np.mean(values)) if values and all(v is not None for v in values) else None


def complete(out):
    out=Path(out); pub=out/'public'; private=out/'private'; start=time.monotonic()
    assert read(pub/'P2_COMPLETE.json')['status']=='PASS'
    metrics=records(pub/'val_metrics.csv'); classes=records(pub/'val_per_class_metrics.csv')
    assert len(metrics)==108 and len(classes)==648
    by={(r['method'],r['mix_seed'],r['order_seed'],r['session']):r for r in metrics}
    assert len(by)==108
    pcby={(r['method'],r['mix_seed'],r['order_seed'],r['session'],r['original_label']):r for r in classes}
    data={}
    for key in by:
        method,mix,order,stage=key
        with np.load(private/f'{method}_{mix}_{order}_s{stage}.npz',allow_pickle=False) as f:
            data[key]={k:f[k] for k in f.files if k!='W'}
    canonical=data['B0',0,1993,2]; idmap={s:i for i,s in enumerate(canonical['ids'])}
    for p in data.values():
        ix=np.array([idmap[s] for s in p['ids']]); assert np.array_equal(p['original'],canonical['original'][ix]) and np.array_equal(p['component'],canonical['component'][ix])
    finals=[]; summaries=[]; stages=[]; forgetting=[]
    extra=('accuracy','macro_f1','head_rank2','mid_rank4','worst_class_recall','component_equal_balanced_accuracy','HM_old_current_macro')
    for method in METHODS:
        for mix in seeds(method):
            rs=[by[method,mix,o,2] for o in ORDERS]
            for field in ('balanced_accuracy','tail_rank2','label_6_recall','label_7_recall'):
                assert max(r[field] for r in rs)-min(r[field] for r in rs)<1e-10
            f=dict(method=method,mix_seed=mix,unique_final_fit=True,equivalence_reference='B0' if method=='B1' else None,
                   synthesis_random_replicate=mix!=0,old_current_definition='mean of three fixed order partitions; not independent replicates')
            for field in METRICS+extra: f[field]=avg([r[field] for r in rs])
            f['AvgBA_all']=avg([by[method,mix,o,t]['balanced_accuracy'] for o in ORDERS for t in (0,1,2)])
            f['AvgBA_inc']=avg([by[method,mix,o,t]['balanced_accuracy'] for o in ORDERS for t in (1,2)])
            f['zero_recall_classes']=rs[0]['zero_recall_classes']; finals.append(f)
            for order in ORDERS:
                for c in range(8):
                    rows=[pcby[method,mix,order,t,c] for t in range(3) if (method,mix,order,t,c) in pcby]
                    eligible=len(rows)>1
                    forgetting.append(dict(method=method,mix_seed=mix,order_seed=order,original_label=c,first_session=rows[0]['session'],
                      first_recall=rows[0]['recall'],final_recall=rows[-1]['recall'],final_minus_first_pp=rows[-1]['recall']-rows[0]['recall'] if eligible else None,
                      signed_forgetting_first_minus_final_pp=rows[0]['recall']-rows[-1]['recall'] if eligible else None,
                      no_later_interval=not eligible))
        fs=[r for r in finals if r['method']==method]
        for field in METRICS+extra+('AvgBA_all','AvgBA_inc'):
            v=[r[field] for r in fs]
            summaries.append(dict(method=method,metric=field,mean=avg(v),std_ddof1=float(np.std(v,ddof=1)) if len(v)>1 else None,
                                  minimum=min(v),maximum=max(v),n_unique_fits=len(v),unit='fixed synthesis seeds' if len(v)>1 else 'deterministic'))
        for stage in range(3):
            for field in METRICS:
                values=[avg([by[method,mix,o,stage][field] for o in ORDERS]) for mix in seeds(method)]
                stages.append(dict(method=method,session=stage,metric=field,mean_fixed_orders_then_mix=avg(values),
                    std_across_mix=float(np.std(values,ddof=1)) if len(values)>1 and all(v is not None for v in values) else None,
                    values_by_mix=values,order_repetitions_are_not_independent=True))
    assert len(finals)==12
    csvwrite(pub/'final_unique_fits.csv',finals); csvwrite(pub/'final_summary.csv',summaries)
    csvwrite(pub/'stage_summary.csv',stages); csvwrite(pub/'forgetting.csv',forgetting)
    paired=[]
    for contrast in CONTRASTS:
        a,b=contrast.split('-'); mixs=MIX if len(seeds(a))>1 or len(seeds(b))>1 else (0,)
        for t in range(3):
            for field in METRICS:
                for order in ORDERS:
                    ds=[]
                    for mix in mixs:
                        va=by[a,mix if a in ('B2','B3','B4') else 0,order,t][field]
                        vb=by[b,mix if b in ('B2','B3','B4') else 0,order,t][field]
                        d=None if va is None or vb is None else va-vb;ds.append(d)
                        paired.append(dict(contrast=contrast,session=t,order_seed=order,mix_seed=mix,metric=field,difference_pp=d))
                    paired.append(dict(contrast=contrast,session=t,order_seed=order,mix_seed='fixed_mix_mean',metric=field,difference_pp=avg(ds)))
    csvwrite(pub/'paired_differences.csv',paired)
    # Exact unrounded gate values; partitions remain order-specific.
    def final_value(m,field):return avg([r[field] for r in finals if r['method']==m])
    deltas={field:final_value('B4',field)-final_value('B0',field) for field in METRICS}
    partition=[]
    for order in ORDERS:
        for field in ('old_macro_recall','current_macro_recall'):
            d=avg([by['B4',mix,order,2][field] for mix in MIX])-by['B0',0,order,2][field]
            partition.append(dict(order_seed=order,metric=field,mean_difference_pp=d,passes=d>=-2))
    improving=[]
    for mix in MIX:
        improving.append(dict(mix_seed=mix,BA_difference_pp=by['B4',mix,1993,2]['balanced_accuracy']-by['B0',0,1993,2]['balanced_accuracy'],
                              tail_difference_pp=by['B4',mix,1993,2]['tail_rank2']-by['B0',0,1993,2]['tail_rank2']))
    n_improve=sum(r['BA_difference_pp']>0 and r['tail_difference_pp']>0 for r in improving)
    zeros=[]
    for row in classes:
        if row['method']=='B4' and row['session'] in (1,2) and row['head_index']>=row['old_classes'] and row['recall']==0:
            base=pcby['B0',0,row['order_seed'],row['session'],row['original_label']]
            if base['recall']>0: zeros.append({k:row[k] for k in ('mix_seed','order_seed','session','original_label')})
    gates=dict(final_BA_gain_ge_1pp=deltas['balanced_accuracy']>=1,
               final_tail_gain_ge_3pp=deltas['tail_rank2']>=3,
               both_tail_classes_non_decreasing=deltas['label_6_recall']>=0 and deltas['label_7_recall']>=0,
               every_order_old_current_loss_le_2pp=all(r['passes'] for r in partition),
               at_least_two_mix_seeds_improve_BA_and_tail=n_improve>=2,
               no_new_current_zero=len(zeros)==0)
    if all(gates.values()): status='POSITIVE_CONTROLLED_DEV_SIGNAL'
    elif deltas['balanced_accuracy']<=0 and final_value('B3','balanced_accuracy')>final_value('B0','balanced_accuracy'):
        status='FIXED_SPHERICAL_SIGNAL_ADAPTIVITY_UNSUPPORTED'
    elif deltas['balanced_accuracy']>0:status='MIXED_DEV_SIGNAL'
    else:
        spectra=records(pub/'spectral_diagnostics.csv')
        b0=next(r for r in spectra if r['method']=='B0' and r['order_seed']==1993 and r['session']==2)
        b4=[r for r in spectra if r['method']=='B4' and r['order_seed']==1993 and r['session']==2]
        changed=avg([r['G_stable_rank'] for r in b4])>b0['G_stable_rank'] or avg([r['A_condition'] for r in b4])<b0['A_condition']
        status='SPECTRAL_CHANGE_WITHOUT_CLASSIFICATION_GAIN' if changed else 'NO_ADDED_BENEFIT_OVER_CBRIDGE'
    gate=dict(status=status,checks=gates,all_pass=all(gates.values()),B4_minus_B0_mean_pp=deltas,per_order_partitions=partition,
              per_mix_improvement=improving,n_mix_improving_both=n_improve,new_current_zero=zeros,thresholds_changed=False,independent_confirmation=False)
    write(pub/'DEVELOPMENT_GATE_CHECK.json',gate)
    # Shared class-stratified component draws on the complete val universe.
    weights=bootstrap_weights(canonical,resamples=2000,seed=42001); boots={}; intervals=[]
    for key,p in data.items():
        ix=[idmap[s] for s in p['ids']]; recalls=boot_unit(p,weights[:,ix]); stage=key[-1];known=(0,4,6)[stage];seen=recalls.shape[1]
        groups={'balanced_accuracy':range(seen),'old_macro_recall':range(known),'current_macro_recall':range(known,seen),
                'tail_rank2':[j for j in range(seen) if p['order'][j] in (6,7)]}
        for c in (6,7):groups[f'label_{c}_recall']=[j for j in range(seen) if p['order'][j]==c]
        for field,cols in groups.items():
            if len(cols): boots[key+(field,)]=recalls[:,list(cols)].mean(1)
    def ci(obj,stage,order,field,values,point):
        low,high=np.quantile(values,[.025,.975])
        intervals.append(dict(object=obj,session=stage,order_seed=order,metric=field,point=point,lower95=float(low),upper95=float(high),
          resamples=2000,bootstrap_seed=42001,unit='class-stratified identity_component',fixed_mix_seeds_averaged=True,
          mix_seeds_resampled=False,validation_adaptively_reused=True,interpretation='conditional descriptive development interval; no selection correction or independent confirmation'))
    def method_boot(method,order,t,field):
        keys=[(method,mix,order,t,field) for mix in seeds(method)]
        return np.mean([boots[k] for k in keys],axis=0) if all(k in boots for k in keys) else None
    for obj in METHODS+CONTRASTS:
        for t in range(3):
            for field in METRICS:
                values=[];points=[]
                for order in ORDERS:
                    if '-' in obj:
                        a,b=obj.split('-');ba=method_boot(a,order,t,field);bb=method_boot(b,order,t,field)
                        if ba is None or bb is None:continue
                        v=ba-bb;point=avg([by[a,m,order,t][field] for m in seeds(a)])-avg([by[b,m,order,t][field] for m in seeds(b)])
                    else:
                        v=method_boot(obj,order,t,field)
                        if v is None:continue
                        point=avg([by[obj,m,order,t][field] for m in seeds(obj)])
                    ci(obj,t,order,field,v,point);values.append(v);points.append(point)
                if values:ci(obj,t,'fixed_order_partition_average',field,np.mean(values,axis=0),avg(points))
        for ts,name in [((0,1,2),'AvgBA_all'),((1,2),'AvgBA_inc')]:
            values=[];points=[]
            for order in ORDERS:
                if '-' in obj:
                    a,b=obj.split('-');v=np.mean([method_boot(a,order,t,'balanced_accuracy')-method_boot(b,order,t,'balanced_accuracy') for t in ts],axis=0)
                    point=avg([by[a,m,order,t]['balanced_accuracy'] for m in seeds(a) for t in ts])-avg([by[b,m,order,t]['balanced_accuracy'] for m in seeds(b) for t in ts])
                else:
                    v=np.mean([method_boot(obj,order,t,'balanced_accuracy') for t in ts],axis=0)
                    point=avg([by[obj,m,order,t]['balanced_accuracy'] for m in seeds(obj) for t in ts])
                ci(obj,'across_stages',order,name,v,point);values.append(v);points.append(point)
            ci(obj,'across_stages','fixed_order_partition_average',name,np.mean(values,axis=0),avg(points))
    csvwrite(pub/'bootstrap_intervals.csv',intervals)
    write(pub/'BOOTSTRAP_AUDIT.json',dict(status='PASS',resamples=2000,seed=42001,n_components_by_class={str(c):len(np.unique(canonical['component'][canonical['original']==c])) for c in range(8)},
       shared_draws_all_methods_mix_orders=True,mix_seeds_resampled=False,independent_confirmation=False,validation_adaptively_reused=True,analysis_seconds=time.monotonic()-start))
    write(pub/'NEXT_DECISION.json',dict(status='STOP',technical_status='COMPLETE_G1_P0_P3',scientific_status=status,
       all_development_gates_pass=all(gates.values()),further_experiments_started=False,new_test_predictions=0,
       reason='The entire predeclared G1 matrix and P3 analysis are complete. No follow-on experiment or test release is authorized.'))


def render_report(pub):
    pub=Path(pub); gate=read(pub/'DEVELOPMENT_GATE_CHECK.json'); res=read(pub/'RESOURCE_REPORT.json')
    summary=records(pub/'final_summary.csv'); spec=records(pub/'spectral_diagnostics.csv')
    intervals=records(pub/'bootstrap_intervals.csv'); stats=records(pub/'final_unique_fits.csv');pc=records(pub/'val_per_class_metrics.csv')
    def value(m,k):return next(r for r in summary if r['method']==m and r['metric']==k)
    def fmt(m,k):
        r=value(m,k); return f"{r['mean']:.3f}"+(f" ± {r['std_ddof1']:.3f}" if r['std_ddof1'] is not None else '')
    d=gate['B4_minus_B0_mean_pp']; checks=gate['checks']
    text=f"""# G1 固定类别权重球面混合：完成报告

技术状态：**COMPLETE_G1_P0_P3**。主比较 B4−B0 的平均 Final BA 为 **{d['balanced_accuracy']:+.3f} 个百分点**，Final tail recall 为 **{d['tail_rank2']:+.3f} 个百分点**。预设六项开发门槛通过 **{sum(checks.values())}/6**，结论 **{gate['status']}**。全部固定变体已完成；结果没有触发继续搜索。

本轮 validation 只有 295 张图像，且已被历次开发反复使用。结果是既有开发集上的条件性观察，不能作为独立确认、统计显著性或临床非劣效证据。三次重复仅来自预先固定的合成随机流，既不是神经训练重复，也不代表训练随机性总体。原始标签 6/7 的临床名称与患者级隔离不作额外推断；预训练暴露限制保留。

## 固定矩阵与主结果

唯一输入为原 AugReg A 的 768 维 train/val 缓存；train=18,718，val=295；实际训练不平衡比 389.963:1。4→6→8 类，三种原顺序，λ=0.001，无 bias，float64 直接求解。真实特征仅由 float32 转 float64；无二次归一化。增强各类 M=N、β=1，固定类总权重 1/C。见 PROTOCOL_G1.json 和 SOURCE_AND_DEVIATIONS.md。

| 方法 | Final BA (%) | Final tail (%) | 标签6 (%) | 标签7 (%) | AvgBA_all | AvgBA_inc |
|---|---:|---:|---:|---:|---:|---:|
"""
    for m in METHODS:text+='| '+m+' | '+' | '.join(fmt(m,k) for k in ('balanced_accuracy','tail_rank2','label_6_recall','label_7_recall','AvgBA_all','AvgBA_inc'))+' |\n'
    text+='\nR0/B0/B1 为各一个确定性最终拟合，B1 是等价控制。B2/B3/B4 的均值和样本 SD 只使用 3 个 mix_seed；三个 order 的最终全类预测已经验证相同，没有把 9 行视为 9 个独立重复。AvgBA 先在每个 mix_seed 内平均三个固定 order，再计算 mix_seed 波动。\n\n'
    text+='## 固定比较与条件性区间\n\n| 比较 | Final BA差值pp [95%区间] | Final tail差值pp [95%区间] |\n|---|---:|---:|\n'
    for contrast in CONTRASTS:
        cells=[]
        for field in ('balanced_accuracy','tail_rank2'):
            r=next(r for r in intervals if r['object']==contrast and r['session']==2 and r['order_seed']=='fixed_order_partition_average' and r['metric']==field)
            cells.append(f"{r['point']:+.3f} [{r['lower95']:+.3f}, {r['upper95']:+.3f}]")
        text+='| '+contrast+' | '+' | '.join(cells)+' |\n'
    text+='\n区间用保存的 validation 分数计算：按类分层抽 identity_component、有放回 2,000 次，seed=42001。所有变体、mix_seed 和顺序共享抽样；每次抽样先平均三个固定合成随机实现，再作配对差。未重采样 mix_seed，也未按图像独立抽样；区间不校正此前的自适应选择或未知患者相关性。\n\n'
    text+='## 门槛机械复核\n\n'
    labels={'final_BA_gain_ge_1pp':'平均 Final BA ≥ +1 pp','final_tail_gain_ge_3pp':'平均 Final tail ≥ +3 pp','both_tail_classes_non_decreasing':'标签6和7均不降低','every_order_old_current_loss_le_2pp':'每个order的平均old/current下降各≤2 pp','at_least_two_mix_seeds_improve_BA_and_tail':'至少2/3 mix同时改善BA与tail','no_new_current_zero':'S1/S2无新增当前类零召回'}
    for key,passed in checks.items():text+=f"- {labels[key]}：{'通过' if passed else '未通过'}。\n"
    text+='\n| order | old差值pp | current差值pp |\n|---|---:|---:|\n'
    for o in ORDERS:
        v=[r['mean_difference_pp'] for r in gate['per_order_partitions'] if r['order_seed']==o];text+=f'| {o} | {v[0]:+.3f} | {v[1]:+.3f} |\n'
    text+=f"\n同时改善 BA/tail 的 mix_seed 数={gate['n_mix_improving_both']}/3；新增当前类零召回单元={len(gate['new_current_zero'])}。完整未舍入证据见 DEVELOPMENT_GATE_CHECK.json。\n"
    text+='\n| 方法 | 最差 mix Final BA | 最差类 recall（所有mix） | 最终零召回类别（各mix） |\n|---|---:|---:|---|\n'
    for m in METHODS:
        ss=[r for r in stats if r['method']==m]
        text+=f"| {m} | {min(r['balanced_accuracy'] for r in ss):.3f} | {min(r['worst_class_recall'] for r in ss):.3f} | {json.dumps({str(r['mix_seed']):r['zero_recall_classes'] for r in ss})} |\n"
    text+='\n逐阶段、逐类正确数/样本数/component数、head/middle/tail、old/current/HM、old/current-tail、raw正确类margin分位数，见 val_metrics.csv 与 val_per_class_metrics.csv。错误分解给出双向 old/current 误分与组内错误的 sample-weighted/class-macro 两种口径，三项各和为100%；restricted-current 仅用于诊断。forgetting.csv 记录首次学习到最后的有符号召回变化；最后到达类没有后续区间，记 NA。\n'
    text+='\n## 谱与机制边界\n\n| 方法 | Final trace(G) | stable rank(G) | condition(A) | Delta负谱和 | Delta正谱和 |\n|---|---:|---:|---:|---:|---:|\n'
    for m in METHODS:
        ss=[r for r in spec if r['method']==m and r['order_seed']==1993 and r['session']==2]
        text+='| '+m+' | '+' | '.join(f'{np.mean([r[k] for r in ss]):.7g}' for k in ('G_trace','G_stable_rank','A_condition','delta_negative_spectral_sum','delta_positive_spectral_sum'))+' |\n'
    text+='\n上述是已观察到的统计与分类结果。stable rank 使用 ||G||_F²/||G||₂²；不是 trace(G)/λmax。每个阶段使用同一个 B0 top16 基底比较各类二阶矩和标签交叉统计的投影，剩余子空间单列；“谱剩余方向”不等同于频次尾类。完整原始谱、数值秩阈值、微小负值和仅用于熵的裁剪数量均已保存。\n\n同类归一化混合仍落在真实行空间内；toy和真实 N_c<768 类的 SVD 残差检查通过。因此不能声称填满真实 null-space。归一化后的 Delta 同时含正负谱，不能套用原论文双流加法的 PSD 增益保证。B3−B2 同时改变能量和几何，不能归因于纯方向效应。标签交叉统计 R 也同步改变；谱指标变化本身不是分类成功证据。以上是机制约束与推断边界，外部泛化、独立 holdout 收益、C-S0 表征归因均未在本轮验证。\n'
    text+='\n## 工程、谱系、资源与交付\n\nB0 对历史高精度 validation CSV 的阶段/逐类复现通过；B1 的 G/R/W/score 与预测等价通过。未来类隔离、配对覆盖、共享随机流、全局随机状态不变、退化输入、span、独立加权批量目标、统计恢复、最终顺序不变性、指标映射检查均通过。代码锁在 P1 通过后、P2 之前生成；没有按分数删方法或调整协议。\n\n源自 R1 2ab2c2690953cc34128d4b702df0a4e920e54d44 和 V3 cda1b371225de3568f7f07a8fc1658ec5f03fa3d。缓存和 association/manifest 的锁定哈希在本轮验证；原始权重本体没有再次加载或复验，来源依据原 CACHE_LOCK 链。旧工作区与历史结果保留；独立分支未合并 main。\n\n'
    text+=f"实际执行耗时 {res['wall_seconds']:.1f} 秒，峰值 RSS {res['peak_RSS_bytes']/1024**3:.3f} GiB，观测新增文件峰值 {res['observed_max_new_file_bytes']/1024**2:.2f} MiB，最低观测磁盘空闲 {res['min_observed_free_bytes']/1024**3:.3f} GiB。单 worker，BLAS/OpenMP 4 线程，GPU 使用为0。正式 solve 调用 {res['actual_formal_solve_calls']} 次；12个方法/随机状态最终拟合，B1引用B0等价控制。P0额外的隔离toy求解和单类基准不计为候选。\n\n"
    text+='每条增量学习器的继续更新状态最多约4.74 MiB；审计归档额外保留11份去重最终累计统计、一个活跃状态、各阶段 W、私有 validation 分数和统计来源。没有保存全量合成特征。原缓存是既有资产依赖；本次运行的总峰值 RSS 包括审计/诊断工作空间，不把它伪称为学习器“零存储”。这些统计/特征不宣称隐私安全。公开内容仅源码、协议和聚合审计表；个体特征、身份关联、精确私有路径和分数保留私有。\n\n'
    text+='```text\nneural_training_epochs=0\noptimizer_steps=0\nencoder_forward_calls=0\nnew_test_predictions=0\nnew_test_feature_reads=0\ncore_logical_metric_rows=108\ncore_logical_per_class_rows=648\nvalidation_adaptively_reused=true\nfull_GSR_reproduction=false\nfurther_experiments_started=false\n```\n\nP3 完成后停止。没有 test 释放、新训练、调参、新编码器或定时监测。NEXT_DECISION.json=STOP。\n'
    (pub/'FINAL_REPORT_ZH.md').write_text(text)
    (pub/'RESOURCE_REPORT.md').write_text('# G1 resource report\n\n'+json.dumps(res,indent=2)+'\n\nCPU only; one worker, four BLAS threads. Existing cache files were read in place. All generated data are on the same data mount; no historical file was deleted. File-size/free-space checks ran after each logical stage. Numerical kernels do not write temporary disk matrices. Aggregate reports and source snapshots are small additional overhead.\n')
