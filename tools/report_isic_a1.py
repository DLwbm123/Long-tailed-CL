"""A1 saved-validation analysis. No encoder, fitting, test, or model selection."""
from pathlib import Path
import numpy as np
from report_locked_holdout_r1 import read,write,csvwrite,bootstrap_weights,boot_unit
from report_isic_g1 import records,avg

METHODS=('A-U','A-CB','S-J-U','S-J-CB','S-M-U','S-M-CB','S-F-U','S-F-CB')
ORDERS=(1993,1994,1995)
FIELDS=('balanced_accuracy','tail_rank2','old_macro_recall','current_macro_recall','label_6_recall','label_7_recall',
        'base_macro_recall','post_base_macro_recall','tail_base_macro_recall','tail_post_base_macro_recall')
CONTRASTS=[('S-J-CB','A-CB'),('S-J-CB','S-J-U'),('A-CB','A-U'),('S-J-U','A-U'),
           ('S-M-CB','A-CB'),('S-F-CB','A-CB'),('S-M-CB','S-M-U'),('S-F-CB','S-F-U'),
           ('S-J-CB','S-M-CB'),('S-J-CB','S-F-CB'),('S-J-U','S-M-U'),('S-J-U','S-F-U')]
COEFFICIENTS={a+' minus '+b:{a:1,b:-1} for a,b in CONTRASTS}
COEFFICIENTS['interaction']={'S-J-CB':1,'S-J-U':-1,'A-CB':-1,'A-U':1}


def complete(out,baseline_only=False):
    out=Path(out);pub=out/'public';private=out/'private'
    rows=records(pub/'val_metrics.csv');pcs=records(pub/'val_per_class_metrics.csv')
    methods=('A-U','A-CB') if baseline_only else METHODS
    coefficients={'A-CB minus A-U':COEFFICIENTS['A-CB minus A-U']} if baseline_only else COEFFICIENTS
    assert len(rows)==len(methods)*9 and len(pcs)==len(methods)*54
    by={(r['method'],r['order_seed'],r['session']):r for r in rows}
    assert set(by)=={(m,s,t) for m in methods for s in ORDERS for t in range(3)}
    final=[];summary=[];forget=[];pairs=[];interaction=[];data={}
    for m in methods:
        for seed in ORDERS:
            rs=[by[m,seed,s] for s in range(3)];f=dict(rs[2],AvgBA_all=avg([r['balanced_accuracy'] for r in rs]),AvgBA_inc=avg([r['balanced_accuracy'] for r in rs[1:]]));final.append(f)
            for c in range(8):
                p=[r for r in pcs if r['method']==m and r['order_seed']==seed and r['original_label']==c]
                eligible=len(p)>1
                forget.append(dict(method=m,order_seed=seed,original_label=c,in_parent_base=p[0]['in_parent_base'],first_session=p[0]['session'],
                    first_recall=p[0]['recall'],last_recall=p[-1]['recall'],first_minus_final_pp=p[0]['recall']-p[-1]['recall'] if eligible else None,
                    final_minus_first_pp=p[-1]['recall']-p[0]['recall'] if eligible else None,no_subsequent_interval=not eligible))
            for stage in range(3):
                with np.load(private/f'{m}_{seed}_s{stage}.npz',allow_pickle=False) as a:data[m,seed,stage]={k:a[k] for k in a.files if k!='W'}
        fs=[r for r in final if r['method']==m]
        for field in FIELDS+('accuracy','macro_f1','head_rank2','mid_rank4','HM_old_current_macro','worst_class_recall','AvgBA_all','AvgBA_inc'):
            v=[r[field] for r in fs];det=m.startswith('A-')
            summary.append(dict(method=m,metric=field,mean=avg(v),std_ddof1=None if det else float(np.std(v,ddof=1)),minimum=min(v),maximum=max(v),
                n_parent_models=0 if det else 3,n_unique_final_fits=1 if det else 3,
                interpretation='one deterministic final fit; partition/stage averages use fixed orders' if det else 'three existing fixed S0/order pairs, not independent splits'))
        if m.startswith('A-'):
            preds=[data[m,s,2]['order'][data[m,s,2]['raw'].argmax(1)] for s in ORDERS]
            assert all(np.array_equal(preds[0],x) for x in preds[1:])
    for name,coeff in coefficients.items():
        output=interaction if name=='interaction' else pairs
        for stage in range(3):
            for field in FIELDS:
                differences=[]
                for seed in ORDERS:
                    values=[by[m,seed,stage][field] for m in coeff]
                    d=sum(coeff[m]*by[m,seed,stage][field] for m in coeff) if all(v is not None for v in values) else None
                    differences.append(d);output.append(dict(contrast=name,order_seed=seed,session=stage,metric=field,difference_pp=d))
                output.append(dict(contrast=name,order_seed='fixed_pair_mean',session=stage,metric=field,difference_pp=avg(differences),
                    std_across_fixed_pairs=float(np.std(differences,ddof=1)) if all(d is not None for d in differences) else None))
    for name,table in [('final_by_pair',final),('final_summary',summary),('forgetting',forget),('paired_differences',pairs),('interaction_differences',interaction)]:
        if table:csvwrite(pub/(name+'.csv'),table)
    canonical=data['A-CB',1993,2];idmap={s:i for i,s in enumerate(canonical['ids'])}
    weights=bootstrap_weights(canonical,resamples=2000,seed=43001);boots={};intervals=[]
    for key,p in data.items():
        ix=np.array([idmap[s] for s in p['ids']]);assert np.array_equal(p['original'],canonical['original'][ix]) and np.array_equal(p['component'],canonical['component'][ix])
        recalls=boot_unit(p,weights[:,ix],tie_original_label=True);stage=key[-1];seen=recalls.shape[1];known=(0,4,6)[stage];order=p['order']
        groups={'balanced_accuracy':list(range(seen)),'old_macro_recall':list(range(known)),'current_macro_recall':list(range(known,seen)),
                'tail_rank2':[j for j in range(seen) if order[j] in (6,7)],'base_macro_recall':list(range(min(4,seen))),
                'post_base_macro_recall':list(range(4,seen)),
                'tail_base_macro_recall':[j for j in range(min(4,seen)) if order[j] in (6,7)],
                'tail_post_base_macro_recall':[j for j in range(4,seen) if order[j] in (6,7)]}
        for c in (6,7):groups[f'label_{c}_recall']=[j for j in range(seen) if order[j]==c]
        for field,cols in groups.items():
            if cols:boots[key+(field,)]=recalls[:,cols].mean(1)
    def ci(name,seed,stage,field,value,point):
        lo,hi=np.quantile(value,[.025,.975]);intervals.append(dict(contrast=name,order_seed=seed,session=stage,metric=field,point_pp=point,lower95=float(lo),upper95=float(hi),
           resamples=2000,seed=43001,unit='class-stratified identity_component',parents_resampled=False,independent_confirmation=False,
           interpretation='fixed parent pairs; adaptively reused development validation; conditional descriptive interval'))
    for name,coeff in coefficients.items():
        for stage in range(3):
            for field in FIELDS:
                ds=[];points=[]
                for seed in ORDERS:
                    if not all((m,seed,stage,field) in boots for m in coeff):continue
                    d=sum(v*boots[m,seed,stage,field] for m,v in coeff.items());point=sum(v*by[m,seed,stage][field] for m,v in coeff.items())
                    ci(name,seed,stage,field,d,point);ds.append(d);points.append(point)
                # Never silently change the mean's set of fixed parents for empty groups.
                if len(ds)==3:ci(name,'fixed_pair_mean',stage,field,np.mean(ds,axis=0),avg(points))
    csvwrite(pub/'bootstrap_intervals.csv',intervals)
    write(pub/'BOOTSTRAP_AUDIT.json',dict(status='PASS_BASELINE_ONLY' if baseline_only else 'PASS',methods=list(methods),resamples=2000,seed=43001,shared_draws_all_methods_parents=True,parents_resampled=False,
         component_counts={str(c):len(np.unique(canonical['component'][canonical['original']==c])) for c in range(8)},
         unknown_patient_correlation=True,validation_adaptively_reused=True,independent_confirmation=False))
    if baseline_only:return
    principal=[r for r in pairs if r['contrast']=='S-J-CB minus A-CB' and r['order_seed']=='fixed_pair_mean' and r['session']==2]
    d={r['metric']:r['difference_pp'] for r in principal}
    if d['balanced_accuracy']>0 and d['tail_rank2']>0:direction='POSITIVE_DESCRIPTIVE_DIRECTION'
    elif d['balanced_accuracy']<0 and d['tail_rank2']<0:direction='NEGATIVE_DESCRIPTIVE_DIRECTION'
    else:direction='MIXED_DESCRIPTIVE_DIRECTION'
    write(pub/'NEXT_DECISION.json',dict(status='STOP',technical_status='COMPLETE_A1_ATTRIBUTION',observed_direction=direction,
          main_comparison='S-J-CB minus A-CB',mean_differences_pp=d,automatic_promotion_gate=None,independent_confirmation=False,
          further_experiments_started=False,reason='Fixed attribution complete; no follow-on training, test release, feature fusion or monitoring.'))
    write(pub/'TRAIN_FORENSICS_SCOPE.json',dict(train_resubstitution_scores_generated=False,reason='Optional scoring omitted; required train feature geometry and isolated after-final reblocking audit retained.',
          audit_reaccess_does_not_feed_earlier_fits=True,validation_used_for_fit=False))


def render(pub):
    pub=Path(pub);summary=records(pub/'final_summary.csv');final=records(pub/'final_by_pair.csv');intervals=records(pub/'bootstrap_intervals.csv')
    decision=read(pub/'NEXT_DECISION.json');resources=read(pub/'RESOURCE_REPORT.json');completion=read(pub/'COMPLETION_AUDIT.json');access=read(pub/'ACCESS_AUDIT.json')
    def fmt(m,k):
        r=next(r for r in summary if r['method']==m and r['metric']==k)
        return f"{r['mean']:.3f}"+(f" ± {r['std_ddof1']:.3f}" if r['std_ddof1'] is not None else '')
    d=decision['mean_differences_pp']
    text=f'''# A1 首阶段表征 × 类别加权归因

状态：**COMPLETE_A1_ATTRIBUTION**。主比较 S-J-CB−A-CB 的平均 Final BA 为 **{d['balanced_accuracy']:+.3f} pp**，Final tail recall 为 **{d['tail_rank2']:+.3f} pp**，观察方向为 **{decision['observed_direction']}**。全部8个预设变体、3个合法父模型/顺序配对、3阶段已完成。A1不设置新增晋级门槛，完成后停止。

本结果只说明既定 S0_POINTWISE_PROBE 路由、归一化、λ=0.001和解析目标下的线性可用信息。validation仅295张且已反复用于开发，既有test反馈的影响也不会因本次未读test而消失；不能称独立确认、临床有效性、不可逆表征损坏或总体因果效应。

## 固定结果

| 方法 | Final BA | tail recall | 标签6 | 标签7 | AvgBA_all | AvgBA_inc |
|---|---:|---:|---:|---:|---:|---:|
'''
    for m in METHODS:text+='| '+m+' | '+' | '.join(fmt(m,k) for k in ('balanced_accuracy','tail_rank2','label_6_recall','label_7_recall','AvgBA_all','AvgBA_inc'))+' |\n'
    text+='\nA-U/A-CB各是一个确定性最终拟合，早期阶段及old/current/base分组仍按三个固定order报告，没有伪造三个训练重复。S的SD（ddof=1）来自三个既有S0/顺序组合，不是独立患者划分或训练总体。M/F是完整报告的固定诊断，不按最好分支替换J主比较。\n\n'
    text+='| 对应父模型/order | S-J-CB BA | A-CB BA | S-J-CB tail | A-CB tail | 标签6正确/总数 | 标签7正确/总数 |\n|---|---:|---:|---:|---:|---|---|\n'
    for seed in ORDERS:
        s=next(r for r in final if r['method']=='S-J-CB' and r['order_seed']==seed);a=next(r for r in final if r['method']=='A-CB' and r['order_seed']==seed)
        text+=f"| {seed} | {s['balanced_accuracy']:.3f} | {a['balanced_accuracy']:.3f} | {s['tail_rank2']:.3f} | {a['tail_rank2']:.3f} | {s['label_6_n_correct']}/{s['label_6_n_images']} | {s['label_7_n_correct']}/{s['label_7_n_images']} |\n"
    text+='\n## 配对归因与区间\n\n| 固定比较 | Final BA差值pp [95%区间] | tail差值pp [95%区间] |\n|---|---:|---:|\n'
    for name in COEFFICIENTS:
        cells=[]
        for field in ('balanced_accuracy','tail_rank2'):
            r=next(r for r in intervals if r['contrast']==name and r['order_seed']=='fixed_pair_mean' and r['session']==2 and r['metric']==field)
            cells.append(f"{r['point_pp']:+.3f} [{r['lower95']:+.3f}, {r['upper95']:+.3f}]")
        text+='| '+name+' | '+' | '.join(cells)+' |\n'
    text+='\n交互差为(S-J-CB−S-J-U)−(A-CB−A-U)，是本固定设计的描述性差中差。区间来自按类分层identity_component的2000次配对bootstrap，seed=43001；全部方法与父模型共享抽样，每次抽样平均三个固定配对，不重采样父模型。区间不校正自适应开发、历史test反馈、未知患者关联或域变化；包含0不是等效，排除0也不是独立确认。\n\n'
    text+='| 方法 | base recall | post-base recall | tail-base | tail-post-base | 最差父模型BA | 最差类recall |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for m in METHODS:
        fs=[r for r in final if r['method']==m]
        text+='| '+m+' | '+' | '.join(fmt(m,k) for k in ('base_macro_recall','post_base_macro_recall','tail_base_macro_recall','tail_post_base_macro_recall'))+f" | {min(r['balanced_accuracy'] for r in fs):.3f} | {min(r['worst_class_recall'] for r in fs):.3f} |\n"
    text+='\n逐阶段/逐类正确数、分母、component数、in_parent_base、两种错误分解口径、old/current/HM、零召回、raw margin及其符号见主表和逐类表。最后才出现类的遗忘记NA，其余报告首次学完到最终的有符号下降。base/post-base与head/tail并不等同。未生成可选train重代入分数；train特征几何与完成所有阶段后的隔离重分块核对不作为泛化结果。\n'
    text+='\n## 工程与解释边界\n\n完整恢复原C分支S0最后checkpoint（epoch10），校验文件和全部网络tensor哈希；不构造训练learner或optimizer。父1993/1994/1995只接原对应顺序。冻结参数及buffers，eval、FP32、无梯度；原始backbone参考与A缓存小样本一致。两pool仅在隔离探针中关闭batchwise majority；pointwise与native B=1的路由ID及特征在预设atol/rtol=1e-5下通过，换同伴/顺序也通过。完整数值见ROUTING_PARITY.json，未据结果选择路由。\n\nJ把原始main/few拼接后整体归一化，保留相对范数；1536维Gram包含交叉块。M/F各自L2归一化，A缓存仅cast float64。U采用1/N平均，CB采用1/C类均衡，λ=0.001，无bias、无搜索。复制样本/重复嵌入、当前类guard、候选列、恢复、同一父模型内的重分块及类内逆序等价均通过；没有跨父模型乱序检验。原始A重现G1全部高精度逐类validation，精确tie按最小original_label，G1预测兼容通过。\n\n候选评分前锁定协议、代码、父状态和特征定义。完整72/432行（核心36/216）全部保留，无按分数删记录。K推理同时涉及不同路由、归一化和分类器，A1不能称“相对K只换了分类器”。图像cosine变化不证明信息损坏；M/F/J容量不同，重复嵌入控制也不使真实1536维和768维统计容量相等。预训练暴露、患者隔离UNKNOWN及数值标签的临床语义限制均保留。\n'
    text+=f"\n## 资源与访问\n\n运行wall {resources['wall_seconds']:.1f}秒，GPU进程驻留观测 {resources['GPU_process_residence_seconds']:.1f}秒（含驻留期间CPU工作），CPU解析累计 {resources['analytic_CPU_seconds']:.1f}秒；峰值RSS {resources['peak_RSS_bytes']/1024**3:.3f} GiB，GPU allocated {resources['peak_GPU_allocated_bytes']/1024**3:.3f} GiB。新增文件观测 {resources['new_file_bytes_observed']/1024**2:.2f} MiB，最低空闲 {resources['min_free_bytes']/1024**3:.3f} GiB。单worker、batch48、4线程、FP32，未降配。\n\n"
    text+=f"新双分支缓存特征行 {access['new_train_val_feature_rows']:,}；工程和正式图像读取分别 {access['image_reads']['probe']} 与 {access['image_reads']['formal']}。wrapper调用 {access['wrapper_batch_calls']} 次，wrapper图像行 {access['wrapper_image_rows']}。内部original/main/few调用分别 {access['module_batch_calls']}，合计 {completion['encoder_forward_calls']}；内部图像行另见ACCESS_AUDIT。工程probe与正式提取区分，不能把57,039行当成内部ViT调用数。\n\n"
    text+=f"正式阶段解析拟合 {resources['analytic_fits']} 次；包含隔离工程/恢复/等价核对的实际solve共 {resources['total_solve_calls_including_engineering']} 次。继续增量的三个视图×两个目标的累计统计约 {resources['continued_increment_all_views_two_targets_bytes']/1024**2:.2f} MiB；审计raw缓存payload另约 {resources['raw_cache_payload_bytes']/1024**2:.2f} MiB，运行RSS包含工作空间，三者均不是零记忆。只保存raw m/f，派生视图不落盘，所有旧资产原位引用。\n\n"
    text+='公开代码、锁和聚合；raw特征、个体身份、分数、W和精确路径私有，不声称统计天然匿名。原checkpoint、历史负结果及旧工作区未覆盖，不删除历史资产、不自动合并main。\n\n```text\n'
    for k,v in completion.items():text+=f'{k}={str(v).lower() if isinstance(v,bool) else v}\n'
    text+='```\n\nNEXT_DECISION=STOP。未启动新的S0训练、GSR/K修复、融合、编码器、外部数据、test或小时监测。\n'
    (pub/'FINAL_REPORT_ZH.md').write_text(text)
