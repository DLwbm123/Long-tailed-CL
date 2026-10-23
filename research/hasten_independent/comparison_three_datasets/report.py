import json,statistics as st
from pathlib import Path
root=Path(__file__).parent;rows=json.loads((root/'results/RESULTS.json').read_text());audit=json.loads((root/'results/PROTOCOL_AUDIT.json').read_text());r12=json.loads((root.parent/'r12_three_datasets/results/ANALYSIS.json').read_text());proto=json.loads((root.parent/'r12_three_datasets/DATASET_PROTOCOL.json').read_text());tr=json.loads((root.parent/'r12_three_datasets/results/TAIL_TRAJECTORIES.json').read_text())
lines=['# 三数据集旧方法与 HASTEN 完整对比','','## 结论','','目前 HASTEN H-unit 在三个数据集上均没有超过本次找到的完整旧路线。NIH 参照仍为 Spatial / CLS＋patch；ISIC/HK 找到的是旧 cf_group 完整方案及其 cf_linear/cf_prototype/cf_weighted 对照，不能把它们改名为 Spatial 或 CLS＋patch。全部数值来自已有封存结果，没有重训旧方法，也未调整正在运行的 R14。','','## 完整主表','','单位为百分比。旧路线均为单 seed74002；HASTEN 为三种子均值。平均采用阶段 BA 等权平均，尾类统一为完整训练数 <250。','','|数据集|方法|阶段平均 BA|最终 BA|最终尾类 BA|','|---|---|---:|---:|---:|','|NIH|Spatial（旧单种子）|33.2341|28.0000|23.8095|','|NIH|CLS＋patch（旧单种子）|34.0053|27.3333|19.0476|','|NIH|HASTEN H-unit（三种子均值）|32.3549|26.3333|12.0635|']
for ds in ['ISIC','HK']:
 for r in rows:
  if r['dataset']==ds and (r['method'].startswith('cf_') or r['seed']=='three-seed mean'):
   name=r['method']+('（三种子均值）' if r['seed']=='three-seed mean' else '（旧单种子）');lines.append(f"|{ds}|{name}|{r['mean_BA']:.4f}|{r['final_BA']:.4f}|{r['tail_BA']:.4f}|")
lines+=['','cf_group：旧完整方案；cf_linear：闭式线性头对照；cf_prototype：加入多原型证据；cf_weighted：直接梯度权重控制。旧报告已指出策略更新不等于非均匀权重实际执行，不能将 cf_group 的结果解释为已证明 RL 有收益。','','## 同编号种子对照及均值差','','同编号种子可以避免把三种子均值当作同一次运行，但骨干、模型和随机数调用不同，仍不是共享随机流的配对消融。差值为 HASTEN − 旧方法，单位 pp。','','|数据集|旧参照|HASTEN口径|平均差|最终差|尾类差|','|---|---|---|---:|---:|---:|']
for ds in ['ISIC','HK']:
 for method in ['cf_group','cf_linear']:
  old=next(r for r in rows if r['dataset']==ds and r['method']==method)
  for seed in [74002,'three-seed mean']:
   h=next(r for r in rows if r['dataset']==ds and r['method']=='HASTEN H-unit' and r['seed']==seed)
   lines.append('|'+ds+'|'+method+'|'+str(seed)+'|'+'|'.join(f'{h[k]-old[k]:+.4f}' for k in ['mean_BA','final_BA','tail_BA'])+'|')
lines+=['','HASTEN 三种子各自的 ISIC/HK 最终总体与尾类结果都低于这里的 cf_group/cf_linear 旧结果；不是只由一个较差种子拉低均值。原始 seed 结果完整保留于 results/RESULTS.json。','','## 数据与顺序核对','','以下记录级证据全部一致：旧 INPUT 的 manifest 引用与本次 config 指向同一目录；旧 DATA_PROTOCOL 的全部训练类别计数、每阶段验证类别计数与本次 SPLIT_SEAL 和当前清单一致；所有四条旧 shuffled 臂覆盖完整序列。','','|项目|ISIC|HyperKvasir|','|---|---|---|','|类别/训练/验证|8 / 18718 / 295|23 / 6821 / 1698|','|增量任务|4+2+2|13+2×5|','|类别顺序|[4,7,5,1,2,6,0,3]|[4,5,6,22,20,1,2,15,14,12,10,11,19,16,7,0,17,9,18,13,21,8,3]|','|完整训练<250的尾类原ID|[6,7]|[0,5,6,7,12,16,17,18,19,21,22]|','','**核对边界：旧聚合结果没有保存逐样本身份快照。** 因此确认的是同清单引用、同完整类计数、同逐阶段验证计数和同类别顺序；没有独立恢复逐行历史成员并证明未发生替换，不声称逐字节或逐身份历史一致。没有哈希或影像重编码。两者均在全部训练后评价开发验证集；ISIC/HK 已知 identity_component 不交叉不能扩大为患者级独立性。','','## 训练口径不同，不能当成单模块消融','','旧方法用 ImageNet 预训练 ViT-B/16、1536维 APART 特征及可训练适配器池；HASTEN 用冻结 BiomedCLIP 512维特征、图文适配器与双曲映射。旧方法每阶段2轮，HASTEN每阶段10轮，目标函数、图像预处理、优化器和信息使用都不同。','','旧方法每阶段按 identity_component 将当期训练池分为 fit/meta：适配器优化主要使用 fit，meta 参与控制器/门控；最终类别统计和头重拟合使用全部当期训练图像。HASTEN 使用全部当期训练样本并回放历史高斯统计。因此虽然总训练池记录匹配，不能声称优化样本和算力预算完全一致。','',f"旧 ISIC 适配器每臂468次更新，HK每臂176次；HASTEN对应每种子2930次、1090次。不同更新含义和闭式步骤成本不同，不能据此直接给出效率倍率。",'','## 尾类口径修正','','旧 final_tail_recall 是按训练频率排序后最低一半类别（奇数类包含中位类别），与本次 <250 不同。以下从原逐类召回重算，没有重新预测或拟合。','','|数据集/旧方法|旧报告尾类值|统一<250后|','|---|---:|---:|']
for r in rows:
 if r['method'].startswith('cf_'):lines.append(f"|{r['dataset']}/{r['method']}|{r['original_report_tail_BA']:.4f}|{r['tail_BA']:.4f}|")
lines+=['','ISIC 统一后的两个尾类为 dermatofibroma 与 vascular lesion；旧 cf 系列分别召回50%与88.46%，平均69.23%。HK 的极稀有类验证只有1–2张，这些类的宏召回对个别图像很敏感。','','## 逐阶段同口径对比','','|数据集|阶段|已见类数|cf_group BA|cf_linear BA|HASTEN三种子 BA|','|---|---:|---:|---:|---:|---:|']
for ds in ['ISIC','HK']:
 old=next(r for r in rows if r['dataset']==ds and r['method']=='cf_group');lin=next(r for r in rows if r['dataset']==ds and r['method']=='cf_linear');h=json.loads((root.parent/f'r12_three_datasets/results/{ds}_aggregate.json').read_text())['results']
 for i,s in enumerate(old['stages']):lines.append(f"|{ds}|{i+1}|{len(s['seen'])}|{s['BA']:.4f}|{lin['stages'][i]['BA']:.4f}|{st.mean(r['metrics']['final']['BA']*100 for r in h if r['stage']==i):.4f}|")
lines+=['','## 逐类最终对比','','旧方案均为 seed74002；HASTEN同时列同编号种子及三种子均值。']
for ds in ['ISIC','HK']:
 old=next(r for r in rows if r['dataset']==ds and r['method']=='cf_group')['stages'][-1];lin=next(r for r in rows if r['dataset']==ds and r['method']=='cf_linear')['stages'][-1];order=proto[ds]['order'];last=len(proto[ds]['task_sizes'])-1
 lines+=['',f'### {ds}','','|类别|训练/验证|cf_group|cf_linear|HASTEN 74002|HASTEN均值|','|---|---:|---:|---:|---:|---:|']
 for c,orig in enumerate(order):
  tt=[v for v in tr if v['dataset']==ds and v['stage']==last and v['class_index']==c];one=next(v for v in tt if v['seed']==74002);hm=100*st.mean(v['final_correct']/v['eval_count'] for v in tt)
  lines.append(f"|{proto[ds]['names'][orig]}|{proto[ds]['train_counts'][orig]}/{old['counts'][str(orig)]}|{100*old['recall'][str(orig)]:.2f}|{100*lin['recall'][str(orig)]:.2f}|{100*one['final_correct']/one['eval_count']:.2f}|{hm:.2f}|")
lines+=['','## 未混入主表的历史结果','','- 旧 ordered 臂使用不同类顺序，未与本次 shuffled 混排。','- Spatial 全序列当前证据为 NIH；没有找到 ISIC/HK 对应的完整 Spatial/CLS＋patch 序列。','- 旧 ISIC G/C 的4→5→6任务不是本次4+2+2的完整8类任务。','- 旧 ISIC/HK local_evidence_transfer 仅第一转移、Qfit/Qgate 子集合；HK层级/OPSA等诊断有13类支持池版本，不是完整23类。','- 上述局部实验的较高数值不填入本主表，也不据此宣称优于完整增量结果。','','## 证据位置与后续','','旧完整结果来自 three_dataset1_20261008 中 ISIC/HK 的四条 shuffled cf_* 轨迹，封存源提交71f9230。本目录 OLD_AGGREGATES.json 保存原聚合指标，PROTOCOL_AUDIT.json 保存匿名协议核对，collect.py 可在原主机只读复核。NIH对照来自既有 WINDOW_SUMMARY_ZH.md；HASTEN来自已核验 R12。','','这次检索修正了此前“尚未找到匹配旧完整结果”的状态：找到了 cf 系列，仍未找到完整 Spatial/CLS＋patch 对应项。R14继续按预先冻结协议运行，不据本次结果修改其损失、门槛、种子或预算。旧方法没有重训或调优。']
(root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
print('report written')
