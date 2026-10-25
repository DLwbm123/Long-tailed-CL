"""Build a source-attributed main-candidate comparison; no training or new scoring."""
import json,csv,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];rows=[]
def nums(cell):
 return [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?',cell.replace('−','-').replace('±',' ± '))]
def report_table(folder,mode='three',col=1):
 p=ROOT/folder/'RESULTS_ZH.md';found={}
 for line in p.read_text().splitlines():
  if not line.startswith('|'):continue
  cells=[x.strip() for x in line.strip().strip('|').split('|')]
  if cells[0] not in ['NIH','ISIC','HK'] or cells[0] in found:continue
  if mode=='three':v=[nums(cells[i])[0] for i in [1,2,3]]
  else:v=nums(cells[col])[:3]
  assert len(v)==3;found[cells[0]]=v
 assert len(found)==3,(folder,found)
 return found

def add(id,name,protocol,verdict,values=None,source=None,missing='未实测',kind='candidate'):
 rows.append(dict(id=id,name=name,comparison_scope=protocol,verdict=verdict,values=values or {},source=source,missing=missing,kind=kind))
add('official','原论文配置原样 HASTEN','未在本医学三域协议实测','不能与下列医学适配数值混称原版结果',missing='未实测',kind='unmeasured')
add('R01','早期 HASTEN 兼容版','HK13类短任务，非当前HK23类任务','已有短任务实测；不填入不同任务的本表数值',source='r01/PROTOCOL.md',missing='不同协议实测，不可填此格',kind='incomparable')
add('R02','OpenAI CLIP-B/32 HASTEN兼容版','NIH相同任务，1seed；含机制/数值修复','不是原版逐位复现，也不是三域3seed参照',{'NIH':[19.8448,13.6667,6.6667]},'r02_nih/RESULTS_ZH.md',kind='reference')
add('R03','BiomedCLIP HASTEN原尺度','NIH同数据任务，1seed；编码器系统改变','比R02数值高；骨干/预处理/分词器均变，不能单归因语料',{'NIH':[28.8446,22,11.4286]},'r03_biomed_nih/RESULTS_ZH.md',kind='reference')
raw=json.loads((ROOT/'r04_unit_scale/results/ANALYSIS.json').read_text())['summary']['H-raw'];add('R04raw','BiomedCLIP原尺度配对参照','NIH3seed，R04单位化的同次匹配对照','用于核实H0尺度修正的真实提升',{'NIH':[raw['mean'],raw['final'],raw['tail']]},'r04_unit_scale/RESULTS_ZH.md',kind='reference')
h0={'NIH':[32.354939,26.333334,12.063493],'ISIC':[55.381695,52.255438,59.188034],'HK':[56.491763,57.509473,37.822500]}
add('H0','H0：单位化医学HASTEN','NIH3seed配对；ISIC/HK3seed完整扩展','✓ NIH相对R04raw平均+3.9425/最终+4.3333pp，3/3同向；当时预设门通过。未测ISIC/HK raw配对效应',h0,'r04_unit_scale/RESULTS_ZH.md',kind='retained_reference')
for id,folder,key,name,verdict in [
 ('R05','r05_balanced','均衡采样完整版','类别均衡采样','NIH尾升、平均/最终下降；局部收益'),
 ('R06','r06_zero_init','Z-full','增量零初始化','NIH三指标下降'),
 ('R07','r07_parent_text','T-full','父节点文本路由','NIH平均/最终下降、尾持平'),
 ('R08','r08_class_ce','C-full','全已见类CE','NIH三指标下降'),
 ('R09','r09_hierarchy_ablation','A-full','移除层级损失','NIH尾升、平均/最终下降；局部收益'),
 ('R10','r10_replay_unit','R-full','回放单位化','NIH平均微升，最终/尾下降'),
 ('R11','r11_stable_adapter','S-full','冻结新增适配器','NIH平均/最终下降、尾持平')]:
 lines=(ROOT/folder/'RESULTS_ZH.md').read_text().splitlines();line=next(l for l in lines if l.startswith('| '+key+' |'));cells=[x.strip() for x in line.strip('|').split('|')];values={'NIH':[nums(x)[0] for x in cells[1:4]]};add(id,name,'NIH3seed匹配，其他域未测',verdict,values,folder+'/RESULTS_ZH.md')
for id,folder,name,verdict in [
 ('R14','r14_batch_multiplicity','批内重复频次修正','三域均下降'),
 ('R16','r16_balanced_readout','类别均衡读出','NIH尾升/最终降；ISIC最终与尾升，仅局部'),
 ('R17','r17_centered_fusion','中心化融合','HK阶段平均升，最终/尾下降'),
 ('R19','r19_train_teacher','TRAIN教师','ISIC三指标真实数值提升、joint3/3；NIH下降'),
 ('R21','r21_consistent_teacher','一致性教师','ISIC/HK局部提升；NIH最终下降')]:
 scope='三域3seed；H0系预注册训练对照' if id in ['R14','R19','R21'] else '三域3seed；同H0已训模型的读出对照'
 add(id,name,scope,verdict,report_table(folder),folder+'/RESULTS_ZH.md')
for id,folder,name,verdict in [('R22','r22_supervised_anchor','全覆盖标签锚定','HK三项微升；NIH下降'),('R23','r23_centroid_anchor','类别中心锚定','HK局部升；NIH最终下降/ISIC尾下降，迁移差异披露')]:
 add(id,name,'三域3seed；H0系对照，新增辅助计算',verdict,report_table(folder,'cell',-1),folder+'/RESULTS_ZH.md')
add('R24core','R24同次core参照','三域3seed；R24应先与同次core比较','不把数值实现差异当成head效果',report_table('r24_detached_readout','cell',2),'r24_detached_readout/RESULTS_ZH.md',kind='reference')
add('R24','独立detached分类头','三域3seed；匹配同次core、历史H0另报','未稳定优于同次core；HK平均升但最终/尾降',report_table('r24_detached_readout','cell',3),'r24_detached_readout/RESULTS_ZH.md')
add('R25','GELU图像适配器首次配置','NIH未完成初始阶段；无完整VAL','数值失败，不能归为分类性能负结果',source='r25_nonlinear_image/RESULTS_ZH.md',missing='失败，未形成完整评估',kind='incomplete')
add('R28linear','R28线性FP64参照','三域3seed；共同entailment FP64公式','精度匹配参照，不把与H0微差冒称结构提升',report_table('r28_precision_nonlinear','cell',1),'r28_precision_nonlinear/RESULTS_ZH.md',kind='reference')
add('R28gelu','R28 GELU适配器','三域3seed；与R28linear配对同精度','NIH/ISIC尾微升但最终下降；未通过',report_table('r28_precision_nonlinear','cell',2),'r28_precision_nonlinear/RESULTS_ZH.md')
for id,folder,name,verdict in [
 ('R30','r30_replay_boundary','边界回放','NIH三项实测升，但平均/最终不足+1pp；跨域受损'),
 ('R31','r31_functional_anchor','功能KL锚定','ISIC尾升但总体降；NIH/HK退步'),
 ('R32','r32_visual_hierarchy','视觉层级叶节点','ISIC/HK三项局部升；NIH总体下降'),
 ('R35','r35_real_feature_replay','固定真实特征记忆','总体/尾类未获稳定收益'),
 ('R37','r37_birth_text_route','出生文本路由','ISIC局部升；NIH最终下降'),
 ('R38','r38_text_positive_pool','文本正集合目标','HK三项实测升、joint3/3；NIH下降/ISIC尾下降'),
 ('R40','r40_geometry_priority','几何梯度优先','近参照；NIH尾微升但最终降'),
 ('R42','r42_initial_identity','初始恒等结构','HK三项实测升、joint3/3；NIH三项降'),
 ('R46','r46_covariance_replay','协方差收缩回放','NIH最终计数持平、尾微升；HK尾下降')]:
 add(id,name,'三域3seed；FP64条件，需对H0及R28双报',verdict,report_table(folder),folder+'/RESULTS_ZH.md')
add('D51–53','三项2epoch小样本结构pilot','TRAIN折分且每类截64；非完整增量评测','有短pilot实测，但不等价本表任务；不足以否定结构',source='TRAINING_SUFFICIENCY_AUDIT.md',missing='仅短TRAIN pilot，无本表完整结果',kind='pilot')
for arm,name,conclusion,col in [('native','R54 native匹配控制','重现R28linear汇总，验证对照一致',1),('flat','R54平坦决策','HK三项实测提升；NIH下降，未过整体门',2),('factorized','R54粗细因子决策','ISIC三项升；NIH/HK下降，未过整体门',3)]:
 add('R54-'+arm,name,'三域3seed；同次三臂完整日程，预算较短pilot充分',conclusion,report_table('r54_training_sufficiency','cell',col),'r54_training_sufficiency/RESULTS_ZH.md',kind='reference' if arm=='native' else 'candidate')
for id,folder,name,arm,verdict in [
 ('R55','r55_balanced_analytic','均衡解析判别','balanced','HK平均/最终升，尾降；NIH改善不足'),
 ('R57','r57_shared_metric_memory','共享低秩＋真实历史','shared_metric','三域不支持，近秩1与预测近不变'),
 ('R59','r59_distributional_discrimination','分布支持＋共享度量','distribution_shared','HK平均/尾升但最终降；NIH/ISIC下降'),
 ('R61','r61_semantic_global_representation','共享表示＋全局语义风险','semantic_global','ISIC尾升；NIH下降，未过整体门'),
 ('R62','r62_visual_backbone','骨干内部视觉LoRA','vision_lora','NIH平均升但最终/尾下降；三域尾均低于H0')]:
 a=json.loads((ROOT/folder/'results/ANALYSIS.json').read_text());v={d:[a[d][arm]['mean'][k] for k in ['mean_BA','final_BA','tail_BA']] for d in ['NIH','ISIC','HK']};add(id,name,'三域3seed同评测；表示/记忆/训练曝光或算量改变',verdict,v,folder+'/RESULTS_ZH.md')
add('R63','支持/查询分离的视觉适配','完整三域3seed预注册；运行中','未形成已核验的完整结果；不填中间VAL或预测效果',source='r63_support_conditioned/PROTOCOL.md',missing='运行中，完整结果待定',kind='running')
add('D系列','其他机制/零步/统计诊断','TRAIN审计、局部方向或统计估计','不是同任务完整分类实验，不能充当accuracy提升',source='POST_R40_EVIDENCE_REVIEW.md',missing='不适用：非完整分类结果',kind='diagnostic')
for r in rows:
 for v in r['values'].values():assert len(v)==3 and all(0<=x<=100 for x in v)
OUT=Path(__file__).parent;(OUT/'COMPARISON.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
with (OUT/'COMPARISON.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['id','method','comparison_scope','NIH_avg','NIH_final','NIH_tail','ISIC_avg','ISIC_final','ISIC_tail','HK_avg','HK_final','HK_tail','verdict','source'])
 for r in rows:w.writerow([r['id'],r['name'],r['comparison_scope']]+sum([r['values'].get(d,[r['missing']]*3) for d in ['NIH','ISIC','HK']],[])+[r['verdict'],r['source']])
head='''# 原始 HASTEN、H0 与全部主候选：同表核对

每格为 **阶段平均宏BA / 最终宏BA / 最终尾类BA（%）**；尾类采用完整TRAIN图像数<250。除明确标注单seed/短任务/未完成外，完整结果均为74002/74003/74004三seed均值。数值保留两位小数，结论依据原精确计数/冻结门，不把浮点近零变化当提升。

**必须纠正的结论：不是“所有改进都没提升”。H0输入单位化相对同次Biomed原尺度对照，在NIH三seed平均/最终均提升且3/3同向，通过当时事前门。后续有真实但局部的单域/单项收益；尚无新方案通过完整跨域双参照采用门。**

“相同协议”需要分层：同数据划分、类别顺序和终点评价允许比较任务性能；同骨干、初始化/日程、精度、记忆与计算匹配，才支持指定干预的较强内部对照结论。表中明确写出差异，不能把同评测误写成所有训练条件相同。多数跨轮H0比较并非同一硬件/精度/曝光完全匹配；R28及R54native是后期数值精度的必要控制。研究结果为反复开发，不是独立测试显著性证明。

原论文配置逐位不改的三域医学复现**没有实测**。R02是OpenAI CLIP-B/32医学兼容/投影修正版，R03/R04使用BiomedCLIP，均不能冒称原版。H0=R04 H-unit（NIH）+R12同配置的ISIC/HK扩展；H0不等于原始HASTEN。

本表列每轮预注册主候选和解释必需的匹配参照；不将每个无投影、冻结、置换或读出诊断臂都当“新方案”。完整控制臂与全seed/全class正负结果见逐行来源，没有按有利结果挑控制替代主候选。R01已实测的是另一短任务，D51–53有TRAIN小样本实测，R25没有完整性能评估，不能混写成0或“完整无效”。

| 方案 | 比较口径 | NIH 平均/最终/尾 | ISIC 平均/最终/尾 | HK 平均/最终/尾 | 证据结论 |
|---|---|---|---|---|---|
'''
lines=[]
for r in rows:
 name=r['id']+' '+r['name'];name=f"[{name}](../{r['source']})" if r['source'] else name
 vals=[' / '.join(f'{x:.2f}' for x in r['values'][d]) if d in r['values'] else r['missing'] for d in ['NIH','ISIC','HK']]
 lines.append('| '+' | '.join([name,r['comparison_scope']]+vals+[r['verdict']])+' |')
tail='''

## 哪些结论可以成立

- **较强匹配实测提升：H0的尺度修正。** R04同Biomed骨干三seed的单位化−原尺度平均+3.9425pp、最终+4.3333pp、尾约+0.6349pp。初始6阶段部分复用同配置封存轨迹，成本已披露。该NIH结论不能直接扩展为ISIC/HK的单位化因果收益，因为那两域没有raw对照。
- **真实但局部：** R19在ISIC三指标提升且final/tail joint3/3；R38/R42在HK有正向配对证据；R54flat对同次native在HK改善。这些不是“假提升”，但不能代替NIH目标和跨域保护。R30在NIH三项有数值改善，但平均/最终未到+1pp，且其他域受损。
- **不能说整体更好：** R62的NIH阶段平均+2.03pp伴随最终−3.33pp、尾−5.08pp；R61的ISIC尾改善伴随NIH退步。改善某一格与通过整体采用门是不同命题。
- **尚未知/不可评估：** 原样原论文医学三域未测；R25数值失败无完整评估；短TRAIN pilot非完整增量结果；R63须等全三域全seed完成，不能用训练进度填性能格。

后期完整采用门保持：相对H0与精度匹配R28，NIH阶段平均/最终各至少+1pp、尾至少+20/7pp，且至少2/3seed最终/尾同时改善；ISIC/HK最终/尾各最多降1pp。早期R04的事前开发门与这个后来完整门不同，不能追改旧门或用新门否认已成立的内部提升。

R63继续原冻结协议，不因本表改变目标/日程、访问中间VAL或挑选seed。全部结果来自历史开发集，无独立TEST/患者外确认/临床效果声明。数据、标签身份、逐例响应、权重、原日志与私有配置路径不在此表。
'''
(OUT/'RESULT_COMPARISON_TABLE.md').write_text(head+'\n'.join(lines)+tail)
print('rows',len(rows))
