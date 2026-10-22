# HASTEN与旧Long-tailed CL路线的数值比较

2026-10-22。HASTEN r01已完成36阶段/4560更新。BA=类别召回宏平均；所有数值为百分数。不同数据集、划分及训练协议分别列出，不能跨表排名。

## 1. HASTEN独立路线：同一5+4+4协议，三个种子均值

|臂|阶段平均BA|最终BA|最终旧类BA|最终新类BA|
|---|---:|---:|---:|---:|
|H-full|67.0495|71.7951|73.0680|69.2494|
|H-no-tree|65.4830|66.1534|73.1471|52.1659|
|H-shuffle|65.2515|66.3661|73.2835|52.5315|
|H-no-proj|68.0277|74.5892|73.0925|77.5826|

## 2. 相同HK留组集合的旧路线

以下专门取旧实验outer fold0，与HASTEN最终阶段使用相同支持划分、相同13个候选类别及12个可评估类别（痔无留组）；每类分母相同。旧实验联合训练全部初始13类，HASTEN按5+4+4增量训练。骨干/目标、训练预算、输入尺度、融合规则也可能不同，所以这是同评估集合的系统性能比较，不是单因素或同预算排名。旧四折OOF均值不混入此表。

|旧实验|骨干|臂|fold0 BA（三种子均值）|
|---|---|---|---:|
|support_scale_abcd|CLIP|A|61.2977|
|support_scale_abcd|CLIP|B|69.9912|
|support_scale_abcd|CLIP|C|70.0255|
|support_scale_abcd|CLIP|D|70.1013|
|support_scale_abcd|BiomedCLIP|A|66.3147|
|support_scale_abcd|BiomedCLIP|B|71.7927|
|support_scale_abcd|BiomedCLIP|C|71.8874|
|support_scale_abcd|BiomedCLIP|D|71.5802|
|hierarchy_geometry|CLIP|Eflat|70.1119|
|hierarchy_geometry|CLIP|Etrue|68.6581|
|hierarchy_geometry|CLIP|Eshuffled|68.7453|
|hierarchy_geometry|CLIP|Hflat|69.0693|
|hierarchy_geometry|CLIP|Htrue|68.5570|
|hierarchy_geometry|CLIP|Hshuffled|69.0468|
|hierarchy_geometry|BiomedCLIP|Eflat|72.1133|
|hierarchy_geometry|BiomedCLIP|Etrue|71.6027|
|hierarchy_geometry|BiomedCLIP|Eshuffled|71.3554|
|hierarchy_geometry|BiomedCLIP|Hflat|72.6310|
|hierarchy_geometry|BiomedCLIP|Htrue|72.6648|
|hierarchy_geometry|BiomedCLIP|Hshuffled|72.4311|

A=NCM，B=纯适配器，C=固定融合，D=尺度融合；E/H分别欧氏/双曲，true/flat/shuffled分别真实/无/打乱层级。旧几何实验为简化模型，不能改称完整HASTEN。

HASTEN完整臂71.7951%，相对旧CLIP固定融合70.0255%为+1.7696pp；相对旧BiomedCLIP固定融合71.8874%为-0.0923pp。关闭投影74.5892%是本表点值最高的候选，但不是已确认优胜路线。

## 3. 旧HK外部于支持池的E/Qgate（另一评估集合）

|路线|合并E/Qgate BA|
|---|---:|
|非CLIP原始F|65.6993|
|非CLIP仅头H|67.4531|
|非CLIP线性适配L|64.9541|
|非CLIP OPSA无MAPC N|64.4833|
|非CLIP OPSA+MAPC M|64.2892|
|CLIP平坦融合|70.2313|
|CLIP真实层级融合|70.3097|
|BiomedCLIP平坦融合|67.9184|
|BiomedCLIP真实层级融合|66.7663|

这里是1374张、13类，不能与HASTEN的支持fold0数值直接相减。历史13+2类Qgate上的A/B也不是这个初始13类协议。

## 4. 旧ISIC与NIH路线（不同任务）

|数据及协议|路线|阶段平均BA/宏准确率|最终或终点BA/宏准确率|
|---|---|---:|---:|
|ISIC 4→5→6，平均仅两个增量阶段|G全局表示|53.0442|未在本表摘录|
|同上|P普通PCA补充|53.2527|未在本表摘录|
|同上|C条件残差PCA|53.4868|未在本表摘录|
|ISIC初始4类组外终点|原始F|不适用|64.7136|
|同上|仅头H|不适用|65.0737|
|同上|OPSA+MAPC M|不适用|60.3654|
|NIH author20单标签开发，6阶段|全CLS ridge|32.01|25.33|
|同上|SCALE-Med v1 global_local|21.90|17.33|
|同上|SCALE-Med v2 ridge_local|34.28|26.33|

## 5. 该如何解释“谁更高”

HASTEN当前最高点值是无投影74.5892%，完整臂71.7951%；但两者差2.7941pp中2.7778pp来自同一张UC1–2图像在一个种子上的正确性差异。完整臂胜无树/打乱约5.6/5.4pp，也几乎由该图在两个种子中救回贡献。数字值得保留，不能等价为广泛提升。

原方法ISIC的C比G增量平均BA+0.4426pp，原区间跨0；NIH v2比ridge最终+1pp，属于另一开发任务。它们不能与HK的71–75%横向比较。HASTEN独立路线有更高的HK局部点值，但尚未证明是项目最强的稳定方法。

## 证据

- [HASTEN本轮结果](r01/results/ANALYSIS_ZH.md)
- [旧A–D](../new_method_20261017/scale_med/support_scale_abcd/results/aggregate.json)
- [旧几何](../new_method_20261017/scale_med/hierarchy_geometry/results/aggregate.json)
- [早期非CLIP/CLIP/Biomed汇总](../new_method_20261017/scale_med/EXPERIMENT_REPORT_NONCLIP_CLIP_BIOMEDCLIP_20261022_ZH.md)
- [ISIC原方法](../new_method_20261017/scale_med/PRO_HANDOFF_METHODS_RESULTS_20261019_ZH.md)
- [NIH v1](../new_method_20261017/scale_med/results/REPORT.md)
- [NIH v2](../new_method_20261017/scale_med/results_v2/REPORT.md)
