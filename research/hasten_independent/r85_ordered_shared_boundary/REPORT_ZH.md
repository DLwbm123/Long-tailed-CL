# R85：等级邻接信息的边界共享诊断

CPU-only，原R84固定特征与S/Q；36新监督CPU头拟合，原NCM/unstructured/natural只读缓存，不重训或重评分；所有新头先seal后Q，无新视觉编码/视觉optimizer/controller/VAL/TEST。

事前必要门 **False**，全六项同时正阶段 **1/6**，极少类门 **False**。

| ordered 减参照 | 宏 pp | 尾 pp |
|---|---:|---:|
| unstructured | -0.049964 | +0.016416 |
| degree_only | +0.008780 | +0.029643 |
| family_mean | +0.014639 | +0.029643 |
| NCM | +0.253789 | +0.254052 |

| stage | 宏 vs unstructured | 尾 vs unstructured | 宏 vs degree | 尾 vs degree | 宏 vs family_mean | 尾 vs family_mean |
|---|---:|---:|---:|---:|---:|---:|
| 0 | -0.080128 | +0.000000 | -0.026709 | +0.000000 | +0.000000 | +0.000000 |
| 1 | -0.046296 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 2 | -0.102124 | +0.000000 | -0.020425 | +0.000000 | -0.020425 | +0.000000 |
| 3 | -0.551568 | -0.909091 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 4 | +0.329369 | +0.628796 | +0.093164 | +0.177858 | +0.093164 | +0.177858 |
| 5 | +0.150966 | +0.378788 | +0.006648 | +0.000000 | +0.015097 | +0.000000 |

固定单位图先验，不搜索graph/强度/类别顺序。degree_only匹配逐类对角收缩；family_mean匹配总边质量但逐类degree不同，此限制保留。标签顺序仅建模先验，不代表线性临床严重度或重新标注。全部23类与Qa/Qb分解保留；class1fold1无双component覆盖，不补证据。以前源模型见过TRAIN，重复stage/fold非独立患者/新seed。监督信息共享即使有效也不能归GRPO；主门不足不改系数救援，所有R80-R84负证据保持。

## 目标类和解释边界

原目标类12的正确数对unstructured/degree_only/family_mean/NCM均无变化，12个stage-fold单元全同；类1也全同。类2仅保留R84相对NCM一次纠正，没有对三种新控制的净增量。因此，当前相邻等级共享没有显示它所针对的稀少等级识别收益。仅1/6stage全部六增量正，不选择更好阶段、不扫图强度。

ordered相对NCM平均宏/尾+.253789/+.254052pp，包含原R84监督头已有的增量，不能全部归新结构。最后stage相对NCM宏+.066648pp、尾-.430369pp，仍不满足最终尾门。degree_only精确匹配逐类对角收缩，family_mean匹配总边质量；小幅相对增量并不建立稳定的语义顺序价值。

R85假设由R84的Q错误聚合启发，又复用这些TRAIN Q划分，属于重复开发，即使门通过也不是独立确认。依据中的27/31是重复阶段预测机会，不是31名独立患者或新增图片。原论文提供等级与中间标签定义，不保证严格线性严重度；不更改既有manifest标签或把类别合并成更容易的评测。

## 核验和费用

全部12cells/36新CPU头，496L-BFGS迭代/606总closure evaluations（每头仍守原上限），最大最终梯度1.12297118e-6；全部头先seal后87651行新分类器TRAIN Q评分。原三参照0重训/0重评分，另2行保存TRAIN特征native工程分类评分单独计费，0新视觉编码/视觉optimizer/controller/VAL/TEST。

CONTROL_EXIT exit0，成功guardworkerwall40.334458s/CPU52.166699s，原钟退出411.857353s；workerRSS553213952B/group2s峰542400512B，受守护磁盘峰153933875B。新1GB硬cap含512MiB早停余量，入场近期保护27489064187B，未删除历史science。全部source features只读复用，所有新head/optimizer/scores及旧产物保护，root/remote spill0，无GPU进程。

两次启动前metadata setup错误均在0software/fit/score时审计修复，原clock不重置；其CPU未独立采样非0，elapsed已包含。软件3toyCPU头；saved-source-binding补核CPU4.444220s/wall6.398573s/RSS421797888B，无新forward/optimizer。文档和交付CPU未独立采样非0。详见ENGINEERING_NOTES及COST。
