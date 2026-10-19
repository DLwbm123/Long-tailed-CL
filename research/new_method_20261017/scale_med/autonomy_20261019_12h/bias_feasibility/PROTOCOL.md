# 原有bias更新的工程可行性审计（运行前封存）

这是训练侧工程审计，不是分类实验。零优化器更新、零分类头拟合、零风险评分；不读验证/测试/未来支持/原图。固定复用此前封存的原reader训练64图像素，取每原类前两张(索引0/1,16/17,32/33,48/49)共8张。该选择仅固定工程探针，不按结果选样。全部数据留规定NFS，不复制像素/模型/环境。

## 预定对象与控制
原APART g完整生产路径，既有原运行环境FP32；仅纳入锁定预训练权重文件中非head、以.bias结尾且在原backbone中逐名同shape存在的参数。排除adapter pool、routing/assigner、head、original_backbone、所有weight，不新增模块，不换层/池化/宽度。参数集合由锁定权重键定义，不能根据梯度大小择子集。

原g与封存8行cache maxabs<=1e-5；相同baseline重复和精确恢复bias后同一输入g需maxabs<=1e-7。固定seed74160的Rademacher1536维线性泛函作用于归一化g，仅检查autograd，不使用标签或训练分类器。全部目标梯度finite，总norm>0，至少一个前六block和一个后六block的bias有非零梯度。沿单位梯度方向±1e-3的中心差分，绝对误差<=1e-4+1%×|解析导数|；这是对称工程扰动，立即copy回原值，无学习率搜索/参数保留/优化器步。

另用seed74161生成每bias坐标固定±1e-4合成delta：原g有限且改变>1e-8；随后按逐名同shape映射交给现代原FP16 block6 h。h需有限、改变>1e-8，精确恢复零delta后输出exact复现本轮baseline；同一modern runtime中检验，不要求旧/新FP16跨运行时等价。无head/routing/pool参数混入、delta写入和恢复逐元素核对。原original_backbone作为既有冻结路由参考不更新；不能将其原有no_grad当作目标backbone梯度断路。

## G/P/C容量控制
固定原G1535/P1375/C1318及rank128，复用初始训练得到的mapper仅作形状模板。构造各臂完整容量模板：同尺寸上三角A、六类B/W、所有norm/U/V/PERM、bias delta字典及元数据；不隐去新增delta、不改宽度。实际序列化并回读，逐tensor等值、完整模板<=19,058,800bytes，全部费用计本窗口。模板不是学习后artifact或性能认证；训练后仍须重新实测。全pool运行时参数量只计数，不训练/存储整个pool，预计FP32下界47.6MB不合规，不能偷偷以half/rank变化救该路线。

## 隔离、预算和停止
共享预训练权重只读、原reader像素只读；不读任何测试缓存/标签/旧预测。g原环境、h现代环境分两独立子运行，完整EXIT/FAILURE/资源/协调回执，原阶段输出不覆盖；source/protocol先公开再启动。NFS挂载余量/写读probe/GPU0显存先核，所有cache/tmp显式NFS，单文件512MiB，预计新增<=128MiB，写前保守预留256MiB；全窗17GB软停/20GB硬限，03:29:18UTC新启动截止不变，不设时间GPU预算。

任一数值/键映射/恢复/梯度/容量条件失败，记UNRESOLVED_BIAS_FEASIBILITY，不启动分类训练、不改阈值/参数集/epsilon搜通过。全通过只记ENGINEERING_FEASIBLE_NOT_TRAINED，method_success=false，不称类1机制或部署收益。后续若进行分类比较，必须另封存固定训练日程、G/P/C及冻结/匹配监督控制、原划分/正则选择/统计家族/逐类保护，不用本审计产生的梯度或delta作训练初始化。
