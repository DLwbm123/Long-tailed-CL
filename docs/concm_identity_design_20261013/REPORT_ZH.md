# MPC/DSM-inspired v2 设计交付：CPU通过，未启动医学实验

基于机制诊断49ad413，完成独立数学helper、CPU检查和贡献协议。当前production_integrated/GPU_launch_authorized/medical_data_access_authorized/checkpoint_access_authorized全部false。没有读取影像、现有医学checkpoint或val，也没有运行原worker、launch、collect、GPU任务或监测。

## 三个实质变化

1. **MPC借二阶形状，保持身份。** 保留mu、全部原型组件中心/质量/半径及类支持量，只以固定支持量和协方差形状相近核借用组件内部协方差。类间均值差及组件间多模态结构保留，R/pairs/local中心不变。这不是可信中心恢复，借用有效性仍待验证。
2. **DSM定位错误尾部，保持原解析曲率。** 以固定控制头W0和原训练聚合统计构造组件高斯混合的错误尾部矩。修正用全部输入维度和完整native competition；没有旧均值span限制、整个local span去重或旧能量硬上限。它仍是统计代理，不是直接观测真实旧类混淆。
3. **REG与ETF分别作用。** REG围绕原头的增量稳定；ETF只作尾部条件化的线性方向力，方向开关不改变Hessian。单开REG严格identity，不再新增等价医学臂。两者都不能按val选择或放大。

## 新类保护和代价

所有当前新列固定；推理时旧类最大分数只能不变或下降。由此当前真实新类raw margin不下降，已正确的新类不被夺走，又允许以前错成旧类的新类获救。旧类可能更多错成新类，新类precision/F1不保证改善；新→另一新无法修复。stage1全新类时全部候选identity，后来成为旧类的类别不继续享有该阶段保证。

这比前版固定旧最大值的guard多留了恢复空间，但仍是非线性readout，不能称整个受保护分类器是单一线性解析头。实际作用是否过弱必须报告；保护证明不能代替正效用。

## local_transport 互补

原local_transport的区域prototype metric及共同运输不改。MPC改组件内协方差，DSM改各自控制W0下尚可能错分的旧-旧尾部；已经改善的边理论上压力较小。没有全span投影去重，因此不因为两个方法共享方向就机械删除修正。是否重复/冲突仍通过固定交互、组合−单模块和负结果检验，不声称保证互补。

## CPU验收

- 最终 base/local两控制、八个数学开关的独立显式二次目标、稠密Hessian/梯度及最优解oracle PASS；最大PCG残差5.757722488e−10。
- 中心/组件/质量/半径/迹保持、共同平移不变、tail矩和补矩PSD、tail曲率不超过全native曲率 PASS；截断矩独立数值积分最大误差1.329807653e−9。
- 同REG、ETF开关Hessian相同；REG单开、无信息MPC/TAIL、少于2旧类全部严格identity；无tail时ETF方向严格0；checkpoint往返、版本/非法授权/坏支持及非对称Q拒绝 PASS。
- float32/float64合成23类输入的新列、raw margin及已正确新类保护 PASS；合成例同时展示新类获救、旧类内部纠正以及可能造成旧类false positive。
- 重叠类fixture尾部delta范数0.005317420，旧均值span外范数0.002392150（约45%）；八开关类别置换最大绝对差3.83e−17。不是医学准确率。
- 1536维ETF接口、23类guard接口 PASS；**未测试完整1536×23真实医学解析准备/求解、模型权重或GPU推理**。

首个易分离fixture尾部delta约2.8e−5，明确保留此弱作用结果。设计过程中发现整体旧contrast正则化会主导变化，最终改为增量稳定并重新完整验收；早期两个成功CPU检查作为历史记录，不当最终方法结果。

## 前瞻对照

未来获授权时，主四格固定A0/AL/B/P，其中B/P是两控制加完整MPC_DSM-v2；MPC、TAIL、TAIL_REG、DSM、MPC_TAIL、MPC_TAIL_REG、MPC_DSM七个固定性能头做嵌套贡献。REG为CPU identity，不额外医学评价。相同18stage原checkpoint、相同特征批共享七头，0重训；126 sidecar和真实4零预检的生产准入仍待接线。

正则化比较TAIL_REG−TAIL及MPC_TAIL_REG−MPC_TAIL；ETF比较DSM−TAIL_REG及主组合−MPC_TAIL_REG；补全与互补比较MPC−控制、组合−单模块和B/P交互。预先固定raw/guarded全部纠正/破坏、31类支持/各stage、原始/归一化margin、风险边与真实混淆对应、AIR差距及费用，负值不筛掉。详见PROTOCOL_ZH。

## 费用与工程记录

所有6次CPU子进程驻留共 **40.249652412秒**（包含初稿两次PASS、一次准入fixture失败和最终三次PASS）；最终三次驻留 **15.679466808秒**，其pipeline **4.471602591秒**包含在驻留中，不相加。一次失败7.671081086秒发生于合成控制构造：float组件质量与原native.evidence的Tensor接口不兼容，已修正fixture，原native源码不改。失败pipeline未单测保持null。

全部CPU父子入口中性且CUDA_VISIBLE_DEVICES为空；0GPU/0医学数据/0正式更新。准备与测试都在真实nfs4存储路径的小probe后执行；历史实验/数据/model/checkpoint及保护的未跟踪工具不改。GPU历史账本没有新增费用。源码与协议属于设计准备交付，不是正式实验完成回执。
