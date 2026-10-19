# 封存协议：同一原G/P/C表示上的类条件协方差参照

## 科学问题、既有工作与差异

核读出已数值完成却未改善受保护风险。此前共享线性分数、独立OvR、成对边界和共享RBF核均不等于按类别拟合异质协方差。本轮检验固定特征上的类条件相关结构是否能改善实际工作点，而不是继续调核、损失、层或池化。它是已知强基线，不是新方法。

[FeCAM NeurIPS2023原文](https://proceedings.neurips.cc/paper_files/paper/2023/file/15294ba2dcfb4521274f7aa1c26f4dd4-Paper-Conference.pdf)使用类原型、协方差收缩和相关归一化；本文已核方法及实现设置，其many-shot使用γ1=γ2=1。本轮固定该一组γ，不扫描。原映射特征有符号，保留原特征故不应用其Tukey平方根；组/图权重也沿用本项目。因此仅称signed-input FeCAM-inspired covariance reference，不称官方完整复现。SLARC的类/池化/球形Gaussian头同样覆盖此已知方向，不能改名创新。

复用审计：旧run_b_local/run_v2是局部patch的Gaussian密度与固定B融合、不同支持/校准/开发query及表示；不是本轮单独G/P/C类协方差。其volume消融负结果保留；本轮必须有显式logdet强控制，不能忽略这一教训。原information_attribution也没有同口径本轮五种几何头。

## 数据和无调参规则

原74006划分，ISIC0..3→4→5，训练2555/1812/28，完整G1535、P1375+128、C1318+128，原归一化/变换/行序冻结。只读OvR本窗各到达stage.features缓存，零原图/新特征/新数据。初始验证681及以后验证442/8仅核互斥，不拟合、不选参数、不评分。所有新头无优化alpha或验证校准；原ridge等既有控制的原选择保持不动。固定γ来自原文，不是看本轮测试后选。

类内权重为原类/已知组/组内图权重，归一至每类总量1。仅新到达类计算μ_c和Σ_c=Σ_i w_i(z_i−μ_c)(z_i−μ_c)'；原最后常数截距列去除，不改变其余特征。是加权population covariance，不声称患者iid估计或无偏性。旧样本/特征到达后退休，后续只用保留的均值/协方差统计；运行guard拒绝打开旧stage输入及缓存。每类统计一次，随后不改；共享头以已见类cov等权均值更新。

设v1=trace(Σ)/d，v2为所有非对角元素均值；S=Σ+v1 I+v2(11'−I)，固定γ1=γ2=1。C=diag(S)^−1/2 S diag(S)^−1/2。不因尾类切换到另一γ。v1或对角≤1e−12、非finite、不对称或Cholesky失败均记未决，不追加jitter或修改γ。

五种新读出、同一μ/Σ/γ，无拟合门控或融合：

1. NCM：−||z−μ_c||²。
2. SharedM：共享pooled Σ（见类等权平均）同样收缩/相关归一化后Mahalanobis。
3. RawM：每类S的Mahalanobis负距离。
4. QDA：−.5[每类S的距离+logdet(S)]，等类先验。
5. CorrM（唯一主候选）：每类C的Mahalanobis负距离。

距离打分不等于异协方差下完整Gaussian概率；QDA控制体积项，不能称CorrM为已证最优Bayes。SharedM控制类别特异性，NCM控制是否仅原型就足够，RawM控制相关归一化。均固定argmax/并列低编号，不测试校准。

## 数值、隔离、内存

3表示×6出生类=18类统计单元，各含Raw/Corr两Cholesky；3表示×3阶段=9共享Corr单元，共27统计单元/45因子。float64，矩阵对称误差≤1e−10、Cholesky重建相对max误差≤1e−8、正对角、logdet与距离finite。合成数据核加权均值/协方差、共享pool、原/direct Gaussian距离及logdet、类分数退休不变、SPD/因子重建、完整15新臂21比较JSON及失败回执。

每阶段重建原A/B与封存统计parity≤1e−8。全部模型/固定γ封存在测试输入前，query不拟合或校准。只在封存后复用原测试特征及kernel_readout封存30方法预测；新增15臂共45方法135阶段单元。主CorrM各自旧类单列分数出生到退休应保持exact；SharedM允许因pooled更新变化。

原类协方差稠密状态明显超过19,058,800bytes单方法预算，必须报告各臂名义状态和真实运行文件占用，即使性能通过method_success=false。这是超预算统计参照，不宣称原有限状态方法已改进。历史缓存保留仅为研究资产，guard的退休不等于物理删除或隐私证明。

## 科学门与停止

G/P/C CorrM各对原G/P/C ridge和自身NCM/SharedM/RawM/QDA，共21比较。主J为两未来阶段imageBA平均；10000类内已知组配对bootstrap，seed74158，95%和21项Bonf[.05/42,1−.05/42]。历史测试已暴露，均探索性。

一主臂同时：J对G ridge≥1pp，对另外两ridge及四个几何控制点差均正，七调整下界全正；两未来阶段BA对自身ridge不降；类4出生/转旧、类5出生召回不降；两未来旧0..3及末旧0..4宏损失≤1pp；类0/1在三阶段各自召回损失≤1pp。通过仅OVER_BUDGET_COVARIANCE_HEADROOM。完整不通过STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION，不换γ/归一化/Tukey/先验、不择四控制最佳升主。缺失或未认证为工程/数值未决，不当科学否证。

## 资源与交付

本12h授权截止2026-10-20 03:29:18 UTC；无耗时预算，纯CPU4线程、CUDA_VISIBLE_DEVICES空。缓存/tmp全在本窗NFS，不复制模型/环境/旧特征。稠密统计及Raw/Corr因子私有保存，预计新增<3GiB，单文件<64MiB；先空间预分配核算和NFS探针，父每2秒扫描17GB软停，单文件512MiB上限，所有新增本地/远端/Git/缓存合计<20GB，保留1GB本地预留，不删除旧资产。

预检通过并公开源码/协议后一次真实运行；失败保留partial/FAILURE/EXIT/资源/父wait4。完成核18类/9pool/45因子、15新臂135cells/21比较、finite、A/B和原ridge得分对齐、封存/退休分数不变、过程退休和成本。源码/协议/全聚合/成本/实质中文报告公开push核SHA与匿名访问，DELIVERY本地远端保存；图像、身份、均值/协方差数组、头/特征/逐例输出/原日志/私有路径不公开。
