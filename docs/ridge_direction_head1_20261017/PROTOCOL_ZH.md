# 完整岭头的单位列方向控制：预先冻结协议

## 假设和唯一改动

旧/新类内相关性的固定四格已交付，整体作用并非相加，宏指标部分受极少数样本影响。完整岭分类头各列长度不同；它既改变方向也改变类别特异logit幅度。本轮仅检验这些长度差异是否参与现有分类收益和损伤。

读取已封存完整raw_ridge头W，固定W'_c=W_c/||W_c||₂，保持每列方向和正号。两数据集各一个固定转换，不重新拟合，不调整任何参数，不扫描范数幂、λ、温度、bias或新旧权重。原始头来自源码 `703dd54c9150a013a386a2b77bd996d9e8c08743`、交付 `3b32f0d63280cd174502f821158d976063d16ad6`，启动时核对来源收据。历史原头不改。

模型保持原头随附的全零adapter，因此表示为既定raw单位特征。score=8 z W'，BETA固定8。单位列转换不是原岭目标的等价最优解，是标准方向控制，不宣称新方法。CE尺度同时改变，不能把CE降低单独视为校准成功；头方向、长度效应及分类排序应分开解释。

## 冻结范围、访问和判断

沿用seed74002、原类别顺序、ISIC旧4新2/HK旧13新2、同一Qfit/Qgate及所有划分。历史原头已用固定H和新S拟合，本轮无H、旧S、新S、teacher状态、保留E、原图、新标签读取；不拷贝banks，不运行特征编码GPU。仅先读原头及其交付来源收据，转换保存后才允许Qfit/Qgate及历史去身份化聚合结果。

完整raw_ridge为主要参考，raw_centroid为必要参考。完整保留原15参考：raw_centroid、MM、HH、identity_ETF、teacher_centroid、teacher_ETF、coordinate_median、centered_centroid、raw_ridge、raw_diagonal_ridge、raw_prototype_ridge、raw_isotropic_ridge、raw_within_diagonal_ridge、raw_new_covariance_ridge、raw_old_covariance_ridge；历史结果直接复用，不重跑。候选固定名raw_ridge_unit_columns。

每query总体BA严格高于参考，旧/新BA均不低于参考1pp，两个query同时满足才通过对应描述性门槛。主结论必须对完整岭报告，不因其他参考有利改主要指标。不按query选择部署或自动替换模型。报告all/old/new BA、full-seen softmax CE、每类recall/CE/count/correct、全部参考差值、微计数、每类原头范数/新范数/正缩放因子。逐类差候选标签和正确数必须核对。

query已反复用于设计，患者分组未知；所有结果均为开发机制证据，不是独立医学、跨种子或全阶段持续学习证据。

## 最小实现、检查与预算

独立小入口run_ridge_direction_head.py复用现有CPU队列、load/save/metrics/select。主队列调用、冻结JSON和中性包装入口一并更新；其他旧实验代码不变。完整头1536×6/15、float64/有限、模型全零且形状兼容；每列范数须>1e−12，否则停止，无epsilon裁剪或回退。归一化后列norm在1e−12内为1。

每dataset启动时运行二维断言：不改变输入，正公共倍率不改变单位方向，W=diag(2,1)、z=(.6,.8)时raw类0而单位列类1；零列和NaN必须被拒绝。访问守卫先用被禁止的支持/query/ref路径检查，转换封存前均应拒绝。query检查有限1536维、标签范围和每类非空。

0正式/工程矩阵解，0逐元素方程解，0梯度/适配器/教师/编码器/记忆更新。工程检查包含范数/除法，非免费；2个正式派生头32256个float64元素、258048字节252KiB。0支持前向，2155查询特征前向；无新的Gram或支持累计。不重计历史拟合成本，客户端准备为null，记录本轮CPU驻留/user/system/峰值/失败/超时和累计成本。

两CPU任务ISIC_direction、HK_direction，各900秒、总3600秒上限，GPU禁用。中性命令启动后一次检查并结束启动回合；由现有半小时监测后续收尾。数值/协议失败不自动改变规则或重跑，工程故障保留成本后最小修复未完成阶段。结果及报告公开交付后才推进有信息价值的后续实验。
