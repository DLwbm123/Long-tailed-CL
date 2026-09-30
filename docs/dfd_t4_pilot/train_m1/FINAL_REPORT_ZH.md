# DFD-T4-P-v1-M1：固定 Task4 方向机制实验

12 条 D/R 轨迹完成，新增 5500 步、120 epoch；四条件 24 行阶段指标、192 行逐类结果。FZ1 六项后置；原五条件计划未完整完成。
控制为 REBUILT_FIXED_CONTROL，历史逐样本一致性 UNKNOWN；Task3 W 为 REBUILT_FROM_VERIFIED_PARENT，独立历史 W 匹配 NOT_AVAILABLE。

|数据|方法|平均 BA|最差 seed BA|
|---|---|---:|---:|
|ISIC|D_T|59.152|56.961|
|ISIC|R_T|59.433|56.623|
|ISIC|FD10_T|58.873|56.146|
|ISIC|FD1_T|59.410|56.583|
|HK|D_T|80.161|70.195|
|HK|R_T|80.161|70.195|
|HK|FD10_T|80.161|70.195|
|HK|FD1_T|80.161|70.195|

## ISIC
M=FAIL；机制=FAIL；U1=FAIL；U2=FAIL；S=PASS。
结论：CLOSE_DFD_T4_PILOT。
- D_T-R_T: -0.281pp [-0.723, +0.138]。
- D_T-FD1_T: -0.258pp [-0.767, +0.245]。
- D_T-FD10_T: +0.279pp [-0.317, +0.894]。
- R_T-FD1_T: +0.023pp [-0.362, +0.417]。

## HK
M=FAIL；机制=FAIL；U1=FAIL；U2=FAIL；S=PASS。
结论：CLOSE_DFD_T4_PILOT。
- D_T-R_T: +0.000pp [+0.000, +0.000]。
- D_T-FD1_T: +0.000pp [+0.000, +0.000]。
- D_T-FD10_T: +0.000pp [+0.000, +0.000]。
- R_T-FD1_T: +0.000pp [+0.000, +0.000]。

U3=NOT_EVALUABLE_MISSING_FZ1；完整方法验收=PENDING_FZ1；frozen_superiority=NOT_ESTABLISHED；full_method_promotion=false。
区间为 class/component 共享配对 bootstrap 2000 次、seed64201，条件于固定模型和既定顺序；不重抽训练 seed，不是独立确认研究，未校正多重比较。HK 是每顺序前 8/23 类。控制恢复不是独立验证。
全部零召回、分母、restricted 诊断与错误流向见配套表。缺失历史纵向字段为 NOT_AVAILABLE。当前输入漂移不代表旧图像真实漂移。
私有终态、Q 来源、W、预测和锁通过独立备份回执验证；最终交付状态见 COMPLETE.json 与 DELIVERY_AUDIT.json。
NEXT_DECISION=STOP。
