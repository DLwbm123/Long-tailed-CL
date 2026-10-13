# 贡献对照协议：MPC/DSM-inspired v2（设计冻结，尚未授权运行）

## 当前范围

本轮只交付设计、独立helper和合成CPU检查，不读取医学影像或checkpoint、不调用既有run/launch/collect、不修改既有科学源/预算/参数/进程、不恢复监测。没有新医学实验或GPU预检。`PROTOCOL_LOCK`全部授权字段为false。

设计依据为公开机制诊断49ad413。理论与假设见DESIGN_ZH；CPU数值见CPU_COMPATIBILITY_CHECK、CPU_STRESS_CHECK。官方ConCM/作者实现身份不成立，标注非官方。

## 将来如获授权，固定资产和主表

复用`docs/concm_statistical1_20261013/PROTOCOL_LOCK.json`里原base及local_transport科学源/交付/数据/order/seed/阶段checkpoint/预算及AIR结果。不重训控制；主方法固定local_transport，不按val换成base或挑候选。

ISIC order=[4,7,5,1,2,6,0,3]，3stage；HK order=[4,5,6,22,20,1,2,15,14,12,10,11,19,16,7,0,17,9,18,13,21,8,3]，6stage。原两个控制共18stage checkpoints，train/val支持18718/295和6821/1698、seed74002均复用。未来只读checkpoint中的合法聚合统计和已有val；0训练图像、0旧图回放/生成特征、0optimizer/policy、test/reserved封存。

主四格固定A0=base、AL=local_transport、B=A0+MPC_DSM-v2、P=AL+MPC_DSM-v2。单模块MPC与DSM是预先指定贡献项，不是备选主方法；AIR仅复用公开各stage数值，不重评模型或收费。

## 七个性能头与一个CPU identity检查，逐项分离

|头|借用协方差|错误尾部|正则化|ETF线性方向|所回答的问题|
|---|---|---|---|---|---|
|MPC|开|关|关|关|二阶借用单独贡献|
|REG（CPU-only）|关|关|.1增量原曲率|关|单独开启严格identity，不新增医学臂|
|TAIL|关|开|关|关|尾部边界单独贡献|
|TAIL_REG|关|开|.1增量原曲率|关|正则化对尾部修正的作用|
|DSM|关|开|.1增量原曲率|.1尾部ETF力|方向净贡献|
|MPC_TAIL|开|开|关|关|二阶借用与尾部的组合|
|MPC_TAIL_REG|开|开|.1增量原曲率|关|组合中正则化贡献|
|MPC_DSM|开|开|.1增量原曲率|.1尾部ETF力|预先固定的主组合|

七个性能头使用同控制同stage的原W0/R/pairs、同一冻结尾部图和同一特征批；新列固定且主读出统一使用单向保护。自支持16/.4、无温度cov核、原boundary beta.5、原曲率reg.1、方向.1均前瞻固定。无网格、额外seed、类14/CSR等类特定强度、GT选择性回退或失败自动重跑。

至少完整报告：每头−各自控制；TAIL_REG−TAIL及MPC_TAIL_REG−MPC_TAIL；DSM−TAIL_REG及主组合−MPC_TAIL_REG；组合−MPC/DSM；DSM−TAIL、MPC_TAIL−MPC/TAIL；REG−控制为CPU恒等式，不重复医学评价；每头B/P交互；所有外部/控制/AIR差距与所有正零负。TAIL或其他单项若更好，不事后换主组合。

## 生产准入仍未完成

未来必须另行接入严格dataset/order/task/seed/control_source/source/版本/预算元数据、float32原头重建容差、单stage缓存、snapshot sidecar和collector。原worker默认不导入本设计helper；本设计没有launcher。

将来应4条CPU checkpoint准备完毕、全部18stage×7头=126个sidecar封存；4真实GPU零更新预检全PASS后统一val。预检只读首64真实val图，所有头CPU/GPU分数有限且一致、模型全部state不变，0optimizer，不输出分类指标；失败保留私有错误和全部驻留，停止依赖。

未来GPU仅0/1，启动前NAS实际nfs/free/小probe和显存准入、中性父子命令/NVML。建议沿用4h全程序上限，从实际启动起含准备/预检/评价，不在设计阶段虚造launch/deadline或费用。

## 评价、保护与风险诊断

四个控制/数据集轨迹共享encoder特征，评价18个checkpoint：固定seen val曝光15966，加4预检256，总16222；七头不重复算影像曝光。正常GPU过程4预检+4评价=8，CPU准备4，各独立出现计费一次；并行wall、CPU synthetic/pipeline、CPU实际准备、GPU逐次驻留及失败分列。126 sidecar增加CPU开销，不称计算免费。

每stage OFF的n/recall/Acc与已公开控制精确一致；MPC/DSM原型均值、组件中心/质量/半径、n不变；R/pairs原封；全部解finite、残差<=1e−8；stage1、少于2旧类或全部force零时identity，不制造作用证据。

主表两个控制及七头共32行（16方法格×2数据集），含BA/avgBA/tail/forget/old/new、Acc/avgAcc、可获得macroF1；AIR两行参考另列，不计为新头或重复收费，不补缺失。所有31类n/recall、各stage与频数组、原始和归一化margin正零负、旧旧/旧新/新旧/新新流、纠正/破坏/错转错完整。新类正确性/原始margin非下降必须实测验收，但这不保证新类precision/F1或old/tail。

固定raw与guarded诊断都公开，不按其优劣选择读出。报告raw损失被guard消除的计数、被消除的潜在旧类纠正、旧最大值下降带来的新类救回和旧类false positives、作用量保留比。不能把没有新类破坏直接称为模块成功。

尾部代理必须与各stage实际val混淆对应核对：所有旧-旧边的预测错误质量、经验计数、纠正/破坏去向；预先固定全边加权重合与Spearman（全零/同值则null）、top3仅作描述；不回馈graph。原始/借用后的trace和rank、无供体类、有效供体数、deltaQ范数、reg/ETF力、尾部风险/曲率、norm full/disc/common、raw/guarded delta全部保留。若tail力近零或预测边与实际混淆错位，按机制未激活/代理不可靠报告，不调大参数或选择类别修复。

相对各自控制开发门：BA>=+1pp、tail>=−.5pp、forget<=+1pp、old>=−1pp；当前新类recall不下降是机制验收而非额外科研功效。主组合须两个数据集都满足才可称该固定开发协议通过；否则完整负交付。AIR差距同时给old/new/tail，不把AIR当无损可达上界。单seed/order重复development val，不称统计显著、临床或独立患者证据。

## 交付与边界

未来获授权且真实结束后，公开源码/协议/全部正负/失败/结果/对照/费用到原DLwbm123/Long-tailed-CL feat/prototype-analytic（显式代理、remote SHA及精确报告匿名HTTP200），随后NAS独立匿名public_delivery/DELIVERY；不得发布影像、模型、样本/IDs/逐样本scores/features/银行/均值/协方差/尾部矩阵/private配置/rawlogs。本轮是public_preparation，不能创建假的实验RESULTS或COMPLETE_DELIVERED。
