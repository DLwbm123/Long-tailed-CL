# RFMiD 正负差异局部先验：CPU 设计与贡献对照

状态：**设计及 CPU 检查，未授权启动、尚未接入正式 worker**。本次没有 GPU、实际影像训练或分类评价。正在进行的 ISIC/HK 残余漂移验证的部署源、数据、配置、预算及进程均未修改。

## 要解决的问题

原 local_transport 将某标签的正、负四区域单位中心分别除以 2 后拼成 8 列。其逆惩罚 `R=(I+PPᵀ)⁻¹` 会沿这些中心的共同方向降低正则。第三轮 CSR 对应区域正负中心 cosine 为 0.930–0.964，共同方向较强；原模块相对基线 AP 为 −1.421659pp。这支持检查共享方向的假设，不能证明 AP 下降的原因。

这里的“共同方向”仅由训练聚合中心定义，不是经医学标注证实的背景、正常组织或病灶定位。CSR 的负例也包含其他疾病；共同方向可能携带真正有用的鉴别信息，因此删除它有风险。

## 固定数学设计

对每个已引入标签、每个区域 r，用现有计数加权中心 `μ⁺ᵣ, μ⁻ᵣ`，先在原坐标取差，**不先单位归一化两个中心**：

`δᵣ = μ⁺ᵣ − μ⁻ᵣ`，`sᵣ = (μ⁺ᵣ + μ⁻ᵣ)/2`。

令 `S=[s₁,s₂,s₃,s₄]`，B 为 S 的完整数值列空间正交基。SVD 数值秩阈值只用 float64 epsilon × max(S.shape) × 最大奇异值，无按验证集选择的秩或阈值。

先构造差异控制的四列：

`pᵣ = δᵣ / (2 max(‖μ⁺ᵣ‖+‖μ⁻ᵣ‖, 1e−12))`。

完整候选再去掉共享投影：

`p̃ᵣ = (I−BBᵀ)pᵣ`，`P̃=[p̃₁,p̃₂,p̃₃,p̃₄]`。

两种方案分别用 P 或 P̃ 替换旧 8 列，沿用 `R=(I+PPᵀ)⁻¹` 与 `/2` 系数。保留差异大小，不将很小的差异或投影残量重新单位归一化。每列范数 ≤1/2，总能量 ≤1，所以 `0.5I ≤ R ≤ I`。完整候选对共享空间满足 `RB=B`：共同方向维持原基线惩罚，不重复获得正则放松。

累计任一符号没有支持时使用 I；相同中心、全零中心或完全被投影去掉的差异也退回 I。非有限值、非法计数或形状失败关闭。所有 29 标签使用同一公式，不为 CSR 单独调强度或增列。

**二次先验仍无法编码正负朝向。** 交换正负后 δ 变号，但 PPᵀ 不变；正负方向继续由 native A 的 signed RHS 决定。本改动利用差异的轴向几何，并非新增有符号读出或局部监督。

## 解析头与运输兼容

原 binary head 每标签使用可用正、负符号等质量的 global n/s/G；signed target 为 −1/+1，含 regularized bias。新先验只改变 feature penalty，`diag(R,1)` 的 bias 仍为 1，原 global gram 与 RHS、ridge=.001 全部保留。Woodbury 四列求逆和直接求逆一致；正定界保证系统可解。

局部存储仍为 `[label,sign,region,feature]` 的 n/mean，无新样本或特征记忆。保留原 observed-pair 加权 global common_shift、每区域共同运输与随后按计数加入当前中心；不另做正、负不同的运输。共同平移在原始 δ 中抵消，但分母和共享空间可能改变，**整个新先验并非平移不变**。它也没有解决旧标签真实非线性漂移的估计偏差。

这是逐标签 binary 头的兼容设计。单标签竞争头没有对应的逐标签负中心，不能直接宣称兼容或沿用该公式；本次不修改它，不组合残余漂移。

实现仅新增 `tools/rfmid_contrast.py` 的 CPU callable。原 `rfmid_local.penalty`、Trainer、RF worker 的模块白名单和权限检查均未修改。CPU 集成测试只在独立 CPU 进程临时替换 callable，并在 finally 中还原；未提供可启动的正式配置。

## CPU 检查与已完成 CSR 聚合核对

`CPU_CHECK.json` 核对：Woodbury/直接逆、特征惩罚特征值界、共享空间不变、sign swap/正交坐标等变、缺符号/零差异/极小差异回退、非法输入拒绝、原 n/s/G 不变、native identity head 精确一致、共同运输保持计数与原始正负差。两种先验通过两任务 synthetic adapter、unknown mask/counts、checkpoint roundtrip 和各一次 synthetic 零 optimizer backward；16 个 tiny-fixture optimizer updates 仅属 CPU 测试，正式科学更新为 0，真实 GPU 预检为 0。

`CSR_GEOMETRY_CHECK.json` 只读第三轮已完成 candidate final checkpoint 的聚合统计与存储头；不读取影像/逐样本分数或特征，不重新评价 val/test，不写 checkpoint。原 CSR 头独立重建到 float32 后与存储值精确一致。新头使用**同一候选 global statistics**，不能冒充原基线或新训练效果。

CSR 的未投影差异能量 0.0296883，去共享后 0.0196594，保留 66.22%；原始差异仍含共享分量。完整候选共享泄漏约 3.4e−18，最小惩罚特征值 0.989730，旧先验为 0.450534。其余三重点差异能量保留 AION 81.89%、ODP 93.75%、LS 82.38%，全部正负公开。

在相同候选 global statistics 下，CSR 原先验头相对 identity 头改变 4.52129%；差异控制改变 0.041032%，完整候选改变 0.022139%，新头残差 ≤7.61e−15。因此它主要尝试减少旧先验对共享方向的过强影响，可能只趋近无局部先验的头，也可能删掉有用鉴别信息。**没有新 AP，不能声称已改善 CSR 或能超过原基线。** 单个头重建也不包含该先验参与 BCE 后产生的表征变化。

本机 bundled Python 首次 import torch 失败，检查未开始；已保存私有回执，随后使用既有远端 CPU 解释器、单线程、低优先级完成。没有安装/改环境、GPU 或实验重跑。CPU 检查和聚合核对驻留分别单列，不加入 GPU 账本；初次失败未测 CPU 时间不填 0 冒充实测。

## 未来贡献对照，当前不执行

| 格 | 先验 | 执行方式 | 新正式更新（若获授权） |
|---|---|---|---:|
| A0 | native A，I | 严格复用 coverage-v3 base | 0 |
| AL | 原正负八列 + 原运输 | 严格复用 coverage-v3 local_transport | 0 |
| AC | 正负差异四列 + 原运输 | 新配对训练，当前未授权 | 60 |
| ACS | 差异四列去共享 + 原运输 | 新配对训练，当前未授权 | 60 |

报告 AC−A0、ACS−A0、AC−AL、ACS−AL、ACS−AC 的所有正负。AC 对 AL 同时改变方向、列数和总能量，不能称纯正负差贡献；ACS 对 AC 同时移除共享分量并减少能量，不能称等强度对照。比较是完整固定设计的效用验证；不通过验证集强行匹配能量或追加强度网格。

复用第三轮已批准的具体 v3 TASK_PLAN，禁止重新求解/选 seed/改变组或 mask。原 train1920/val640、29多 hot、每任务384图、5任务9+5+5+5+5、每图一次、Normal81/80/80/80/80，2470次阳性/650共病对/34010已知负例保持；19200未来负例仍 unknown。CSR train37P/1883N、val11P/629N。任务数据与划分不是本轮的新增因素。

ViT 双 adapter/global1536、BCE、FD10、AdamW .0003/WD.01/clip1、每task 2epoch cosine eta_min1e−5、batch64/workers4/ridge.001/seed74002、无meta/controller均不变。AC/ACS 各每任务12更新、总60；未来新增120更新、0policy，2零更新预检+2train+2eval共6个正常 GPU 费用记录，复用控制不重新收费。

未来执行还需明确人工授权、版本化 worker/权限及 checkpoint prior 标识、严格控制匹配复核、源冻结代理公开、两臂**真实 GPU** firstbatch backward/0optimizer 预检 PASS；这两臂60更新、5checkpoint 均 TRAINED 后才统一 val。等待受保护的残余漂移验证结束/active空与真实 NAS/free/probe、GPU0/1≥32768MiB准入；不借GPU2/3，不干预其他任务，24h包括等待不重置，不自动失败重跑。CPU callable 检查不代替这些步骤。

## 报告与结论边界

主指标全29标签最终 mAP；同时完整5stage/29label AP、AUROC、固定.5 macro/micro F1、P/N、训练固定频数组、平均增量mAP、forget/old/new和全部增量。CSR 为机制重点，AION/ODP/LS与其余25标签的损失都必须报告，不只选正标签。记录每stage shared rank、raw/retained energy、retained fraction、shared leakage、prior delta、finite/solve≤1e−8。

不套单标签 BA 门，不按 val 改强度/阈值/seed/阶段，不将激活或头参数变化当效用。test一直封存；只用单seed/order反复 official val，CSR只有11阳性、无patient/eyeID及图像内容重叠核对，不称独立临床泛化或 APPLE/MONICA 精确复现。未来实际实验如获授权，仍须完整 GitHub+NAS 正负/失败/费用交付；**本次仅交付设计源码、CPU 检查和聚合数学诊断，无新实验结果。**
