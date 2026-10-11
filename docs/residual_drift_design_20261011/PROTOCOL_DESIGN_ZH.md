# 残余漂移约束：实现与贡献对照草案

状态：CPU_COMPATIBILITY_PASS / NO_NEW_EXPERIMENT_AUTHORIZED。
本次只实现默认关闭的模块、CPU 合成检查及协议设计。没有新正式训练、GPU 预检、分类评价、数据准备或监测。正在运行的 RFMiD negative_coverage_v3 的部署科学源、私有配置、数据、任务分配、预算、进程和自动化均未修改。

## 机制和准确目标

参考 LCA（Local Classifier Alignment, ICLR 2026，https://arxiv.org/abs/2603.09888）的特征邻域稳定性思想；这是解析改编，不是原版 LCA 损失或复现。没有模型合并、旧类高斯采样或可学习分类头。

对当前训练来源的固定顺序配对特征，沿原流程计算共同位移 Delta；定义 e_i=z_after_i-z_before_i-Delta，U=sum_i a_i e_i e_i^T，a_i>=0、sum a_i=1。U 不再按标签重中心、不做 trace 归一化、不累积为历史样本计数。它包含原共同位移未解释的均值偏差与离散变化；不是类内协方差，也不是旧类真实漂移的直接估计。

新增目标为 **(lambda/2)*tr(W^T U W)**。二分之一与原平方损失/竞争损失的约定一致，导出系统矩阵 A+lambda*U。原 ridge 强度 rho=.001 保留，以 R_eff=R+(lambda/rho)*U 接入现有求解，使 rho*R_eff=rho*R+lambda*U；不能将 lambda 误写成额外乘 .001 的强度。

CPU 检查及本草案采用固定 lambda=.1；这是待正式授权前冻结的设计值，不是通过验证集筛选的值。本轮默认及所有已有配置都是 lambda=0。非零参数需独立 residual_drift_authorized=true、residual_drift_version=residual_drift_v1，且只支持原 base 或 local_transport。没有新 ARMS，报告必须同时识别基础模块和 drift weight，不能将 base+drift 标成原基线。

## 单标签竞争头接入

作用路径：module_additions.head -> 原 reference head/competition mass -> static_pc 控制器的求解 -> proximal_base/Woodbury -> 每 batch competition.solve -> 同一 ridge 训练目标 -> task-end competition.bank_head。并未以普通 ridge 替换竞争头、删除 pair loss、proximal 或 FD10。

每个 epoch 开始，U 仅从当前 fit 的 teacher-start 与当前特征估计，采用原 sample_weights 的归一化质量并减原 class-balanced common_shift；两者的质量并不必然相同，故 residual mean 可非零，公开其范数。U 在该 epoch 内 detach/freeze；不通过 U 对 adapter 反传，不引入 minibatch 历史/未来特征。

task-end 原 meta 加入训练来源统计时，以全部当前 train 重新估计 U，沿原全部当前训练拟合重建竞争头。checkpoint 私有保存 U、weight、version 和原 pair weights；推理仍只用原 global head，增加推理计算/读出数为零。不允许把残余漂移静默 retrofit 到历史 checkpoint。

第一次 epoch 的 before==after，U=0；初始零更新预检可以确认 backward/预算，却不能证明非零约束已经激活。正式运行后必须另检查 boundary U trace、lambda、实际系统扰动、有限性和求解残差；激活不等于净效用。

## 多标签原 native A 兼容

保持逐 label/sign 的 n/s/G、signed binary ridge、bias penalty=1、available-sign equal mass、weighted masked BCE、task2 FD10、clip1 和全部原 optimizer 步。未知标签仍 unknown，不计 negative；inactive label head 仍为零。

U 是一个共享全局特征矩阵。其 row weights **精确复用原 translation 的当前 observed-pair 权重**，不读取未来标签或历史频率。各 binary head 的正负统计/质量不变，只在特征正则块加 lambda*U；bias 行列增加零，原 bias 惩罚不变。局部候选的每 label 正负四区域 P 及 R 原封保留。

native A 在任务内使用固定 start head：首任务无历史 U，start head 走原路径；后续 start head 使用上一 boundary 保存的 U。任务结束由本任务更新前后特征重估并替换 U，重建 final head。不是每 batch 更新，也不是累积各任务 U；每个 checkpoint 保存其对应 U，加载时拒绝 weight/version 不匹配。保持任务内 BCE/FD，而不是声称增加了直接 encoder 鲁棒损失。

共享 U 的选择是本次最小接入，**没有新增逐 label/sign 漂移模型或 Normal/其他疾病负例分层**。它不能修复未观察的共病监督，也不能证明旧类/所有疾病有同样漂移。将正负统计保留准确与漂移估计可靠分开报告。

## 贡献对照

采用四格、固定参数对照：

| 标识 | 原模块 | 漂移权重 | 目的 |
|---|---|---:|---|
| A0 | base | 0 | 同配置原竞争头或原 native A |
| AL | local_transport | 0 | 既有局部模块贡献 |
| AD | base | .1 | 残余漂移独立贡献 |
| ALD | local_transport | .1 | 组合效果 |

全部指标报告 AD-A0、ALD-AL、ALD-AD、ALD-A0，以及交互 (ALD-AL)-(AD-A0)。不得将组合效果全归漂移，不只报告有利类/阶段。forgetting 的正负方向单独解释。

**单标签方案（独立待批准）**：ISIC/HK 的 split/order/seed74002、原预训练、fit/meta、2 epochs/batch64/workers4/lr.0003/WD.01/clip5/FD10/ridge.001/proximal.01 不变。每臂 ISIC468/HK176 步。严格匹配且已交付的 A0/AL 可复用；主三项 A0/AD/ALD 为每数据集三项，AL 作为已匹配历史贡献控制。若匹配和 lambda=0 复现审计成立，则新增 2 datasets×AD/ALD=4 trajectories，1288 formal updates；否则不能悄悄重训扩大为8条，必须重新审查预算协议。采用原 BA>=+1pp、tail>=-.5、forget<=+1、old/new>=-1 各门，不以单个子组为成功。

**RFMiD 方案（另一独立待批准）**：只采用已批准 negative_coverage_v3，原29映射、五组、每task384图、Normal81/80/80/80/80、完整原共病/旧阳性和全部 mask 固定；不重求解或重新抽样。每臂12×5=60步。只有当前 RFMiD 先完整交付且控制匹配审计成立，才能复用 A0/AL，并考虑新增 AD/ALD 共120 updates。若控制不能复用，需另批四臂240 updates，不能冒充已授权。两条新臂的预检/训练/评价通常6 GPU 子进程，不收费已交付控制；失败逐项额外计。只GPU0/1、每个独立科学轮<=24h，test继续封存。

RFMiD 主 mAP 及完整29 label AP/P/N、AUROC、fixed .5 F1、平均增量mAP、AP遗忘、旧/新AP、固定head/middle/tail全部公开；AION/ODP/LS/CSR和推迟引入标签均完整报告，不按当前val换参数或组。不套单标签BA门，不声称达到自然医学双目标。

## 运行前仍需完成的工作

本次没有 launch.py、私有正式运行配置或新的 PROTOCOL_LOCK。现有两臂协调器/barrier/collector绑定base/local_transport目录，不能直接拿来运行四格实验。未来批准具体数据集/预算后，须独立冻结控制来源与贡献命名、匹配的全轨迹训练屏障、报告/成本收集和科学源；完成真实两臂 GPU 零更新预检后才能 TRAIN，全部新臂 TRAINED 后才统一 val。不能把本次 CPU TinyRegions 合成预检叫真实 ViT/GPU 预检。

仅保存新 U 聚合矩阵：float64、1536维约18MiB，每个checkpoint一次；不保存本任务配对特征或残差行。估计含一次 O(N*d^2) 聚合，正则准备另有 PSD 检查，实际 GPU/CPU 增量费用必须实测。当前样本漂移代表旧类变化只是代理假设；其有效性、医学判别信息损失和最终BA/mAP均未验证。

## CPU 检查与失败边界

CPU_CHECK.json 记录最终 PASS，AUDIT.json 保留工程尝试及 CPU 驻留；全部 CUDA_VISIBLE_DEVICES 为空、torch.cuda 未初始化、未读取医学影像/标签/val/test，正式 adapter/policy/GPU 更新为0。CPU五任务小模型有合成 optimizer 更新，用于轨迹兼容，不计正式科学更新。

测试包含原竞争 Hessian/目标梯度、proximal Woodbury 一致性、冻结 U 的 envelope gradient、binary Hessian/bias/unknown/inactive 支持、pure translation U=0、非法输入/PSD拒绝、旧版基线和旧局部候选 lambda=0 五任务轨迹精确复现、竞争头 checkpoint 重建和多标签 checkpoint 往返、原多标签权限/barrier/report 回归。lambda=0 的单标签头/正则和竞争数学路径精确；没有声称整条真实单标签 ViT 轨迹已复跑等价。

复现需已有 Python/PyTorch/NumPy。给 Q150_NATIVE 指向本次修改前提交的 tools/run_multilabel.py（见 AUDIT.json native_reference_commit），Q150_SOURCE 指向含 tools 的检查源码目录；中性 /tmp/q150.py 仅将该 tools 目录放入 sys.path 并执行 check_residual_drift.py。必须禁用 CUDA、限制 CPU 线程；不要通过运行正式 worker 来复现 CPU 检查。
