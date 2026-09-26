# NB2-MedVLM-12H：医学 VLM 领域匹配与语义利用验证

## 1. 核心研究问题

本轮严格保持既有持续学习设定：

> 通用预训练初始化 → 正常两类 Task1 → 后续类增量；不设置额外医学类别预训练阶段。

需要回答四个问题：

1. 通用 OpenCLIP 表现有限，是否主要因为 VLM 与医学域不匹配？
2. 医学 VLM 的优势来自视觉表征，还是图文语义对齐？
3. 医学 VLM 与 APART 结合后，是否提供稳定的互补信息？
4. 如果医学 VLM 有用，但 RASP 仍无收益，问题是否主要出在语义先验的门控/注入方式？

本轮不训练 ACTM，不搜索 FD β，不搜索 prompt，不进行 LoRA/微调。

---

# 2. 固定实验协议

| 项目 | 固定设置 |
|---|---|
| ISIC | 2+2+2+2，完整 8 类 |
| HyperKvasir | 2×10+3，完整 23 类 |
| seed/order | 1993 / 1994 / 1995 |
| APART 起点 | 已存在的正常两类 Task1 末状态 |
| 额外 Task1 训练 | 0 |
| 新增神经训练 | 0 epoch |
| optimizer steps | 0 |
| test/reserved | 0 访问 |
| 验证 | 原 val；全部方法锁定后统一评价 |
| 主指标 | Final BA |
| 辅助指标 | Macro-F1、AvgBA_inc、old/current recall、HM、tail recall、zero-recall、forgetting |

所有新方法共享同一 APART Task1 父状态和同一类别顺序。

---

# 3. VLM 矩阵

## G：Generic VLM

沿用当前已经完成的：

- OpenCLIP ViT-B/16
- 当前锁定权重、tokenizer、preprocessing
- 作为 Generic VLM baseline

原 NB2-VLM-R1 的结果保持不变，不重写。

## B：BiomedCLIP

主 medical VLM：

- microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- 使用其自己的视觉 encoder
- 使用其自己的 PubMedBERT text encoder
- 使用其自己的 tokenizer
- 使用官方 preprocess
- 所有文件下载后 SHA256 锁定
- 正式运行禁止联网 fallback

ISIC 和 HK 都完整运行。

## D：DermLIP

仅用于 ISIC 的探索性专科对照：

- DermLIP ViT-B/16
- 使用其自己的 tokenizer、text encoder、preprocess
- 不替换任何组件为 Generic CLIP 版本

但运行前必须完成：

- 当前模型卡许可证记录；
- 权重许可证文件记录；
- 训练数据来源审计；
- 与已有 ISIC train/val 的可检查重叠审计。

DermLIP 不作为主方法，不决定本轮 PASS/FAIL。

---

# 4. 第一阶段：R1 旧结果机制审计

预计 30–60 分钟，尽量零新增图像 forward。

对 Generic CLIP 原 R1 的 45 个阶段检查：

- semantic scale；
- component count；
- reliability；
- gamma；
- A2 与 A6 的 W 差异；
- A2/A6 预测是否逐样本一致；
- 哪些类别 gamma=0；
- gamma=0 的具体原因；
- gamma 非零时先验实际引起的权重变化大小。

必须区分：

A. 先验被 gate 完全关闭；

B. gamma 非零，但 W 变化数值极小；

C. W 改变，但 argmax 不变；

D. prediction 改变，但 correction 与 destruction 抵消。

输出：

`GENERIC_PRIOR_AUDIT.csv/json`

不得因为发现 gamma 小就直接调 gamma。

---

# 5. 第二阶段：每个 VLM 的核心读出

设 APART 特征为 a，VLM 图像特征为 u。

全部特征先独立 L2 normalize。

联合表示：

h = [a ; u] / sqrt(2)

每个 VLM 固定运行以下方法。

| 方法 | 定义 |
|---|---|
| V | VLM 单视觉 + class-balanced ridge |
| J | APART + VLM joint ridge，λ=5e-4 |
| J1 | APART + VLM joint ridge，λ=1e-3 |
| Z | VLM zero-shot，只在当前已见类别中分类 |
| L | J + fixed raw-text top-k score fusion |
| R | J + 当前固定 RASP spectral/text prior |

同时复用：

| 方法 | 定义 |
|---|---|
| P | 原始通用预训练 AugReg + ridge |
| F | Task1 后冻结 APART + ridge，λ=1e-3 |
| F2 | Task1 后冻结 APART + ridge，λ=2e-3 |

J 与 F 是匹配原始分支正则尺度的主比较。

J1/F2 是第二个固定尺度对照。

不是 λ search，不允许增加其他 λ。

---

# 6. 主比较

## Primary-1：医学 VLM 是否优于通用 VLM？

最重要：

B.J − G.J

分别在 ISIC 和 HK 完整序列报告。

这是本轮真正的 medical-VLM 检验。

## Primary-2：医学 VLM 是否值得加入 APART？

B.J − F

回答医学辅助分支相比 Task1 后冻结 APART 是否提供真实增益。

## Mechanism comparisons

B.V − G.V
→ 医学视觉 encoder 本身是否更适合。

B.Z − G.Z
→ 医学图文对齐是否更好。

B.L − B.J
→ 简单文本信息是否有额外作用。

B.R − B.J
→ 当前谱先验是否有额外作用。

J − F / J1 − F2
→ 排除联合表示缩放导致的 ridge regularization 混淆。

D.J − G.J / D.J − B.J
→ ISIC 专科模型探索，不进入主效用 gate。

---

# 7. 如果时间允许的固定消融

只有核心 G/B 全矩阵完成以后才执行。

顺序固定：

1. R-no-gate：取消 reliability，但保持 rarity 和 spectral prior。
2. R-identity：不用 spectral direction weighting。
3. R-zero-prior：保持相同 regularization strength，但 prior center=0。
4. L-permute：固定错位文本标签，作为 semantic negative control。
5. template-0 / template-1 zero-shot：只用于模板敏感性分析。

不搜索参数。

任何一个扩展方法未完成，都记录 NOT_RUN，不补跑到超时。

---

# 8. 医学 VLM 数据暴露审计

BiomedCLIP：

记录：

- 模型 ID；
- revision/commit；
- 权重 SHA256；
- preprocess；
- tokenizer；
- text config；
- PMC-15M 训练来源；
- 能否确认 ISIC/HK 直接样本暴露。

若无法确认：

`PRETRAIN_EXPOSURE = UNKNOWN`

不能写成 CLEAN。

DermLIP：

必须特别记录 ISIC 来源重叠风险。

只允许检查：

- 已有公开模型数据说明；
- 当前项目 train/val 样本已有 ID/hash；
- 合法公开索引。

不得打开 sealed test 来做 exposure 检查。

---

# 9. 12 小时墙钟计划

总墙钟硬上限：12:00。

| 时间 | 工作 |
|---|---|
| 0:00–0:45 | 路径、空间、GPU、git、父状态、backup gate |
| 0:45–1:30 | 下载/锁定 BiomedCLIP，准备 native factory |
| 1:30–2:00 | Generic prior audit + synthetic/unit tests |
| 2:00–2:30 | 小规模 Task1 qualification、吞吐测量 |
| 2:30–6:30 | BiomedCLIP × ISIC/HK × 3 seeds 全部 train-stat extraction / analytic fits |
| 6:30–7:30 | DermLIP ISIC core（只有许可/暴露/资源 gate 通过才运行） |
| 7:30–8:30 | 固定扩展消融 |
| 8:30–10:15 | 所有锁定方法统一 val |
| 10:15–10:45 | bootstrap、逐类、forgetting、error decomposition |
| 10:45–12:00 | 中文报告、审计、备份、Git 公开交付 |

硬停止：

- 8:30 后不得新增 fit variant；
- 10:15 后不得启动新的图像 forward；
- 10:45 后只允许 CPU report / hash / archive；
- 到 12:00 必须停止。

如果核心 BiomedCLIP 比预计慢：

优先级：

G/B 主矩阵
> G/B L/R
> DermLIP core
> DermLIP extensions

不得牺牲主比较和最终报告。

---

# 10. 资源与备份

当前最大的操作风险是：

大型私有资产只有 remote-home，独立第二存储尚未验证。

本轮开始前：

- 检查至少一个独立故障域 backup destination；
- Task1 六父状态；
- shared/core dependency；
- Generic VLM weights；
- Medical VLM weights；
- task/manifest locks；
- 新生成 bank/W；
- protocol/code locks；

必须生成 SHA256 inventory。

如果无法验证独立备份：

- 不删除任何 remote-home 历史资产；
- 每个阶段完成立即保存；
- 聚合结果及时 commit；
- 报告明确写 `SECOND_BACKUP_UNVERIFIED`。

不为了腾空间删除旧实验。

---

# 11. 预定效用门槛

## BiomedCLIP 主候选 B.J vs Generic G.J

一个数据集判为 positive signal，需要同时：

Final BA mean gain >= +1.0 pp

至少 2/3 seed/order Final BA 提高

AvgBA_inc 不下降

Final tail recall 平均不下降

Final old recall 平均下降 <=2 pp

Final current recall 平均下降 <=2 pp

任一 seed old/current recall 不下降 >5 pp

不得产生新的 zero-recall class

这是 development utility gate，不是统计显著性或临床门槛。

## B.J vs F

同样完整报告，但不要求两个比较都 PASS 才能分析机制。

最重要的是不要把：

B.J > G.J

自动解释成：

“持续学习方法成功”。

它首先只能说明 medical VLM 辅助分支优于 generic VLM。

---

# 12. 预先规定的结果解释

## 情况 A

B.V > G.V
B.J > G.J

支持：

医学 VLM 的视觉表示更适合当前医学域，并带来可利用的互补信息。

## 情况 B

B.Z >> G.Z
但 B.J ≈ G.J

支持：

医学图文语义更强，但当前 joint ridge 未利用这种优势。

下一阶段优先研究语义注入，而不是 adapter training。

## 情况 C

B.J > G.J
但 B.R ≈ B.J

支持：

medical representation 有价值，但当前 RASP prior 没有额外价值。

## 情况 D

B.R > B.J

说明 medical text prior 开始产生可测额外收益。

再检查：

是否稳定；
是否来自少数类别；
是否损害 tail/current；
gate 与无 gate 对照。

## 情况 E

B.V、B.Z、B.J 全部没有改善

削弱“generic CLIP domain mismatch 是主要瓶颈”的假设。

此时再继续换更多 medical VLM 的优先级应下降。

---

# 13. 本轮禁止事项

不得：

- 改两类 Task1 设定；
- 增加医学类别预训练；
- 重新训练 Task1；
- 搜索 lambda；
- 搜索 gamma；
- 搜索 prompt；
- 根据 val 选最好模板；
- 调 FD beta；
- 启动 ACTM；
- LoRA/fine-tune VLM；
- 使用 test；
- 因一个 seed 表现差取消其余 seed；
- 将 DermLIP 的探索性结果替换预注册 BiomedCLIP 主比较；
- 将最大 BA 的方法事后指定为主方法。

---

# 14. 最终交付

公开：

- FINAL_REPORT_ZH.md
- FINAL_REPORT.json
- METHOD_MATRIX.csv
- stage_metrics.csv
- class_metrics.csv
- summary_by_seed.csv
- bootstrap_intervals.csv
- forgetting.csv
- error_decomposition.csv
- medical_vlm_audit.json
- exposure_audit.md
- resource_report.json
- access_ledger.json
- backup_report.json
- protocol_lock.json
- source_lock.json
- completion_receipts.json

私有：

- per-sample logits
- image IDs
- image paths
- checkpoints
- raw feature caches
- private identity/component maps

实验完成后：

`NEXT_DECISION=STOP`

不自动启动 ACTM。
