# CT1：无独立基础阶段的持续适配实验

**状态：执行前修订协议；尚未在用户服务器运行。**
本文件覆盖 A1/HK1 中“较大的 S0、只训练 S0、之后冻结适配器”的要求。历史实验、结果与 checkpoint 保留，不改名、不覆盖。CT1 是新协议，不是给 HK1 的 S0 改一个编号。

## 0. 执行范围与版本身份

用户要求：直接从公共预训练视觉模型进入第一个任务，在每个任务持续训练，并比较新提出的分布风险方案、S-J-CB 路线、ConCM-lite 路线。

本轮将“持续训练”明确为：**每个任务均更新 ViT 内的适配器、路由/频次相关可训练组件和训练分类头；公共预训练 ViT 核心保持冻结。**这是 continual parameter-efficient tuning，不是仅更新分类器，也不是全量微调 ViT。所有任务执行同样训练日程，不在 Task 1 后冻结适配器。

原 S-J-CB 的精确方法定义包含 S0 后固定表征，不能在改变后继续声称原样复现。本轮名称固定如下：

| ID | 本轮方法 | 含义 |
|---|---|---|
| CT-J-CB | 持续双分支适配＋漂移补偿＋CBRidge | S-J-CB 的无基础阶段持续适配变体 |
| CT-Risk | CT-J-CB 的同一神经轨迹＋分布风险读出 | 上一轮提案的可运行受限版本：对角风险统计、分类器子空间中的凸约束 |
| CT-ConCM | 持续双分支适配＋ConCM-lite 高斯特征回放 | 继承 K 的经验方差/回放系数规则，但适配器每任务更新，并加入明示的统计漂移补偿 |

CT-Risk 本轮的风险项只优化任务末分类器，不通过求解器反传适配器。其适配器在每任务都有真实梯度更新，但与 CT-J-CB 共享同一训练轨迹。这是有意隔离读出干预，不是三套独立端到端训练，更不是完整分布鲁棒算法已得到验证。

**工程说明不等于新颖性声明。**漂移补偿、对角收缩、类别加权均有相关先例。以下参数是一次固定可证伪的实验设定，未经性能搜索，不称最佳值或临床置信参数。

## 1. 数据与任务：没有大基础任务

保留已经审计的图像划分、身份关联和规范类别编号，仅重新分任务。

### 1.1 数据集

- ISIC：V2 的 train=18,718、val=295，既有 test=764 不释放。8 类，实际训练不平衡度沿原锁，不再称实测 100:1。
- HyperKvasir：HK1-M1 的 fit=6,821、val=1,698；保留历史外层 reserved，不释放。规范 ID 必须是 M1 的 `official_name_sorted_id`，不能退回 legacy ID。
- 不重建 split、不补回排除样本、不做新的长尾下采样、不改变病灶/内容关联。
- 重用既有输入和权重，不重新下载图像或公共预训练权重。
- 新 test 图像读取、test 模型前向、test 特征读取、test 预测均为 0。重用旧 test 特征乘新 W 也是新的 test 预测，禁止。
- P0 只读既有身份/成员锁及 fit/val 资产，不重新打开 reserved 图像做审计。
- ISIC/HK 数据已参与开发，任何新 validation 结果仍属开发证据。

### 1.2 任务大小

- ISIC：`[2,2,2,2]`，累计 `[2,4,6,8]`，Task 1–4。
- HyperKvasir：`[2,2,2,2,2,2,2,2,2,2,3]`，累计 `[2,4,6,8,10,12,14,16,18,20,23]`，Task 1–11。
- 23 不能整除 2，余下 1 类合入最后任务，避免最后出现单类任务。不丢类、不为数据较少任务更改分组。
- Task 1 没有额外预训练、更多 epoch 或预先校准；没有 hidden S0。Task 1 只是旧类记忆尚为空的普通任务。
- 这仍是有任务边界、每任务多 epoch 的 CIL，不称 single-pass online/task-free learning。

### 1.3 固定排列

ISIC：
```
1993: [4,0,3,7,5,6,2,1]
1994: [3,7,4,5,1,0,6,2]
1995: [1,3,0,6,4,5,2,7]
```

HyperKvasir（M1 规范 ID）：
```
1993: [0,4,12,8,2,11,17,3,21,16,22,13,18,14,9,19,15,6,5,20,10,7,1]
1994: [22,16,12,13,21,4,19,17,6,11,3,15,0,9,5,7,18,1,10,14,20,8,2]
1995: [9,13,16,14,15,5,12,1,8,22,19,17,18,6,20,0,3,11,2,21,4,10,7]
```

每个排列按上述任务大小连续切片。order_seed=train_seed=1993/1994/1995，为三个配对的顺序/训练组合；并未隔离两种随机性，不称三个独立患者划分。

频率组：ISIC 沿 G1/A1 固定 head2/mid4/tail2；HK 沿 HK1 固定 head8/mid7/tail8。按原始类别 ID 对齐，阶段内只取已见交集，空组记 NA，不重新排名。完整 fit 类数仅用于数据协议与预先声明的输出容量；训练样本权重、先验、统计只使用已到达类。

## 2. 全部方法与工作量

| 表格身份 | 方法 | 神经训练流 | 主预测器 |
|---|---|---|---|
| 主方法 | CT-J-CB | U：当前真实图像的连续适配 | 漂移补偿统计上的 CBRidge |
| 主方法 | CT-Risk | 完全复用 U | 受限分布风险分类器 |
| 主方法 | CT-ConCM | R：U 的真实损失＋旧类高斯回放 CE | raw main+few heads |
| 强参照 | PT-CB | 无神经训练 | 原始预训练特征上的 CBRidge |
| 漂移诊断 | CT-J-Stale | 复用 U | 故意不补偿旧统计的 CBRidge |
| 读出诊断 | CT-ConCM-CB | 复用 R | R 的同一联合特征＋相同漂移补偿＋CBRidge |

主比较：`CT-Risk − CT-J-CB`。固定次比较：`CT-ConCM − CT-J-CB`（系统比较）、`CT-ConCM-CB − CT-J-CB`（同读出比较）、`CT-J-CB − CT-J-Stale`、三主方法对 PT-CB。不能拿 stale 这个有意失配的控制作为唯一主要对手。

- 两数据集 × 3 repeats × 2 神经流 = **12 条完整连续适配轨迹**。
- U 被 CT-J-CB/CT-Risk/Stale 共享，不重复训练后伪称独立。
- `(4+11)×3×2 = 90` 个实际任务训练单元，各 10 epoch，共 **900 个 task-epoch**；不是 900 次遍历整个联合数据集。
- 本轮 U/R 的 Task 1 分别真实运行并验证一致，不跨流复用 Task 1 节省预算；因此正式任务末 checkpoint=90。
- 实际 optimizer step 在任务计数锁后按 `10×ceil(N_task/48)` 计算并累加；不能预填未知数。
- 三主方法：**135 行阶段主指标、1,377 行逐类指标**。
- 加三个固定参照/诊断：**270 行阶段指标、2,754 行逐类指标**。
- 独立重复单位是固定 run，不是每个 task、共享 readout 或每张图像。

## 3. 所有神经流的公共训练配方

### 3.1 初始化与可训练参数

- 只载入原始 `timm/vit_base_patch16_224.augreg_in21k_ft_in1k`。
- 固定 revision：`2ec9fb3d7bb664aac471ac44582c94d18de33780`。
- 固定 SHA256：`c401d219603ac3e20b6373c7b198c78d3a733f80b755d148bda3bc320ae69800`。
- 不载入 ISIC/HK 已训练 S0、K、Full Dynamic 或其他下游父状态。
- `vit_b16_224_adapter_pool`；原两组 adapter pools、bottleneck=64、pool_size=5、top1 和频次 embedding=12726×16，保留来源架构。
- ViT core/original backbone 冻结；原允许的 adapter/pool/keys/assigner/heads 训练。
- 训练前逐名锁定白名单；每任务记录至少一个适配器张量发生非零更新，而非仅 head 更新。未被路由选中的单个模块不强求每任务都有梯度。
- Task t 从 Task t−1 的自己的最新权重继续，不能每次重载公共预训练、复制旧 S0 或额外扩张 task-specific pools。
- 所有任务重用同一组 pools，不调用隐藏的任务槽位复制/额外 prompt 预热。原 shared_prompt_pool/shared_prompt_key 均关闭；必要时用明确无操作实现替代通用 `_init_prompt`。
- 训练类别 remap 到已声明 order 的位置。可预分配 8/23 行训练 head，但 CE/指标仅访问已见行；记录未来行无数据梯度，AdamW 衰减不误报为数据泄漏。

### 3.2 日程

```
epochs_per_task: 10
batch_size: 48
drop_last: false
optimizer: AdamW
pool_lr: 0.0003
other_trainable_lr: 0.003
weight_decay: 0.01
scheduler_per_task: cosine
T_max: 10
eta_min: 0.00001
reset_optimizer_at_task_boundary: true
reset_model_at_task_boundary: false
AMP: false
TF32: false
head_norm: false
```

这是新设定：**所有任务均用同一 cosine 日程**，不继承旧协议“只有 S0 cosine”的特殊处理。每任务重新初始化 optimizer，但权重连续继承。自然分布 shuffle，不追加 balanced sampler 或 class-balanced CE。

训练图像增强沿来源：RGB、Resize256×256 bicubic/antialias、RandomCrop224、horizontal flip 0.5、ToTensor、mean/std=0.5；确定性提取 RGB Resize224×224 bicubic/antialias、mean/std=0.5。

### 3.3 真实图像损失

继承 APART 的三项 main/sum/pool-weighted-few CE、原 sum/mean reduction、`/3`、pool assignment、pull coefficient=0.1、theta=100。

**唯一统一的候选集规则：三项真实 CE 均为 all-seen。**
```
ce_start = 0
logits = logits[:, :C_seen]
target = global_remapped_target
```
不能再次使用 current-only CE，却让回放 CE 覆盖所有已见类。assignment 的 epoch<5 日程在每个普通任务执行相同的本地 0–9 epoch 规则；这是所有任务一致的训练规则，不是额外 S0。

训练可用当前真实标签查询当前类的 fit_count；推理、统计映射及路由不接收真实类别频次或 oracle task ID。关闭 DSM、MPC、USFM、uncertainty replay controller、head norm、GSR 等未列模块。源码局部变量 `match_loss`（APART pool assignment）不等于 ConCM DSM。

## 4. 核心修订：连续特征变化时的历史统计版本管理

### 4.1 禁止操作

不允许把不同 encoder 版本提取的旧/新特征统计直接相加，并声称这是同一固定特征空间的精确解析更新。不允许重新遍历旧训练图像刷新特征；文件仍在磁盘并不构成合法回放授权。也不允许把旧 S0 全数据缓存当作当前 encoder 的旧类缓存。

在固定表示中，统计累计可与批量解等价；现在只能声称**与显式定义的、经近似输运的虚拟统计目标等价**，不能声称等于当前网络重新编码全部历史图像后的真批量解。

### 4.2 共同的对角仿射输运（工程基线，不作原创声明）

对当前 Task t 的同一批 **train** 图像，在训练前的模型版本和某个 epoch 后的版本，确定性提取成对特征 `x_i^-`、`x_i^+`。每个当前类总权重相同：

`omega_i = 1 / (C_current * n_current[y_i])`。

对每个坐标 j：
```
mx = sum omega_i*x_i^-
my = sum omega_i*x_i^+
vx_j = sum omega_i*(x_ij^- - mx_j)^2
cxy_j = sum omega_i*(x_ij^- - mx_j)*(x_ij^+ - my_j)
tau_map = 1e-3 * max(mean(vx), 1e-12)
a_j = clip((cxy_j + tau_map)/(vx_j + tau_map), 0.5, 2.0)
b_j = my_j - a_j*mx_j
r_j = sum omega_i*(x_ij^+ - a_j*x_ij^- - b_j)^2
```

记 D=diag(a)。这是向恒等映射收缩的固定近似；无法表达旋转或类别特定非线性漂移。所有 clipping 数量、残差、当前类支持数均报告。不根据 val 误差选择补偿或不补偿，也不把 r 称为经校准的置信方差。

使用同一个公式分别拟合：
1. normalized joint J 坐标的映射（CT-J/CT-Risk/CT-ConCM-CB）；
2. raw main/few 各自坐标的映射（CT-ConCM 采样）。

**先在每张真实图像上构造当前 J，再拟合 J 空间的映射。**不能把 raw moments 先仿射后直接声称其归一化矩也精确。输运后的旧 J 统计是单位特征分布的近似，可偏离精确球面统计；本轮不事后只归一化均值而不更新二阶矩。必须报告旧均值范数、trace 和失配诊断。

### 4.3 无需逐类存完整协方差的精确仿射统计更新

在某个已声明模型空间，维护：
```
S = sum_c E_c[z z^T]    # 类别均值的和，不是所有图像的数量加权和
mu[:,c] = E_c[z]
v[:,c] = diag(Cov_c[z]) # population denominator n
n[c]
e[:,c]                  # accumulated diagonal drift-error proxy
class_arrival_task[c]
space_version
```

设旧类数 m，u=sum_c mu_c。对所有旧类使用同一 D,b 时：
```
S_new = D S D + (D u)b^T + b(D u)^T + m b b^T
mu_c_new = D mu_c + b
v_c_new = a^2 * v_c
e_c_new = a^2 * e_c + r
n_c_new = n_c
```

然后追加当前类在**当前任务结束后的 encoder**下真实确定性特征的准确矩；新类 `e_c=0`。由 `S` 保留联合特征完整 cross-Gram，风险部分仅用各类对角 v。

该 S 更新对所选仿射模型代数精确；仿射模型对真实旧类漂移是否正确则未知，两者要分开报告。误差代理 e 不是独立患者方差或风险保证。

### 4.4 epoch 与 task 的版本关系

- Task t 开始时，仅为当前训练图像缓存 pre-task 参考特征；不需要持续保存全部旧 encoder 副本。
- 每个 epoch 后，在隔离 eval/probe 中重新提取当前训练图像，得到从 **pre-task anchor 到当前 epoch** 的映射。
- U/R 两神经流都执行相同日程的当前图像探针，以保持访问口径可比；无 validation 参数反馈。
- R 的第1个 epoch 使用上任务结束的旧类 memory（identity map）；第 e>1 个 epoch 使用从 pre-task anchor 输运到 epoch e−1 结束模型的 memory，整个 epoch 内固定。
- 这是**epoch 边界校正**，不是每个 minibatch 精确对齐。记录此近似。
- 不能把第 e 个 anchor→epoch 映射再作用于已经按 anchor→epoch−1 输运的 memory，否则重复补偿。
- task 结束只把最终 anchor→epoch10 映射正式提交一次，e 也只累计一次；再加入当前类真实统计。
- Task 1 没有旧类输运，只有普通训练和任务末建库。
- 所有 probe 恢复 train/eval 模式、RNG；使用 A1 验证的 pointwise inference 副本，不改变真实训练算子调度。

## 5. 各方法的精确定义

### 5.1 CT-J-CB：持续适配变体

U 神经流每任务仅用第3节真实图像目标更新。任务末完成第4节输运，设 C=C_seen：
```
G = S/C
R = [mu_1,...,mu_C]/C
W0 = solve(G + 0.001*I, R)
score(x) = J_theta_t(x)^T W0
```

无 bias，统计/solve float64。J 为 raw main/few 拼接后的整体 L2，1536维；不能改成平均logits、分别归一化或原始A+adapted拼接。所有已见类参与预测。

### 5.2 CT-Risk：分布风险提案的本轮受限实现

复用 CT-J 的 U 模型、访问流、输运后的 S/mu/v/e/n。**不另训一套适配器，也不把风险项回传 U。**

本轮为控制计算和明确定义，使用：
- 校准仅进入风险统计，**不修改 CBRidge 的经验拟合项 G/R**；
- 类特定风险协方差用对角近似；
- 分类器限制在 baseline W0 的列空间，W=W0 B，B为C×C。

这三项都是本轮的可运行限定，不声称等价于上一条回复中“全空间、完整校准协方差”的提案。

固定风险统计：
```
v_pool = mean_c(v_c)
a_c = 10/(n_c+10)
v_tilde_c = (1-a_c)*v_c + a_c*v_pool + 1e-8/1536
u_c = v_tilde_c/max(n_c,1) + e_c
```
不修改 mu_c，不把尾类总损失权重调小。v_tilde 表示类内分散的收缩代理，u 表示均值与漂移不确定性的代理；n是图像数，不伪称有效独立病例数。先验只来自当期已经到达类，不需要基础类知识库。

设 `A=G+lambda*I`，`H=W0^T A W0`。求解：

min over B, xi>=0:
```
tr((B-I)^T H (B-I)) + (eta/C)*sum_c xi_c + 1e-10*||B-I||_F^2
```

对全部有向类别对 c!=k，令 `q=W0*(B[:,c]-B[:,k])`：
```
mu_c^T q - rho*sqrt(q^T diag(u_c) q)
  >= delta + kappa*sqrt(q^T diag(v_tilde_c) q) - xi_c
```

固定 `eta=0.1, rho=1, kappa=1, delta=0.1`。每类一个 slack，不为旧类、新类另选系数。该目标是软约束的、限定分类器子空间的凸问题；不能声称保证所有旧尾类不退化。

实现时预计算 C×C 的 `W0^T diag(v_tilde_c) W0` 与 u 对应矩阵，再分解构造范数，避免创建1536维×全部类别对的稠密锥系统。这样23类时B仅529变量，而非在全部1536×23空间求解。

`tr((B-I)^T H (B-I))` 与经验CB平方目标相对 W0 的差精确相同；P0须核验。额外1e-10是已声明的B坐标唯一化/数值项，不在看结果后改变。

CVXPY+CLARABEL，float64，`max_iter=200, time_limit=120秒/solve, tol_gap_abs=tol_gap_rel=tol_feas=1e-8`，仅接受OPTIMAL且重算归一化最大约束违反≤1e-6。最终保存B、W、xi、objective、primal/dual残差与求解器版本。`OPTIMAL_INACCURATE`/超时不得静默当成功或回退W0补齐结果；允许继续不受影响的其他候选，但全矩阵标记PARTIAL。

未安装时在隔离环境进行一次最小依赖安装并记录精确版本，禁止升级训练环境的torch/cuda；无法安装则PARTIAL_SOLVER，不伪造CT-Risk结果。缺包可以不阻塞U/R完整轨迹和其余方法。

### 5.3 CT-ConCM：有神经梯度更新的统计回放路线

R 神经流采用相同初始化、真实all-seen损失、任务预算，并加入旧类合成特征CE：
```
L_R = L_real_APART_allseen + (C_old/C_current)*L_synthetic_sum_CE
```
Task1旧类数0，回放项0。系数是替换为旧/新类数比，不是再乘历史0.05。该配方继承V9-K的回放系数逻辑，但属于CT1下的新变体。

每个old class：raw main/few 均值与population对角方差独立高斯采样，每类4对；全局均匀随机裁到48对。使用独立synthesis RNG，保持真实图像增强与loader随机流和U对应。
```
std = sqrt(max(var,0) + 1e-6)
```
不设方差上限，不归一化合成raw特征，不添加完整协方差或main/few相关性。使用第4.4节epoch-start输运后的raw统计。该分布近似的局限必须报告。

合成特征直接进入 main/few heads：
```
CE(head_main(m_tilde)[:C] + head_few(f_tilde)[:C], old_label)
```
主预测使用raw main+few all-seen logits，无HN。合成CE对head有直接梯度，对上游adapter没有直接梯度；adapter每任务由真实图像损失更新，不能宣称旧伪特征直接训练了视觉编码器。

原 `concm_stage1` flags 仅作底层兼容，不触发S0或first-task freezing。关闭原0.05 schedule、controller、MPC/DSM、uncertainty gate等其他历史路径。统计版本、方差公式、合成配额、实际权重逐epoch记录。

### 5.4 固定参照/诊断

- PT-CB：原始冻结预训练CLS＋CBRidge，按新task边界重新评价。只它不更新adapter，明确是参照，不代替用户要求的三个持续适配路线。
- CT-J-Stale：独立诊断统计支路，旧类统计停留在各自到达后的旧空间，不输运；当前类追加U当期真实矩。不得将它称为有效的同空间RLS。其失败不自动证明输运准确。
- CT-ConCM-CB：在R自己的最终encoder上，使用相同joint定义与相同输运规则，求CBRidge。用于区分合成训练后的表征与head读出的影响，不按分数选择主预测器。

## 6. 锁定、训练与评价流程

### P0：工程和资源

定位实际服务器/目录；读取已有协议/权重锁，保护工作区。生成新代码分支与CT1锁，所有新路径与历史分离。

必要验收：
1. 公共权重严格加载，U/R同一seed初始化一致，无下游S0残留。
2. 任务覆盖无重复/遗漏，Task1大小2，HK最后任务3；未读取旧或未来训练图像。
3. 每任务同日程，T>1 adapter梯度和参数变化非零；无隐藏first-task freeze。
4. 三真实CE均覆盖all-seen；未来输出无数据梯度；真实标签不影响推理/路由。
5. S0_POINTWISE_PROBE/native B1一致、batch同伴不变、推理数值调度不进入训练。
6. loader/增强/synthesis RNG隔离；compact恢复后下一步一致。
7. 输运identity、已知toy仿射的均值/完整二阶矩/对角方差准确；anchor→epoch映射不重复复合。
8. joint Gram交叉块保留；agg S更新与显式class-balanced目标等价。
9. CT-Risk eta=0恢复W0；子空间目标差等价；小维toy风险约束和求解器状态正确。solver最大类数23用纯合成统计完成吞吐/内存测试，不能用未来真实统计。
10. PT-CB最终val重现原相同数据和特征的高精度参照，证明不是数据接口问题；不拿旧test分数作目标。

数据/标签/状态准入失败阻塞相关运行；未知患者关联沿旧报告保留，不循环等待不存在信息。

### P1：两数据集的U/R完整任务轨迹

先HK、后ISIC；每数据集按1993、1994、1995，U/R依次完整运行，调度顺序不按性能更改。每task按第4节探针与统计更新，保存最后epoch checkpoint。

每task结束可以提取**已见类别的val**并保存封存分数，供旧模型可恢复核验；不得输出实时排行榜或把val返回trainer。不可访问未来类别的val来选择参数。完整矩阵锁定前只看技术检查和train日志。

CT-J/CT-Risk在同一U状态上读出，不能由Risk分类器初始化下一任务的U训练heads；ConCM-CB也不能回写R的heads。否则神经轨迹不再一致。

### P2：全部读出和封存评价

全部规定候选/阶段生成后验证覆盖与哈希，统一解封本轮validation指标。求解失败保留PARTIAL状态及失败单元，不偷换候选或参数。

未来任务迭代不受本轮validation分数影响。不存在best epoch、best branch、挑seed或自动扩展。

### P3：完整报告后STOP

不把新协议分数与旧A1/HK1数值直接相减当配对提升；旧大基础阶段只能单独列作历史上下文。所有差值必须在新CT1相同task划分下计算。

## 7. 评价与解释

主指标：Final BA、全任务平均BA、Final tail recall。另报逐task/逐class正确数与分母、old/current及HM、频率组、任务首次学习质量、有符号遗忘与最大遗忘两个明确区分指标、current→old与组内错误、无新零召回检查、最差run。

不再将“大基础类 vs novel”作为主分组，因为没有base阶段；可以报告first-task classes，但只能称Task1 classes。

真实训练日志同时报告current-restricted与all-seen。Restricted-current仅作诊断，不当主CIL结果。Final任务的类没有后续观察，首次到最终遗忘NA；0→0不当成功保持。

漂移诊断：当前train pre/post拟合残差、a/b统计及clipping率、空间版本、各类记忆年龄、e代理、旧均值范数、G trace/condition、query margin。验证集旧类pre/post差异可以在完整解封后用于评价诊断，但不能回写输运或风险估计。

CT-Risk：每类xi、活跃约束、风险代理与实际错误的关系；这是相关性诊断，不称临床概率校准/覆盖保证。

三个固定配对的均值、ddof=1标准差及逐配对差值保留；共享readout和阶段不作为更多训练重复。PT最终解在固定特征/类集合下顺序不敏感，只报告一次确定性最终整体结果，old/current分组另按order列。

2,000次按类分层component级配对validation bootstrap，seed45001。方法间和run间共享抽样，固定已训练模型，不重采样run来伪造训练总体。singleton component不产生病例不确定性，必须标注；无可靠关联则区间NA，不擅改独立逐图像抽样。

不设通过后自动继续的门槛。无论结果正负/混合，完成固定矩阵后STOP；不自动test、新backbone、全量ViT训练、其他dataset、GSR/DSM或监测。

## 8. 资源、存储与中断

- 单worker，一次一条GPU训练流，最多8 CPU核。
- 默认全部新GPU-process residence预算12小时，CPU读出/报告预算6小时。预算不是预测耗时；包含P0 smoke/修复/提取，并分别记录训练、probe、读出成本。
- 允许一次有限依赖安装，禁止购买算力、改变CUDA核心环境或重新下载大模型。
- 先小规模train-only吞吐和23类纯toy求解测算，再锁执行预算。任务epoch数量虽900，每个图像每条轨迹仍仅在其所属任务出现10个训练epoch；额外probe曝光另计。
- hb01新增活跃文件峰值≤3GiB，始终保留≥1GiB空闲；跨现有hb01/my-gpu的新归档总量默认≤10GiB。
- 允许把**本轮新建、已完成task checkpoint**转存到用户现有my-gpu归档新子目录；传输SHA/严格可读恢复PASS后，回收本轮的hb01临时副本。不得迁移或删除任何历史checkpoint/数据/权重，不能把归档空间当无限。
- compact checkpoint保存全部非共享参数与公共core引用。每task最后状态必须可恢复，不能只保存最后一个任务；活跃resume仅一份原子滚动。
- 每类只保存对角风险统计与均值，完整joint二阶量用aggregate S，无需把23份1536²矩阵每epoch存盘。若保留完整逐类矩阵会超过预算，不在工程中暗增此存储。
- 成对current特征缓存只保留active task，raw float32派生J时float64；无需持久保存每epoch全特征。
- 不对历史滚动文件继续清理。空间、依赖或预算不足真实报告BLOCKED/PARTIAL，不擅自缩epoch、改batch、删seed或放宽统计目标。
- 技术修复最多两次相同问题的有界重试，不依据性能修复；协议变化须新版本。不得创建/恢复每小时任务。

## 9. 交付

```
CT1_PROTOCOL_AND_SCOPE.md
METHOD_ALIASES_AND_DEVIATIONS.md
DATA_AND_TASK_LOCK.json
WEIGHTS_AND_INIT_LOCK.json
CODE_ENV_LOCK.json
P0_ENGINEERING_TESTS.json
MODEL_LINEAGE.json
train_epoch_metrics.csv                 # 900正式task-epoch
training_step_audit.json
transport_audit.csv
risk_solver_audit.csv
validation_main_metrics.csv             # 三主方法135
validation_main_per_class.csv           # 三主方法1377
validation_all_metrics.csv              # 六方法270
validation_all_per_class.csv            # 六方法2754
paired_differences.csv
bootstrap_intervals.csv
MEMORY_ACCESS_AND_RESOURCE_AUDIT.json
FINAL_REPORT_ZH.md
NEXT_DECISION.json                      # STOP
```

同时交付90个逻辑且可恢复的任务末checkpoint（实际U/R均跑，不以共享结果伪造计数）、解析W和Risk的B/slack、父状态/公共权重依赖。公开仅源码/协议/聚合，不发布私有图像、身份、逐样本分数、特征或权重。

最终必须实填：没有单独S0、每task adapter更新数、encoder调用、neural optimizer steps、convex solver iterations、旧训练图像访问=0、未来训练图像访问=0、新test各访问=0。不能因Risk共享U或ViT core冻结，就写本轮神经训练为零。

## 10. 来源与非等价说明

已核对的项目来源：
- `4460e870217b3a903f03ba64f21bdc711c4d6bb0`：HK1方法卡、`tools/prepare_hyperkvasir_hk1_m1.py`、APART配置和`third_party/APART/models/apart.py`。
- `2fdca56b513b9ea67072e91085a6a07e39f93e47`：A1归档及pointwise probe修复。
- `a59156d2d2b0b2b4096e6f91e224e660374bdf10`：V9-K经验方差与回放系数来源。
- `8507c2d22916cce7c12c68527eb2515585c0de9e`：G1强基线及类均衡目标核验。
- `a18b25a8c582a0ac0f027bb177dcbdb85ddd7596`：ISIC V2数据/预训练锁。

外部参考只用于定位设计边界：
- Yu et al., CVPR 2020, Semantic Drift Compensation：用当前数据估计旧表征漂移；CT1对角仿射不是其完整复现。
- Gomez-Villa et al., ECCV 2024, Learnable Drift Compensation，arXiv:2407.08536：持续变化表征下的漂移补偿。
- 上传ConCM论文：有基础阶段、MPC/DSM；CT1不是完整ConCM复现。
- 上传GSR论文§3.1：冻结编码器下的解析统计；不为CT1变化表征的精确RLS背书，本轮不跑GSR。
- CVXPY solver documentation、Clarabel settings：凸读出实现依据，不是医学有效性证据。

本地仅完成合成数据的代数检查（aggregate仿射、子空间目标差、CB目标等价）；没有医疗模型训练，也没有在当前对话运行CVXPY风险求解器。服务器P0必须自行核验全部实现。
