# PMT：配对矩运输的主四格与独立区域贡献协议

本轮以已公开的 PMT CPU 设计为前瞻依据；当前生产 CPU 检查和正式冻结完成前不启动 GPU。保留原 Core 与 local_transport；不降低控制、不以已见 val 混淆选类别对、阈值、秩、强度或步长。

## 方法与可检验假设

当前任务同图 teacher/current 的 canonical indexed 行构成配对；以当前类等质量、类内等权拟合行向量仿射映射 `y=x(I+UV)+b`。相对 ridge 固定 .001，秩仅由机器阈值决定。未观察输入方向保留线性残差 identity，不声称旧域真实无漂移。旧域推广是待验证假设。

旧类全局均值、二阶矩及 component 中心按同一仿射 pushforward 一次，保持原 n/mass；不伪造历史样本再次 L2 归一化。非等方旧半径记 unknown 并保留有效上下界；原 head、competition、controller 不消费这一半径。当前新类使用真实 current 特征统计，不套旧映射。每 epoch 只用 current fit 配对，从不可变 task-start 快照重新构造；boundary 与原 common_shift 相同，使用 all-current train（含原 meta），仅封存一次坐标 frame。meta 不是独立校准数据。

local_transport 保留原 regional prior 公式。组合单独拟合四个区域各自的映射，不能拿全局 CLS 映射替代区域映射。全局-only机制对照保留原四区域 translation；不叠加 common_shift 或重复 component 中心。原解析目标、class competition、Woodbury/proximal、FD 和 envelope 梯度都保留。

## 固定矩阵与预算

复用 Core/base、local_transport 四条历史训练轨迹；新增 `pmt`（Core+global PMT）、`local_transport_pmt`（global+独立四 regional PMT）、`local_transport_pmt_global_only`（global PMT+原区域 translation），各 ISIC/HK，共六条。

ISIC 每臂 100/72/296=468 更新；HK 每臂 98/4/2/20/18/34=176 更新；总1932。每task 2epoch、batch64、workers4、AdamW .0003/WD .01、clip5、FD10、ridge .001、proximal .01、原 per-task cosine eta_min1e-5；原 seed/order/split74002/fit-meta/变换/缓存 IN21k→IN1k 权重和 train/val 全部复用。训练时零 val 读取，无旧训练图像新增访问/回放/校准，test/reserved封存，RL暂停。仅GPU0/1。

六条真实 firstbatch backward 均零 optimizer、参数/buffer精确不变、梯度非零有限；另各做显式 synthetic-map shadow 1536维、输入满秩1536、最大旧类数（ISIC6/HK21）矩运输与解析求解/峰值检查。shadow不是实际旧类漂移或正式统计，不计正式更新。全六 PASS 后才 TRAIN；全六精确 TRAINED，步数/全部 stage metadata seal 核对后才统一 development val。worker 在打开 val manifest 前复核屏障。

## 完整报告与交付

主四格相对各自控制、full regional−global_only、global_only−LT、组合−单独、交互及 AIR gap 全部正零负；六指标 BA/avgBA/tail/forget/old/new、Acc/avgAcc、提供的 F1（缺失 null），全部 stage/31类 n/recall、old/new混淆、救回/破坏/错误换错误及真实类别间隔。类间隔以配对控制相同 canonical val 计算；不以平均守卫称逐样本无损。报告训练 map 秩、配对残差、范数、旧半径界及所有费用。

正常6preflight+6train+6eval=18次GPU进程驻留，每次出现保留；配对控制推理包含在新eval费用，旧训练不重收费。失败额外全保留，missing不补0。CPU pipeline含于operation wall及实测驻留，client wall另列，不重叠相加。实际launch起24h包括等待/预检/训练/评价，不提前虚造时间；单次launch、防重collect/deliver。

源码/协议/匿名全部正负与失败/报告/AUDIT显式代理公开 GitHub，核对remoteSHA和精确报告HTTP200，再NAS匿名public_delivery/独立DELIVERY/HANDOFF。权重、影像、样本/逐样本输出、映射/统计矩阵、privateconfig及raw logs不外传；允许匿名类别混淆计数。

单seed/order、反复development val，不能作为独立患者/临床/test/显著性或等资源SOTA证据。未观察旧漂移与新类无损均无保证；不能把数学一致或非零激活当净收益。
