# R130 真实准入与有界机制资格启动（非最终结果）

状态：真实native通过，固定诊断已完成；五臂科学资格已进入RAPO臂，至少5/32次更新实际非零。其余臂和最终资格尚未完成，不宣布方法有效。单转换32步不是完整continual，也不替换R125/R128失败门。

## 真实准入与修复

原准备1794310608.4689498、23h截止1794393408.4689498不改。初native源码ceaf24ba7304df6c2c97df262081cc927ebc70e5；当前运行修复源码83bd2b662173edc4a7d591120cd15c341be159f7。只修跨设备恢复断言/保存收据时机，科学目标、随机流、五臂、32×8/G8与门不变。失败完整封存，详见FAILURE_AND_RECOVERY_ZH.md；未重复native。
原实测完整forecast 11433.555004秒，已含1.5margin+1800reserve，小于当时remaining 78127.785984秒；恢复时重新核剩余仍足够。五真实优化路径均非零并全部丢弃还原，匹配hidden/head差0，rawfullhead差.0625/采样TF差.0759369保留，.05阈值未放宽。正则full/conditional真实shifted-target工程profile梯度均非零，不作效能。
原生A/B虽然retention恒定，但实际浮点梯度norm .516884/.517129并非逐字相同；不将之后A/B全部差唯一归因retention或证明CTAN全效应。

## 固定TRAIN探针：局部方向冲突

R125独立RAPO stage1保存权重/own首task anchor，预定自然首batch8query/G8一次64响应，正确40、mixed4；不是患者独立重复。accuracy PG norm 2.124922，retention PG norm 1.131444，tokenKL norm 0.139082；accuracy/retention梯度余弦 -0.984864。该检查点/批次两项明显反向，不能推广到全任务/VAL的原因或断言所有retention有害。
合并梯度裁剪前/后norm 0.902570/0.902570，一次counterfactual Adam实际L2 0.00853693，全部丢弃并核actor/anchor还原。TARS_PG仅另一臂固定表的诊断加权，不冒充RAPO训练表或加入五臂。
成功探针原始tokens/mask/text/old/ref等留远端私有。CPU-only独立核当前tape、文本/EOS正确性、h/reward/CTAN通过，0新生成/encoder/optimizer。梯度norm/cos为实际前向反向仪器量；未保存全梯度向量，因此不宣称CPU独立重算过梯度。第一次失败探针未存原token，不补造或当成独立重复。

## 首转换菜单×参数2×2：全部旧5类macro

| 保存参数 | 原5类菜单 % | 到达10类菜单 % |
|---|---:|---:|
| 共同首task前缀 | 74.4379 | 45.9148 |
| R125 RAPO task1后 | 65.6921 | 40.0816 |

同624旧VAL组件/相同batch的4读出共2496响应，私有预测/整数分母CPU独立核过。交互F11−F10−F01+F00=+2.9126pp；预定路径菜单项-28.5232pp、最终菜单参数项-5.8332pp，和为-34.3564pp。另一菜单下参数效应不同，不能赋予唯一因果百分比。这里是全部旧5类macro，区别R127的首task四head类指标，不混用。
旧菜单仅回顾性诊断，不部署挑菜单、不改变原head门。该诊断没有证明旧菜单对当前新图的teacher有用；所有结果不进入loss/系数/采样/选择。

## 五臂与后续判定

A独立RaPO；B正确性PG/同own totalreward CTAN统计；C=B+全10候选KL；D主=B+qO旧5条件KL；E强CE+同G8 rollout tokenKL+同D正则。C/D/E同候选评分信息及路径，E额外CE费用公开，不保证同梯度强度。候选归一化是有限canonical+EOS字符串分布，不等于全部自由生成语义概率，也不保证旧图风险。
所有臂同prefix/自然query/更新预算、各自model/Adam/sigma；无旧TRAIN/replay/TARS校准/答案teacher。全五臂32步seal后才VAL；D对C的macro/tail、旧类保留/新类获得及匹配CE全部预定门通过，才具完整CL后继资格。任何失败不扫系数/seed/菜单救分，完整5task128/G8仍须另冻准入。

## 成本和边界

首失败160生成/5native+1probe optimizer/4TRAIN工程编码/wholeworkerCPU73.672867秒保留；成功attempt计划18584生成/160science+1probe optimizer/0新编码。全attempt计划18744生成与167 optimizer尝试，区别原无失败18680/166预算。科学CE256query，native CE另2；科学C/D/E anchor/eval/backward各7680候选TF行，实际每phase TF rows/inputcells与generation active/padded/decodecells留账。
CPU软件完整child5.224675秒；probe保存审计完整child2.090272秒，菜单保存审计完整child2.011244秒，body均只是子集不重复加。两次审计进入INTEGRITY_PULSES，finalizer汇总；准备shell/SSH/Git CPU未知非0。科学worker尚未退出，不虚构最终CPU或成本。
现场root物理保守89530368B（含失败目录），全部封存基线7605948416B+本机200MB+本root=7895478784B，严格<10GB。独立2s guard原因None，原deadline−120/9GB/RSS48GB/freeFS1GB预停；实际mount写读probe及GPU1显存核实，只GPU1，不动他人/旧任务、不删账。
source-only公开代码、安全聚合/数学/成本，不公开私有tokens/逐例预测/索引/访问日志/权重/feature/身份或私有路径。实际退出0后才本R130 CPU-only完整保存验证、独立整数复核及最终交付，不提前finalize。
