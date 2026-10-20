# GROUP_GRADIENT_AGREEMENT：固定参考处的组梯度一致性

## 独立问题与边界

历史梯度质量ESS只测mapper输出z处的范数；D只测表示方向漂移；真正排组producer与匹配ridge已终止CE-bias性能救援。这些均未测真实102912既有bias参数处各训练组梯度方向是否一致。本轮独立零更新诊断：少数组在固定原预训练模型、该fold support-only F头下是否给出相反的平均CE梯度方向？不拟合或部署新头、不重启任何旧训练、不称SADP AGOP复现；其选定中间层/归一化未唯一，本轮对象是参数梯度。

所有结果只条件于两个旧固定support半组及其F参考，不能从训练组梯度推出query风险，不能据此选保护方向/lambda/模块。没有风险收益或method_success。若方向一致，也不是方向泛化已认证；若不一致，也不是过去损害的因果定位。

## 输入及固定控制

仅initial2555及已封存split seed74290。只读producer_group_crossfit的prep PLAN、两个support-only teacher和源依赖，复用初始pixels0；不读旧T/P/C/D learned encoder、任何query逐例结果、681开发、future/test或rawimages。共享chunk含681必须立即按原row_ids过滤。原APART FP32/pointwise/双分支归一化G1535、mapper及alpha.01 F头不变；原native双head冻结，102912既有bias仅requires_grad，绝不optimizer.step/参数更新。

每fold每类：排序support已知组后用单一Generator seed74310依fold0..1/class0..3顺序randperm取8组；类1每半恰好8组，所以全取，不按梯度选。每组包含全部原训练图，组loss为图均值。8组再按已抽取顺序前4/后4两个固定半组。总64组，最多512图；超限工程未决停止，不截断、不换seed。每个teacher、mapper、所选组均来自该fold support；另一半query组不计算该teacher下梯度、不参与归一化或任何选择。此为support内稳定性诊断，不另造outer-test。

四类和两fold共8cell全部报告，不以类3代偿类1。固定参考头输入z梯度作为低成本对照，与真实bias梯度用同图同组同CE；它不等于encoder参数梯度。无额外随机方向或筛选训练。

## 指标、门与停止

每cell保存8组平均参数梯度的Gram，公共只出聚合：两半平均梯度cos、8组单位方向平均成对cos、norm(mean g)/mean(norm g)、各组范数min/median/max、Gram谱有效秩。z梯度同样完整输出。零范数记NA并判UNRESOLVED，不把NA写0。向量维度102912，不保存/发布梯度向量；按组索引的Gram/清单仅NFS。

唯一主描述门是class1参数半组均值cos：两fold均<=0则CLASS1_DIRECTION_DISAGREEMENT_AT_REFERENCE；均>0则POSITIVE_CLASS1_MEAN_ALIGNMENT_AT_REFERENCE；其他为MIXED_CLASS1_MEAN_ALIGNMENT_AT_REFERENCE。任一缺失UNRESOLVED。若其他类也不一致，明确非类1特有。以上不是显著性判定；不跑bootstrap、不换分组/头/seed/metric追结论，method_success=false。完成后仅分析能排除哪一前提，不自动启动保护或适配。

## 数值预检、封存及资源

合成parallel/antiparallel/orthogonal/zero四例核cos与NA。实际首组最多32图，超限工程未决停止；逐图平均参数梯度与整组CE梯度allclose(atol1e-5,rtol1e-3)，原G回放<=1e-10；所有CE/梯度/Gram finite、覆盖64组/8cell、support/query互斥、参数版本与初始bias/nativeheads exact不变。完整64组清单先PRIVATE_PLAN_SEAL，指标只在所有组梯度完成GRADIENT_SEAL后汇总。最多512图外加首组重算；不为工程未决重试或追改容差。

独立原始1800秒（协调提前10秒留TERM5/KILL和落盘），含合成/实际预检/正文/等待，不重置。新增128MiB含16MiB本地Git/helper，单文件64MiB，RSS12GiB；GPU0启动至少8192MiB余量。预计新增<2MiB，仅组Gram和小回执，无新像素/模型/环境/feature副本。旧窗17GB软停20GB硬限，按max(logical,allocated)+defaultcache1040384+localGit1GB计；独立协调每秒守卫deadline/RSS/空间、仅kill本轮worker进程组。NFS实际挂载/容量/写读probe先核。父子及GPU入口中性。

首次运行前冻结公开源码/协议，完成需8cell/64group/finite/parity/参数不变/封存/多EXIT/退休/完整成本一次并账；全部成本、失败及NA公开。worker wait4含后代计一次，协调wall不叠加worker正文。图像/组身份/清单/Gram/梯度/模型/原日志不下载公开。唯一scale-med30min继续。
