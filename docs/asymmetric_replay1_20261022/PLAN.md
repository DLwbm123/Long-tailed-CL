# 非对称准入重放：独立 NIH 主线实验（冻结）

## 完整机制及强对照

本轮从“当前类如何获得判别能力、何处学习新旧类竞争”重构更新与记忆准入。不同于已关闭的 native-token 上层训练：不保存197×768 token，不训练预训练上层，不接旧 cache/头/teacher/score。新鲜编码冻结 ViT 全部12块，拼接 CLS 与 patch 均值并逐图 L2，得到1536 FP32输入；学习单残差1536→128 SiLU→1536（up零初始化）和scale25余弦head。旧功能蒸馏/硬风险/对比/扩展分支均不加入。该学习器容量与本轮ER强控制完全相同，不宣称新架构创新。

每实际阶段：仅编码已经到达类；以metadata seed74002预选每类至多32患者、每患者一图的不可变输入feature，**本阶段第一次更新前**准入全seen memory（最终615患者而非640）。当前完整feature只驻留本阶段RAM，结束释放；每新类head从当前患者等权原始feature均值初始化。全seen replay因此同时有新旧正例，不是仅对old memory做全seen损失。

两通道每步各32逻辑行：incoming从当前类均匀抽类→均匀患者→均匀影像；replay从全seen类均匀抽类→均匀保留患者。均有放回，2个输入dropout.1 views，共128模型行。同一批索引、dropout张量直接供两个臂使用。

令K=seen类数，N=当前类数，q=N/K。固定联合目标：

`L = q/2 * mean CE_incoming + mean [w_y * CE_replay_allseen]`，`w_y=1(old), 0.5(current)`。

ER：incoming CE也对全seen；ASYMMETRIC主臂：incoming CE仅对本次当前类，replay仍对全seen。期望每old类风险权重1/K，每current类来自incoming与replay各1/(2K)，合计1/K。实际未加权抽样类频并不相等，不能称实际曝光每类相等；**两臂实际曝光逐步匹配**、期望目标类权重相同。主不添加辅助项/调权/teacher。基阶段mask为全部类，两个臂state必须完全相同。共享表示更新仍能间接影响旧类，不承诺零遗忘。

这是成熟非对称重放思想的阶段化、患者类平衡适配，非忠实在线ER-ACE复现或新TMI。来源Caccia等ICLR2022 arXiv2104.05025v3，第3、4.1—4.3、5.1—5.2、附录A.1与B相关段已读，非全附录/官方代码审计：https://arxiv.org/pdf/2104.05025 。原文为自然图像在线流，不能外推NIH收益。与其不同之处已显式冻结：stage全量当前访问、确定性患者类记忆、显式准入、双通道类权重及预训练冻结feature学习器。

## 科学配方、数据和停止门

NIH author20原单标签train68058、val300（每类15）、10+2×5阶段，seed74002及order见config。患者交叉0，不读test/Etest，不下载新数据/借未来类/根据val调参。2臂同容量410240→425600参数、同记忆/抽样/优化机会4500（base2000+later500×5）、AdamW3e-4/wd1e-4/clip5，每阶段reset，FP32训练/FP64部署、encoderFP16、禁TF32、CUBLAS4096:8。无扫描/重试科学配方，不更换主臂。

主九门全部要求：final≥91/300、mean≥36.005291005291%、tail7≥25/105、initial≥44/150、later≥41/150；相对本轮强ER final+4、mean+1pp、tail4+2/60、later不降。任一科学失败关闭固定配方，不改阈值或扫lr/容量/记忆/权重/seed救分。对照不优化追旧ER51，跨轮协议不同不混因果。300val为重复开发筛查而非独立临床test/SOTA。方法设计不能保证实际正向效果。

## 预算与真实GPU准入

独立本实验3,000,000,000B硬限/2.7GB预停，预计输出<0.2GB（18.2GB全token表不再存在），RSS16GiB/GPUreserved8GiB；准入host16GiB/GPU0free8GiB，实际NFS余量与小写读探针。首次远端准入到所有组退出≤8h，准备/数学/资格/故障/等待/验收共享原钟，绝不重置。独立guard5s统计全新增根+中性入口max(st_size,st_blocks*512)，截止前60sTERM/30s必要KILL，保留费用/部分结果，不自动科学重试。

人工新增离线窗口：北京时间2026-10-22 23:15:19—10-23 11:15:19，唯一30min监测原位。窗口内所有新实验/重试/后继/资格产物累计新增严格<30,000,000,000B，27GB预停；本轮3GB更严。后继启动前必须登记所有窗口根及入口给独立guard合并核算，不按新名重置30GB，不删旧证据腾额。窗口末给真实汇总，不把健康后台擅停；单实验原截止优先。

bulk硬门：20真实合法基类图，经正式reader→GPUFP16全ViT encoder→正式GPUFP32/128views两臂base及内部8+2到达/memory/head两次完整前反向/clip/AdamW（4丢弃步）→实际参数变动→保存加载→FP64/独立NumPy score和目标；数学目标/有限差分/梯度路由也须PASS。程序写QUALIFICATION PASS后才正式68058编码，未来图/val未用于资格。首次工程错误不计科学效能。

## 独立验收、成本和交付

12模型全部val前seal，300val仅编码一次。run0自动独立verify：NumPy12score≤2e-6且预测全同、真实训练probe目标、metadata独立memory复选/旧feature完全不变/患者隔离、准入先于第一更新、replay含已到达新类正例、实际seen与更新/曝光计数、基阶段state完全同、资源原钟。验收0新编码/优化；独立重算端点及probe，不称独立重放全训练轨迹。

正式每臂288000逻辑抽样行/576000 views/4500更新，incoming与replay各288000 views；初始化feature聚合68058共享，不是隐藏模型调用；probe768/val1350每臂。资格20编码、每臂256 GPU前反向行/2更新、FP64与NumPy各256行；数学synthetic各16行前反向+112 NumPy行/臂，梯度路由另16行前反向，0synthetic优化。全成本保留失败/未知准备、读取等待和嵌套时长，不把重复当新样本。实际完成须全组退、独立验收、完整中文指标比较/费用、匿名源码聚合经有效7897代理push/远端SHA/匿名访问。身份/feature/权重/逐图score/原日志私有。无需人工审核。
