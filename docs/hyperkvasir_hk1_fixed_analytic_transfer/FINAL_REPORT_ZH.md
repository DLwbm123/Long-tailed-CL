# HK1 P0 前置审计：BLOCKED

截至2026-09-17，实际状态为 **BLOCKED_HK1_CLASS_MAPPING**，初始的 **BLOCKED_STORAGE_ON_CURRENT_SERVER** 已按用户追加授权解决。已执行官方metadata校验、10,662张白名单文件定位、23类映射核对、三个历史seed的逐样本划分核验及当前服务器资源检查。未进入训练、smoke、特征提取或解析拟合；不将计划的30 epoch、54/972行写作完成结果。

## 1. 类别规则与实际数据不相容

复用了已有官方CSV，无下载。Git blob与附件指定的`0d8abe6781a82c4452416390d93a176f875a6f23`完全一致；SHA256为`0230b465bdf73c23e949c74e1d4d68c4905a758195ea6b69b10f32c71a3be426`。白名单包含10,662个唯一文件、23类，全部图像路径可定位，未发现缺失或额外图像。

按精确文件名对齐，官方CSV名称与旧项目图像目录是23类一一对应，但**8类名称不同，20类字典序数字ID不同**。这不是按模型分数发现的现象，也尚不能视作标签冲突。完整23类证据见[CLASS_MAPPING_DIFF.csv](CLASS_MAPPING_DIFF.csv)。

| 旧名称 | 旧ID | 官方名称 | 官方ID |
|---|---:|---|---:|
| barretts-short-segment | 1 | short-segment-barretts | 16 |
| cecum | 4 | normal-cecum | 8 |
| esophagitis-a | 7 | oesophagitis-a | 11 |
| esophagitis-b-d | 8 | oesophagitis-b-d | 12 |
| hemorrhoids | 9 | hemorroids | 5 |
| polyps | 12 | polyp | 13 |
| pylorus | 13 | normal-pylorus | 9 |
| z-line | 22 | normal-z-line | 10 |

附件PLAN §2.1要求：“核对其与旧 `_discover_hyperkvasir_classes` 的映射完全一致；如不一致，输出差异并阻塞，不静默重新映射旧划分。”实际不满足此项，故停止。若直接用官方ID替换旧分折公式中的class_id，`default_rng(1+7919*class_id)`的种子随之改变，不能声称仍保留旧外层成员。

历史身份名单确实存在，不需要从特征/预测容器中取出。seed1、2、3/fold1各10,662条成员记录均与锁定旧代码规则一致，逐样本不一致数均0；每个seed均为outer_train 8,519张、reserved 2,143张。它们仅用于身份核验，没有读取旧模型特征或逐样本性能。历史seed1外层成员已可被明确保留，但本轮新ID的选择需要显式修订上述要求。

可供批准的最小修订：保留历史seed1/fold1逐样本成员不变；按已核验的精确文件名建立明确别名表，HK1使用官方名称字典序ID，附件三个数字order保持原样。此方案会修改“ID与旧项目完全相等”的前提，**目前未应用**。另一选择是保留旧ID并另行规定官方名称映射及order语义；不能由执行者静默选择。

## 2. 初始空间缺口与已完成修复

当前项目服务器hb01:30154的4090 D空闲显存24,081 MiB，数据盘实际剩余1,897,340,928 bytes（1.767 GiB）。已有HyperKvasir图像位于my-gpu的项目NAS；当前服务器现有项目树未发现HyperKvasir资产。

10,662张图像的逻辑文件大小合计3,934,375,143 bytes（3.664 GiB）。早前`du`显示约7.5 GiB是源文件系统的已分配空间口径，不等于传输字节。即使只搬运历史outer_train的8,519张图像，也需3,146,187,899 bytes（2.930 GiB）。加上计划要求保留1 GiB空闲，**尚未计任何实验输出就短缺约2.16 GiB**。不能靠768 MiB的实验输出预算解决图像缺口。

输入预检时未执行迁移。随后用户明确授权清理空间及迁移重要文件；现已完成下述归档。未传输HK1图像、下载权重、购置算力或更换运行服务器。my-gpu的既有NAS可读写，已完成一次小型写入/读回并清除本轮探针。它仅承载本次无模型的数据前置审计；未在其GPU上启动训练。后续仍需检查现有权重、训练吞吐和compact状态峰值，不能把本次静态检查算作工程/预算通过。

归档完成于2026-09-17 03:02 UTC：6个V3 D/E滚动resume合计4,537,984,738 bytes（4.226 GiB）迁至my-gpu私有项目NAS。直连rsync退出0，目标大小及PyTorch ZIP中央目录/version可读；未重复做全量hash，也未宣称经过逐tensor恢复。核验后逐项移除原服务器副本，源目录保留归档标记和总索引。历史资产未丢弃，12个V3正式session checkpoint、原始权重及S0资产未动。

磁盘空闲增至6,435,332,096 bytes（5.993 GiB）。按outer_train逻辑文件大小、768 MiB输出上限和1 GiB保留量估算，尚余约1.31 GiB余量；该容量修复不等同于尚未执行的compact状态、实际GPU/吞吐或完整工程验收。最初1.767 GiB的读数保留为前置证据，当前状态见[STORAGE_RECOVERY.json](STORAGE_RECOVERY.json)。大文件归档只改变既有资产位置，不是新增HK1模型或实验结果。

## 3. 已执行与未执行的边界

- 已创建独立分支`exp/hyperkvasir-hk1-fixed-analytic-transfer`，基于A1交付`2fdca56b513b9ea67072e91085a6a07e39f93e47`；原source-only目录及五个既有CSV修改保留，没有合并main。
- 已读取V2实际配置、S0训练三项CE及assignment/pull实现、optimizer/embedding工具，以及A1只用于推理的Linear修复和缓存锁。8类/3任务硬编码被识别，尚未制作或运行HK1训练入口；不将静态阅读称作loss/gradient保真验收。
- 仅读取官方metadata和历史身份名单、枚举图像路径并查询文件大小。尚未做图像字节/像素哈希、去重或冲突隔离，因此未选择内层fold、生成最终fit/val、计算真实fit不平衡比或锁定tail8。它们必须在映射问题解决后按原规则执行，不能用旧统计补填。
- 所有正式/工程训练epoch、optimizer step、encoder forward、解析拟合、val结果行均为0。test预测、test特征读取、test模型前向及历史逐样本分数读取均为0。
- reserved图像载荷读取和解码均0；有2,143条reserved路径的大小元数据查询。不能把元数据审计说成完全没有接触reserved身份信息。完整计数见[ACCESS_AUDIT.json](ACCESS_AUDIT.json)。
- 输入审计用时约5.24秒；GPU驻留和CPU解析拟合时间均0。没有创建/恢复监测任务，没有新增方法或修改科学规则。

## 4. 最小待决事项与停止

继续前需批准具体标签/别名映射及order语义修订。用户随后已明确授权清理空间或将重要文件迁到my-gpu；已归档6个已完成V3的滚动resume，正式session checkpoint保留在hb01。原空间不足是修复前事实，不再要求更换计算服务器。

历史HyperKvasir开发暴露仍然存在，本轮也不会成为独立确认。不存在HK1性能结果或迁移结论。**NEXT_DECISION=STOP**；保留审计后等待上述实质问题解决，不训练、不重分折、不恢复定时监测。
