# 生产 CPU 验收分析与证据边界

本轮用户要求的生产入口、关闭回归、快照保护/失效、checkpoint/eval 和预算接线已完成 CPU 验收。八个独立子进程全部 PASS，无超时或数值失败。只执行一次本轮终态收集，父/worker 与本地收集器均实际退出。科学源保持 `7a077b69f16d843573b2bc7af5a46d9701c29bb0`；报告工具独立，未同步进冻结科学源。此结论是 CPU 工程验收，尚无新医学结果。

## 关闭时原路径精确回归

用固定 science21660bf 的原训练源码作 reference，只复用源码，未读旧 NAS 运行或医学模型。小合成 encoder 的真实 run 调用链覆盖 ISIC 三 task、HK 六 task，以及 Core/LT 两背景；分别比较原 Core/LT 与关闭新增 hook、原 PMT 与关闭守卫的 PMT。18 个阶段中，model/adapter/head、global bank、component/regional memory、boundary pair weights、selected coefficients、steps/seen 全部精确相同，encoder forward 次数与曝光也精确相同。合成全链执行 200 次 optimizer 更新，不是正式更新。该精确回归使用 6 维合成 encoder，不冒充真实 ViT 的完整多 task 轨迹回归。

开启后全链至少一条合成模型轨迹发生变化，说明生产 hook 没有全部退化为空操作；这既不是分类改善，也不是医学贡献。四条完整合成 eval 和四个模拟零更新入口通过。完整 eval 用合成 manifest/loader/encoder；真正 barrier 另通过实际 evaluate 入口验证在 manifest 前拒绝 TRAIN phase 或坏 stage seals。非法 GPU/scope/预算/版本/test 配置也拒绝。

## 保护时点、范围与失效

epoch 在原 static_PC choose 后、minibatch 前，对当前 encoder/frame 下注册的完整 fit true-vs-all margin 投影；boundary 在原最后 head 重建后、checkpoint 前再投影。meta 只参与原 bank 统计，不成为保护点或新校准。reference 是同状态原 common_shift/local memory/native metric/competition 的 OFF 头，不是独立 Core/LT 轨迹。

全链记录 54 次事件（36 epoch、18 boundary）与 36 次有证书到无证书的失效。每次 optimizer 更新均推进 version；36 不是 optimizer 更新数，而是当时持有证书的失效次数。高维 Session 检查拒绝支持行替换和过期 version，advance 后 require 拒绝旧证书。生产投影 detach，中间原 minibatch proximal/loss/梯度不改；不会把旧证书延伸到 encoder 更新后，也不声称持续约束隐式梯度已接入。

## 全约束的非 identity 与负结果

以下均为 1536 维生产 Session、合成单位样本与固定已知旧 affine；ISIC 使用 Nfit9434、支持 rank1536，HK Nfit1038、rank1038。boundary bank 包含原 all-current（ISIC11835/HK1298），守卫点仍仅 fit。全部乘子从初始保留，无 active cap、无逐条 inverse direction cache；maxcor10 仅优化历史。阈值/资源门未改。表中收益是相对同状态 OFF 的二次目标改善占无约束可用改善的比例；判别变化排除公共 logit，二者不等同分类改善。

| 检查 | 约束/正dual | 判别变化 | 二次收益保留% | 最小slack | native逆次数 | 合成救回/破坏* |
|---|---:|---:|---:|---:|---:|---:|
| ISIC_T3_epoch_init_Core | 66038/513 | 0.967450150 | 99.980036 | -1.4073e-09 | 174 | 0/0 |
| ISIC_T3_epoch_init_LT | 66038/512 | 0.967601721 | 99.980035 | -1.9079e-09 | 155 | 0/0 |
| ISIC_T3_boundary_final_Core | 66038/515 | 0.967192632 | 99.979941 | -3.4226e-09 | 150 | 0/0 |
| ISIC_T3_boundary_final_LT | 66038/514 | 0.967341794 | 99.979941 | -3.5381e-09 | 134 | 0/0 |
| HK_T6_boundary_final_Core | 22836/149 | 0.900890076 | 99.987993 | -2.4527e-09 | 99 | 0/0 |
| HK_T6_boundary_final_LT | 22836/148 | 0.901260013 | 99.988032 | -2.2111e-09 | 112 | 0/0 |

*本轮高维救回/破坏的比较基线是同状态 ON 无约束 proposal，不是 OFF 或独立训练对照。六条全部 0/0，未显示合成分类改进；当前未导出错误储量，不能区分饱和和不跨界，也不为补该项而重跑。六条全是非 identity，但不能由范数/目标收益推断真实 ISIC 新类损害已修复。固定已知 affine 不验证真实 PMT 的旧域外推，旧旧风险的信息限制仍在。

六条 stationarity 最大 2.508742132e-12，互补最大 8.600294317e-14，绝对 primal-dual gap 最大 5.023537142e-12；全部冻结 KKT 门通过。全约束可解，解决了本合成规模下 64 active cap 的工程阻塞；不是扩大 cap 或筛点。

## Guard-aware checkpoint 和评分

18 个合成阶段通过权威 float64 head 评分与 checkpoint 校验；篡改 steps/force/frame/兼容 float 头/source 被拒绝。高维四条 boundary 另有实际序列化/读回及评分精确检查。以 H^-1(B+force) 重建并核对 stationarity/source/task/steps/seen/frame/transport version；boundary 重建相对误差不超过 1.297951438e-10。cert 只保存 force/reference/proposal/head 与当时全点标量凭据，不保存原支持或逐点 dual，因此加载时没有独立重验全部 margin 的证据。

守卫的权威 double 头与旧公开对照 stored float 不同。旧对照指标保持原样；未来独立 PMT 的 stored float 与 double 预测差异必须另列。same-state OFF/proposal 使用同特征 double，可隔离快照头，但不能替代独立训练控制。此次合成 evaluator 路径已覆盖这些诊断字段；没有新的医学精度差异测量。

## 真实预训练 CPU 反传与组合覆盖限制

既有公开预训练 12 层 ViT、1536 维特征和两张合成图 CPU 反传通过；optimizer0，全部参数/buffer 精确不变，双 pool 梯度范数分别 0.0685929675/0.0304771534，均有限非零，CUDA 未初始化。该项仅第一 task。最大旧 bank/全 fit 是另外六个独立 CPU shadow，不能合成同一次真实 ViT+全 fit 反传，也不是四条真实 GPU 零更新预检；当前真实 GPU 预检0。

## 实际资源、完整预算与尚未测量项

八条完整驻留和 user/system/peak 见 CPU_PRODUCTION_SUMMARY.csv。六高维共 824 次 native inverse 调用；ISIC 完整进程128–171秒，HK174–196秒，资源有实际非零成本。全链合成 forward 次数在关闭/开启间精确相同，不额外提图。CPU 输入已在主机，host transfer 记录0；GPU主机传输尚未实测，保持 null，不由0推断未来免传输。

四条高维 boundary 的实际私有合成 checkpoint 序列化大小分别151646262、152039731、435975215、437106092 bytes（约144.621/144.996/415.778/416.857 MiB），包含合成 bank，不含真实 ViT 参数或 optimizer，不能当未来完整医学 checkpoint 大小。最大 guard 四矩阵 tensor payload ISIC393216/HK1130496 bytes，不等于总文件大小或峰值内存；epoch 无落盘文件时 checkpoint_bytes 为 null，不填0。

前瞻主四格 Core/LT/守卫运输/组合与两个无约束 PMT 贡献对照保持冻结，未来只新增四守卫轨迹18stage、1288更新，复用八控制轨迹36stage，不重训控制。固定54守卫事件；canonical153234、student81436、FDteacher81436图 forward，load曝光234670；guard不增加提图。未来每 stage 自身/Core或LT/PMT 三次encoder推理；same-state OFF/proposal复用自身特征，但头计算不是0。CPU求解嵌入未来GPU worker驻留时不重复加墙钟。固定54×840=45360秒（12.6小时）是守卫软停止上限的静态总和，不是实测未来耗时，也不是24小时总预算已满足；等待、预检、ViT训练、PMT提取、评价等仍需计入。未来完整inverse次数/GPU峰值/传输/医学checkpoint/全campaign墙钟都未测，保持null。BUDGET_SPEC是前瞻规格，BUDGET_AUDIT独立列实际CPU证据与未知项，不后改科学预算。

## 全部失败与费用

首轮八子进程因目标 Python 静态嵌套块上限在解析即失败，未进入模型/数值检查，2.190434728秒已完整公开77bd9e29。R2仅改ExitStack包装及真实层数断言，新增目标解释器compile gate，保留数学/参数/数据/预算。R2 predeploy 首次代理TLS失败在NAS/CPU前，同代理重试成功；不算新科学驻留或第九子进程。全部原始私有诊断留指定NAS，只公开匿名原因。

R2独立进程驻留 1056.696917263 秒（约17分36.697秒），pipeline965.585483491和operation1042.369875220包含其中，不相加。父wall 1056.841117570 秒另列。当前user 1949.487690/system 62.374770秒，双线程user大于wall正常；peak 4529.46484375 MiB＝4.423306 GiB。首轮+R2共16次独立子进程尝试，累计驻留 1058.887351991 秒、user 1950.011976/system 62.588543；不叠加旧设计或cumulative。client/collector驻留未单测为null。新GPU0，旧闭合391490.24710153416秒不变。

本轮验收请求完成后不自动新增科学检查或启动GPU。医学图像/医学checkpoint/val/test读取均0，test/reserved继续封存；没有val调参。CPU通过不是医学贡献、独立患者/临床/统计显著性证据或GPU预算适配证明。未来GPU仍需用户明确新授权及公开正式冻结和四条全真实零更新预检。
