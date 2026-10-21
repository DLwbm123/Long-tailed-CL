# 人工授权的原精度F1统计重建（corrected_run）

用户明确授权按原Task1精度重建后来类统计并完成验收/push；不重训Task1、不调参、不放宽门，全部失败及重复成本保留。

仅纠正已取证的运行设置：matmul TF32=false、cuDNN TF32=true、cuDNN benchmark=false/deterministic=false、float32 matmul highest，与原Task1入口一致。保持Task1 checkpoint/head/bank、seed74002、六阶段/类顺序、数据/变换、1536整体L2特征、组件/类权重、R/static_pc竞争、所有正则/阈值、原CPU读出、原继续门不变。使用原q107v解释器与原APART依赖，记录有效运行flags及库版本。

新输出corrected_run与旧run并存；旧失败/诊断bank/头/query/回执/报告不覆盖，不复制旧bank作为新类统计。严格恢复原Task1/1504更新后，重新资格32图×2=64，再按原阶段到达重编码后来类7929张、重建原统计与后五头；新增神经更新0。旧类不回放，Task1不重训，不读取未来模型。全部后五头封存及encoder/旧bank不变检查后才打开300val一次。此轮新增预计train7993/val300，作为重复成本显式加账，不抵销旧7993train/728val；继承Task1更新不重复计两次新训练。

验收阈值完整沿用PLAN：source head relative<=2e-6/PCG<=1e-8、资格双前向exact、Task1十类正确计数全部精确一致（Fibrosis6/15且52/150）、六头独立NumPy/显式class差CG重解relative W<=2e-5/分数<=2e-5、覆盖/患者隔离/全部计数正确。新增独立核对有效precision flags与原历史Task1计数，不能只相信worker的parity布尔。科学门不变：final>=74、mean>=28.319223985890655%、tail4>=15；额外final>=77或mean>=29.319223985890655%才标改善开发证据。通过数值/复现验收不代表通过科学继续门。

共原首次1792552424.3168447/截止1792581224.3168447（8h），不启新时钟；全原根含corrected/old/失败/临时/源及中性入口，max(st_size,st_blocks*512)硬3GB/2.7GB预停，组RSS8GiB、GPU0准入16GiB/reserved12GiB。旧退出约.767GB，新final bank约.754GB分配，原子替换保守合并峰<2.4GB；实际guard5秒检查全根/进程组/提前60秒TERM和必要KILL，跨旧/新/verify峰值保持。无新pack/逐图训练缓存，不删任何旧证据腾空间。

失败保持独立失败回执，按原钟定位，不改设置救分；若精确parity仍失败不宣布完成。实际成功或失败均交付匿名聚合、完整失败和重复成本、中文报告，经有效代理push/远端SHA/匿名访问核对。仅新corrected结果可标验收结论，旧run始终未获科学验收。私有原图/身份/模型/bank/逐图特征分数/日志继续保留。
