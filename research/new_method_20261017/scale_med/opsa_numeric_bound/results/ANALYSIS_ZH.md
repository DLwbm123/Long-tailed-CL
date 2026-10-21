# OPSA：训练完成，query原F回放未通过

结论：UNRESOLVED_OPSA_QUERY_F_PARITY。scientific_complete=false、method_success=false。尚无可解释的完整性能结果，不是OPSA科学负结论。

三个producer的支持32图原回放全过后，H/L/N/M各800步，共9600次backward和optimizer更新全部完成。12个模型已保存、逐字段回读通过，原骨干版本保持，支持分数回读最大误差8.43769498715119e-15。MODEL_SEAL先于query图像读取。MAPC真实触发，三producer累计重置元素次数分别3136662、3081097、3105316；这些是重复累计操作次数，不是独立参数数目或效能证据。

失败发生在第一个ISIC0 query fold：五臂分数已由query_scores返回，但原F分数相对旧F_out的maxabs未通过1e-5门（entry.py第243行）。具体err只在内存中，未在断言前保存，因此精确误差NA；后续预测一致性、两fold/HK完整覆盖与全部风险门均未到达，不能称为通过。无最终aggregate或持久化responses。已读query图，不能再称本轮无query访问。

计数120776逻辑encoder图次、7377次wrapper调用，包含双流；9600更新，0ridge solve/mapper fit。按固定控制流，训练与支持检查115872图次，ISIC0 query1226图×F/L/N/M四次=4904图次，H复用F，不另做encoder。这是源码与计数闭合，不是新增逐例读取审计。

当前只确认评估回放断言失败，未定位唯一根因。小支持批通过不保证整个query或尾批一致；batch组成、尾批、输入/数值上下文等仍需区分，不能直接断言是哪一个。保留12个已训练模型和失败源，不自动重训、补跑或放宽容差。后续须独立封存工程诊断，避免依据已学习臂性能挑执行条件。

本次续跑自启动至退出6580.161662817001秒，保留原始实验时钟；未触发时间/CPU/RSS/空间守卫。父1115064、子1115068退出并退休。新增round5832544字节，加此前链399971及32MiB余量仍低于512MiB；窗口加outside与报告余量低于17GB。原错误记录、所有历史停止门保留。

源码bdc4f0f099405593f9d416ff1ff704b46f271927预先封存。公开训练轨迹、模型检查聚合、失败位置与成本；模型、身份、原图、特征及私有日志留在服务器。

成本一次计入：
{
  "activity_wall_seconds": 6580.161662817001,
  "known_worker_wait4_CPU_seconds": 8965.272127999999,
  "coordinator_CPU_seconds": 510.811744,
  "peak_RSS_bytes": 1750630400,
  "round_remote_bytes": 5832544,
  "window_remote_bytes": 14175830065,
  "unmetered": "Preparation/reading/network/report CPU unknown; worker body not double-counted"
}

累计：
{
  "activity_wall_seconds": 38957.06763011296,
  "known_worker_wait4_CPU_seconds": 49710.26815499998,
  "coordinator_CPU_seconds": 1784.604758
}
