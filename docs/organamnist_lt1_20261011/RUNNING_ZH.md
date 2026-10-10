# OrganAMNIST-LT首轮已后台启动

已冻结源452af31b5d94ac485144e19ab7279d347bed808e，官方协议匿名访问HTTP200。北京时间2026-10-11 01:30:30启动，24h截止2026-10-12 01:30:30；当前PREPARING_DATA/DOWNLOAD，两个中性CPU父子进程存活、无立即错误。仅GPU0/1授权，准备结束且每卡≥32768MiB后2个真实零更新预检→2新配对train（各212，共424更新）→统一官方val评价。

启动前GPU0/1各39268MiB、实际NAS nfs4及小型写读探针通过，不干预其他任务。test始终封存；未声称预检已PASS、正式GPU训练已开始或方法有效。CPU检查及首次总和fixture失败修正见CPU_CHECK与CPU_CHECK_attempt1，保留工程失败。完整参数和采样见PROTOCOL_LOCK，当前不能交付科学指标。
