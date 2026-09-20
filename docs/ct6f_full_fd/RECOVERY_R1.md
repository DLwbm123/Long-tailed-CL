# CT6-F 归档超时后的同协议恢复

2026-09-19 13:35 UTC监测发现train exit1。HK1995 Task5已完成epoch10，累计180task-epochs/2350steps；36个W已保存，17个末checkpoint已归档，第18个完整文件保留在hb。原始GPU进程墙钟6227.688秒、资源账本累计6263.875秒（含资格阶段的内部计时）保留，未归零。

失败为my-gpu至hb归档子进程180秒超时，未发现神经训练或非有限数值错误。原REQUEST/ACK/STOP、FAILURE、资源与进程回执均复制/移至R0证据，原日志保留。没有重启已完成轨迹。

修复仅存储与编排：建立新SSH master并加入连接/keepalive边界；传输期限改为600秒、worker等待ACK上限660秒，不做无限自动重试。原科学源码、种子、loss、beta10、batch48、任务长度和数据不变。单一替换归档服务；旧服务已退出。

先严格普通restore_new恢复完整HK1995 Task5末文件（SHA810a2192bb84ff2c966a2b6c2bad6c09f11dc1efb863d21654ec44aab82f16f6）。核验模型指纹、delta、原args/code/manifest、RNG/loader/synthesis；以epoch10 rolling内保存的last_features/epoch_stats/epoch_T重建末次提交A/T，与完整末状态及既有C2/C3 W一致。STRICT_RECOVERY_R1=PASS，未读取任何新图像，未重复任何optimizer step。

先将第18个checkpoint原字节重试归档，经SHA与delta可读检查PASS后才补登记谱系并清理可恢复本地副本。然后从HK1995 Task6开始；Task6教师严格为该Task5模型。继续其Task6–11及三个ISIC Task4，剩余90task-epochs/3220steps/9个新checkpoint；完成总数仍270/5570/27。Task1–5均不重复。

恢复源码及worker/archive/driver哈希独立绑定RECOVERY_LOCK_R1，旧PROTOCOL与LAUNCH锁保留。新train/evaluate/report日志/退出码及PROCESS_RECEIPTS均以_r1/R1保留，需和原R0一并计账。最终评价仍等全部W/状态锁定，test/reserved/旧未来fit/oracle访问继续为0。
