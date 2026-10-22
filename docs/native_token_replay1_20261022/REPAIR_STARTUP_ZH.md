# NIH token重放修复启动：真实GPU全链PASS，正式基类编码进行中

状态 **RUNNING_REPAIRED_QUALIFIED_BASE_ENCODING**，不是科学完成。原冻结科学源1f1cc1e1e603cc59e3b1c0ce37a2a3aee19c3c4f，键名/资格增量回执修复源c606d742fe1fe5b5e7650a769dadfb70d4ecf223，已有效7897代理push、远端SHA/匿名REPAIR200标题确认。原失败[PREFLIGHT_FAILURE.json](PREFLIGHT_FAILURE.json)、[REPORT_ZH.md](REPORT_ZH.md)及汇总94b3e88保持原截点不追改。

用户明确授权NIH修复并同原预算继续；不恢复HyperKvasir。最后两块只改局部state键为0/1，原权重顺序、原生层、head、目标/优化配方/数据/seed/记忆/九门不变。冻结臂与主臂总模型容量相同，可训练参数和FLOPs不同，不能混称等训练容量。

## bulk前真实资格已完成

- 修复后合成attention目标误差0，有限差分最大误差4.08207543288e-11，两行CPU前反向、NumPy目标7次/14行，0合成优化；首次失败CPU两行前反向及不完整NumPy入口另计。
- 20张合法基类图正式reader→GPU FP16前十层token→正式GPU FP32逻辑64/microbatch8两臂各2次完整forward/backward/clip/AdamW；基类内部8+2模拟到达和head/记忆分支→保存加载→FP64/独立NumPy完整两块attention通过。未读未来类图或val。
- 两臂分数最大误差均2.13162820728e-14，原完整encoder与prefix/upper拆分接口误差0。优化器执行后上层最大参数变化：固定臂0，主臂6.103515625e-05；验证实际更新，不以.grad代替。
- 每臂实际资格batch上层/头前向128行、head反向128行，主upper反向128、固定upper反向0；初始化20行、FP64/NumPy各2行，实际计数见[REPAIR_QUALIFICATION.json](REPAIR_QUALIFICATION.json)。另同2图重复fullencoder2行+prefix2行/接口upper2行，不隐藏重复计算。资格4次优化丢弃，非正式更新。
- 资格20图编码1.903554s，编码后的工程资格驻留19.833386s，包含拷贝/保存加载/独立检查，不是正式吞吐或ETA。

程序在资格PASS后才进入正式extract；截至 **2026-10-22T18:48:06.570338+08:00**，正式480/68058（当前基类480/60129）、两臂正式更新均0、val0，无分类效能或科学门判。原父子start_ticks/完整中性argv、GPU PID现场匹配，独立guard健康，未重复启动。

## 首钟、预算和后续交付

原first 2026-10-22T18:25:36.801989+08:00，原硬截止 **2026-10-23T02:25:36.801989+08:00**，从未重置/延长。旧失败、失败后等待、修复准备/数学/真实资格、正式训练和独立验收都共享原8h。恢复输出单独repair2，同一个原根和两个入口统一计费；原6GB硬限/5.4GB预停、RSS32GiB/GPUreserved8GiB，5s独立guard/提前60s TERM/30s必要KILL不变。

当前全根+入口采样340750130B、RSS采样峰20424527872B；非连续峰保证。原失败32.351302s/CPU合成2行、全部旧费用保留，早于远端first的本地准备未知不记0。恢复资格未复用旧图缓存，不保存整个train token到磁盘；当期token约18.2GB RAM在阶段后释放。

run成功后自动独立verify，重算12score/真实probe、独立memory患者复选/旧token逐项不变、冻结prefix/upper与原权重一致、主upper改变、真实seen/封存/计数；0新编码/优化。完成后确认全组退出、合并所有旧失败与重复成本，完整两臂中文结果/门判/费用proxy push、远端SHA和匿名访问确认后交付，无需用户审核。单轮科学门失败固定配方关闭，不扫参/换主臂救分。

身份/图像/token/feature/权重/逐图score/原始日志私有，不test/Etest、不future、不val调参、不新下载、不恢复静态pack/cache、不扩大清理。300val仍是重复开发筛查。唯一原有30min监测继续，健康无变化静默。
