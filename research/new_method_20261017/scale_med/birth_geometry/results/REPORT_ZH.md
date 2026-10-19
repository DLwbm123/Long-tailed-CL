# 同核心出生几何诊断：失败与停止报告

源码与协议在拟合前以 b271e83 封存。唯一运行在最终保存阶段失败（worker exit=1）；全部拟合已完成，但科学主门未决，不能标为通过或否证。没有重跑或追加预算。

## 完成覆盖

- 2555条初始训练支持的四折query覆盖，类1共20行、16已知身份组。原永久验证与测试不参与拟合或评分。
- 132次唯一头求解、36次C几何与4次普通PCA估计、128个唯一扰动单元均有计数；128个已保存类1图像错误率有限。
- 四折query行数675/644/599/637，类1为5/4/5/6。拟合路径仅接收各折训练数组，组不交叠断言已通过。
- 20个基础控制评分曾在内存计算，但未保存；不能声称其结果覆盖已公开验证。

## 已保存的辅助结果

以下为20行类1的图像等权错误率，四次扰动平均。它不是协议的组等权主指标，也没有可恢复的配对逐组区间。

|映射|H0原支持|H1独立支持扰动|
|---|---:|---:|
|FIXED|45.00%|58.75%|
|NORM|45.00%|58.75%|
|GEOM|45.00%|57.50%|
|FULL|45.00%|57.50%|

## 失败与证据缺口

访问守卫按文件名拦截历史预测访问，同时误拦了本轮私有预测的写入。错误发生在主对比与区间已于内存计算之后、aggregate持久化之前；进程退出后这些内存结果不可恢复。未保存的结果不作推断。旧status仍显示RUNNING，最终EXIT=1及协调退出回执优先，主子进程均已退休。

类1组等权风险、六项调整区间、C−G_CORE/全局截断/随机/PCA强控制、交互项与完整最终峰值成本均未持久化。必要主门不可评估；辅助结果不替代主门。

## 成本

```json
{
  "coordinator_wall_seconds": 265.21753764152527,
  "worker_entry_wall_seconds": 254.23899173736572,
  "last_checkpoint_body_seconds": 253.66125422157347,
  "remote_directory_bytes": 47387894,
  "worker_cpu_seconds": null,
  "peak_RSS_bytes": null,
  "peak_GPU_bytes": null,
  "note": "Times are nested, not additive. Final CPU/RSS/GPU receipt unavailable; last checkpoint passed runtime guards. No budget expansion or retry."
}
```

900秒墙钟未超限，目录约45.19MiB，小于1GiB；最终CPU、RSS、GPU峰值缺失，不报告精确数值或完整预算认证。最后一个checkpoint通过资源守卫，非最终完整成本回执。协调、入口、正文时间具有包含关系，不相加。

停止状态：STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION。禁止据辅助结果调参、补种子、重跑、追加几何或新头。工程失败不算科学负结果；本轮未解决的机制保持未识别。
