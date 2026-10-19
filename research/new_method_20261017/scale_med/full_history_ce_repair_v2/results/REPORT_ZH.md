# 完整历史CE工程修复后独立运行结果

同原冻结G/P/C表示、原永久训练与验证划分；CE是特权完整历史参照，不满足退休部署约束，不是BA上界或新方法。原内部测试已研究暴露。

退出回执：worker=0；coordinator={'worker_exit': 0, 'stop_reason': None, 'wall_seconds': 708.1973890559748, 'worker_CPU_seconds': 2210.828125, 'worker_peak_RSS_bytes': 2180558848, 'coordinator_CPU_seconds': 203.39053, 'worker_usage_scope': 'entire_child_including_imports', 'source_commit': '269999e46cbb88f9cc8b2c143673454eb7341e29'}。

科学判断：`UNRESOLVED_OPTIMIZATION_NO_EXTENSION`；完整收敛臂：[]；通过臂：[]。

|臂|初始选择lambda|初始BA%|+类4 BA%|+类5 BA%|平均J%|类4出生%|类4转旧%|类5出生%|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G_ridge | 0.01 | 58.1453 | 51.0186 | 55.0698 | 53.0442 | 57.0000 | 54.0000 | 75.0000 |
| G_CE | 0.0001 | 57.8488 | 52.6512 | 未决 | 未决 | 76.0000 | 未决 | 未决 |
| P_ridge | 0.01 | 58.3953 | 52.6837 | 53.8217 | 53.2527 | 55.0000 | 50.0000 | 72.0000 |
| P_CE | 未决 | 未决 | 未决 | 未决 | 未决 | 未决 | 未决 | 未决 |
| C_ridge | 0.01 | 58.1453 | 51.8186 | 55.1550 | 53.4868 | 62.0000 | 57.0000 | 74.0000 |
| C_CE | 0.0001 | 59.1802 | 未决 | 未决 | 未决 | 未决 | 未决 | 未决 |

|CE臂减ridge控制|J差pp|95%区间|9项调整区间|
|---|---:|---|---|

## 优化认证

|臂|阶段|lambda|复用|状态|停止分支|原坐标梯度max|目标|closure次数|wall秒|
|---|---:|---:|---|---|---|---:|---:|---:|---:|
|G|0|0.0001|False|CONVERGED|if abs(loss - prev_loss) < tolerance_change:|3.465e-08|0.32724453|570|4.98|
|G|0|0.01|True|CONVERGED|OPTIMIZER_STOP|5.871e-08|1.04810509|30|0.05|
|G|0|1.0|True|CONVERGED|OPTIMIZER_STOP|2.897e-09|1.37841669|12|0.04|
|P|0|0.0001|False|UNRESOLVED|HEAD_BUDGET_LIMIT|3.905e-08|0.33588996|3000|24.98|
|P|0|0.01|True|CONVERGED|OPTIMIZER_STOP|6.247e-08|1.03105364|35|0.04|
|P|0|1.0|True|CONVERGED|OPTIMIZER_STOP|2.262e-08|1.37626324|10|0.02|
|C|0|0.0001|True|CONVERGED|OPTIMIZER_STOP|2.409e-08|0.30500828|316|0.03|
|C|0|0.01|True|CONVERGED|OPTIMIZER_STOP|2.807e-08|1.06650270|22|0.03|
|C|0|1.0|True|CONVERGED|OPTIMIZER_STOP|3.681e-08|1.37974686|11|0.03|
|G|1|0.0001|False|CONVERGED|if gtd > -tolerance_change:|8.660e-08|0.54210613|453|5.23|
|C|1|0.0001|False|UNRESOLVED|if abs(loss - prev_loss) < tolerance_change:|1.383e-07|0.51239537|1268|10.56|
|G|2|0.0001|False|UNRESOLVED|if n_iter == max_iter:|3.321e-07|0.51270942|2620|25.00|
|C|2|0.0001|False|UNRESOLVED|if gtd > -tolerance_change:|2.114e-07|0.48597035|1277|11.94|

复用行的优化次数和原认证梯度继承旧回执，wall为本轮复核时间；不算本轮重新拟合。主J比较缺失时不生成区间，不将缺失当零收益。

## 逐类风险

|臂|阶段|图像召回%按类顺序|组等权BA%|
|---|---:|---|---:|
|G_ridge|0|[73.0, 45.0, 39.0, 75.5814]|59.5453|
|G_ridge|1|[54.0, 45.0, 27.0, 72.093, 57.0]|52.0206|
|G_ridge|2|[55.0, 43.0, 29.0, 74.4186, 54.0, 75.0]|55.8027|
|G_CE|0|[63.0, 34.0, 53.0, 81.3953]|58.5922|
|G_CE|1|[42.0, 32.0, 40.0, 73.2558, 76.0]|53.4086|
|P_ridge|0|[75.0, 42.0, 41.0, 75.5814]|60.6714|
|P_ridge|1|[61.0, 41.0, 32.0, 74.4186, 55.0]|54.0601|
|P_ridge|2|[60.0, 40.0, 30.0, 70.9302, 50.0, 72.0]|54.7276|
|C_ridge|0|[75.0, 37.0, 45.0, 75.5814]|59.8432|
|C_ridge|1|[57.0, 36.0, 32.0, 72.093, 62.0]|53.1564|
|C_ridge|2|[59.0, 39.0, 31.0, 70.9302, 57.0, 74.0]|55.9450|
|C_CE|0|[66.0, 28.0, 59.0, 83.7209]|60.3124|

## 预设门


## 成本和访问边界

```json
{
  "wall_seconds": 695.8462439943105,
  "cpu_seconds": 2207.173163,
  "peak_RSS_bytes": 2180558848,
  "peak_GPU_allocated_bytes": 739425280,
  "GPU_residence_upper_seconds": 656.336737726815,
  "feature_extraction": [
    {
      "tag": "stage0",
      "images": 3236,
      "wall_seconds": 292.4016395965591
    },
    {
      "tag": "stage1",
      "images": 2254,
      "wall_seconds": 213.24744452536106
    },
    {
      "tag": "stage2",
      "images": 36,
      "wall_seconds": 2.489149544388056
    },
    {
      "tag": "test",
      "images": 586,
      "wall_seconds": 54.24812151212245
    }
  ],
  "output_bytes": 111787469,
  "privileged_old_training_access": true,
  "full_history_memory_cap_compliant": false
}
```

协调墙钟包含worker正文，不相加。GPU驻留为保守阶段墙钟上界，不是精确硬件计费。原global缓存成本未重新发生；旧训练逐图特征在本轮被特权离线复用，不能称可部署无回放。

所有训练统计及原ridge输出parity见aggregate；不发布原图、身份、逐例分数、头或私有路径。
