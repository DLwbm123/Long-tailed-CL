# 原生隐藏零空间诊断

状态COMPLETE；必要门=False；method_success=false；STOP_NO_NULL_SPECIFIC_EVIDENCE。

|臂|类1组错误率%|类1图像错误率%|四类BA%|
|---|---:|---:|---:|
|G_PAD|52.0833|55.0000|59.6141|
|RANGE|33.3333|40.0000|63.8197|
|NATIVE|34.3750|45.0000|62.8273|
|NULL|34.3750|45.0000|62.3497|
|UNIFORM_NULL|27.0833|30.0000|66.2054|
|CLS|30.2083|35.0000|65.5085|

## 三个预设主对比：控制风险减NULL风险

|控制|NULL收益pp|95%区间|三项调整区间|
|---|---:|---|---|
|G_PAD|17.7083|[2.083333333333333, 34.375]|[-2.0833333333333344, 37.5]|
|RANGE|-1.0417|[-9.375, 6.25]|[-12.5, 9.375]|
|NATIVE|0.0000|[0.0, 0.0]|[0.0, 0.0]|

## 数值投影检查

```json
{
  "W2_gram_eigen_ratio": 0.00026930916975353183,
  "null_relative_error": 6.09967685256433e-16,
  "null_range_cross": 7.546047120499111e-16
}
```

## 留出审计

```json
[
  {
    "fold": 0,
    "train_rows": 1880,
    "query_rows": 675,
    "class1_query_rows": 5,
    "overlap": 0,
    "normalization_and_head_train_only": true
  },
  {
    "fold": 1,
    "train_rows": 1911,
    "query_rows": 644,
    "class1_query_rows": 4,
    "overlap": 0,
    "normalization_and_head_train_only": true
  },
  {
    "fold": 2,
    "train_rows": 1956,
    "query_rows": 599,
    "class1_query_rows": 5,
    "overlap": 0,
    "normalization_and_head_train_only": true
  },
  {
    "fold": 3,
    "train_rows": 1918,
    "query_rows": 637,
    "class1_query_rows": 6,
    "overlap": 0,
    "normalization_and_head_train_only": true
  }
]
```

## 正文与提取成本

```json
{
  "body_wall_seconds": 184.90965345222503,
  "feature_extraction_seconds": 166.94965877570212
}
```

## 最终资源回执

```json
{
  "entry_wall_seconds": 184.93446707725525,
  "cpu_seconds": 243.578882,
  "peak_RSS_bytes": 2119155712,
  "peak_GPU_bytes": 729773568
}
```

## 协调回执

```json
{
  "worker_exit": 0,
  "stop_reason": null,
  "wall_seconds": 194.8674590587616
}
```

每臂保守状态上界18561200bytes；固定1318+96，不挑层/维度/seed/alpha。

本轮为InnerCIL启发机制诊断，不是新方法；旧永久验证/未来类/测试不拟合或评分。特权离线原训练支持访问不等于可部署退休实现。20行16已知组、未知患者来源；区间条件于交叉拟合模型，非新独立患者泛化认证。协调/入口/正文时间包含，不相加。所有新增投影/提取/缓存/求解均计入本轮，旧成本另保留。
