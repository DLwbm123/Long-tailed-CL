# 出生几何的严格组外诊断

执行COMPLETE，必要几何风险门=False；method_success=false，下一判定STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION。

仅原初始永久训练支持，四折组外预测；原永久验证与测试不参与。全部变换和头从各折训练组重估，不复用历史已拟合状态。类1仍仅20行/16已知身份组，不等于新增患者独立证据。

## 六个预设主对比

|对比|类1组等权错误率增加pp|95%区间|六项调整区间|
|---|---|---|---|
|D_G|0.0000|[0.0, 0.0]|[0.0, 0.0]|
|D_N|0.0000|[0.0, 0.0]|[0.0, 0.0]|
|D_H|11.7188|[1.5625, 21.875]|[-1.5625, 26.5625]|
|D_G_minus_D_N|0.0000|[0.0, 0.0]|[0.0, 0.0]|
|D_G_minus_D_H|-11.7188|[-21.875, -1.5625]|[-26.5625, 1.5625]|
|C_minus_G_CORE|-12.5000|[-31.25, 6.25]|[-37.5, 12.5]|

## 基础强控制

|条件|类1组等权错误率%|类1图像错误率%|类1margin均值|四类BA%|
|---|---|---|---|---|
|G_FULL|52.0833|55.0000|0.064104|59.8643|
|G_CORE|52.0833|55.0000|0.061956|59.8692|
|R_CORE|39.5833|45.0000|0.084182|62.9136|
|P_CORE|39.5833|45.0000|0.129203|62.1833|
|C_CORE|39.5833|45.0000|0.092866|62.3984|

## 固定四重复全量条件

|条件|类1组错误率%|类1图像错误率%|类1margin均值|四类BA%|
|---|---|---|---|---|
|FIXED/H0/0|39.5833|45.0000|0.092866|62.3984|
|FIXED/H1/0|50.0000|60.0000|-0.029152|59.3824|
|NORM/H0/0|39.5833|45.0000|0.092274|62.1213|
|NORM/H1/0|50.0000|60.0000|-0.030123|59.3824|
|GEOM/H0/0|39.5833|45.0000|0.095844|62.1298|
|GEOM/H1/0|56.2500|65.0000|-0.031152|58.0874|
|FULL/H0/0|39.5833|45.0000|0.095604|62.1419|
|FULL/H1/0|56.2500|65.0000|-0.032162|58.3852|
|FIXED/H0/1|39.5833|45.0000|0.092866|62.3984|
|FIXED/H1/1|39.5833|45.0000|0.024911|63.8458|
|NORM/H0/1|39.5833|45.0000|0.092404|62.4105|
|NORM/H1/1|39.5833|45.0000|0.024233|63.8579|
|GEOM/H0/1|39.5833|45.0000|0.109155|62.3825|
|GEOM/H1/1|39.5833|45.0000|0.040479|63.7765|
|FULL/H0/1|39.5833|45.0000|0.110111|62.5175|
|FULL/H1/1|39.5833|45.0000|0.040526|63.8008|
|FIXED/H0/2|39.5833|45.0000|0.092866|62.3984|
|FIXED/H1/2|62.5000|70.0000|-0.015018|55.9733|
|NORM/H0/2|39.5833|45.0000|0.092470|62.2513|
|NORM/H1/2|62.5000|70.0000|-0.015163|56.1483|
|GEOM/H0/2|39.5833|45.0000|0.102327|62.4761|
|GEOM/H1/2|60.4167|65.0000|-0.005344|57.3169|
|FULL/H0/2|39.5833|45.0000|0.103125|62.5576|
|FULL/H1/2|60.4167|65.0000|-0.004260|57.2926|
|FIXED/H0/3|39.5833|45.0000|0.092866|62.3984|
|FIXED/H1/3|53.1250|60.0000|-0.032883|58.7819|
|NORM/H0/3|39.5833|45.0000|0.092936|62.2391|
|NORM/H1/3|53.1250|60.0000|-0.033160|58.7576|
|GEOM/H0/3|39.5833|45.0000|0.106510|62.2355|
|GEOM/H1/3|50.0000|55.0000|-0.022868|59.7670|
|FULL/H0/3|39.5833|45.0000|0.107104|62.2233|
|FULL/H1/3|50.0000|55.0000|-0.022601|59.8606|

## 坐标删减与随机/PCA控制

{
  "G_CORE_minus_G_FULL": {
    "error_increase_pp": 0.0,
    "CI95_pp": [
      0.0,
      0.0
    ]
  },
  "C_minus_G_FULL": {
    "error_increase_pp": -12.5,
    "CI95_pp": [
      -31.25,
      6.25
    ]
  },
  "C_minus_R_CORE": {
    "error_increase_pp": 0.0,
    "CI95_pp": [
      -12.5,
      12.5
    ]
  },
  "C_minus_P_CORE": {
    "error_increase_pp": 0.0,
    "CI95_pp": [
      0.0,
      0.0
    ]
  }
}

## 非加性交互

{"H0": 0.0, "H1": 0.0}

## 留出审计与成本

[
  {
    "fold": 0,
    "train_rows": 1880,
    "query_rows": 675,
    "train_class1_rows": 15,
    "query_class1_rows": 5,
    "train_query_group_overlap": 0,
    "all_fit_indices_from_outer_train": true,
    "normalization_geometry_and_heads_exclude_query": true
  },
  {
    "fold": 1,
    "train_rows": 1911,
    "query_rows": 644,
    "train_class1_rows": 16,
    "query_class1_rows": 4,
    "train_query_group_overlap": 0,
    "all_fit_indices_from_outer_train": true,
    "normalization_geometry_and_heads_exclude_query": true
  },
  {
    "fold": 2,
    "train_rows": 1956,
    "query_rows": 599,
    "train_class1_rows": 15,
    "query_class1_rows": 5,
    "train_query_group_overlap": 0,
    "all_fit_indices_from_outer_train": true,
    "normalization_geometry_and_heads_exclude_query": true
  },
  {
    "fold": 3,
    "train_rows": 1918,
    "query_rows": 637,
    "train_class1_rows": 14,
    "query_class1_rows": 6,
    "train_query_group_overlap": 0,
    "all_fit_indices_from_outer_train": true,
    "normalization_geometry_and_heads_exclude_query": true
  }
]

{"heads": 132, "C_geometry": 36, "P_geometry": 4}

{"body_wall_seconds": 47.854451809078455, "cpu_seconds": 192.506947, "feature_extraction_seconds": 0.0, "peak_RSS_bytes": 1489494016, "peak_GPU_bytes": 101279744}

四重复是预设扰动而非可选择训练seed；区间为给定交叉拟合模型的条件性近似，不是完整学习流程或患者总体置信区间。NORM/GEOM/FULL是操作定义的分阶段反事实，不能忽略交互或声称唯一因果分摊。未按指标改alpha、rank、fold、重复或预算。模型/身份/逐例预测/特征/图像不公开。
