# 等图像预算支持补全诊断

状态COMPLETE，必要门=False，STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION；method_success=false。

## 两个预设G主对比

|对比|收益pp|95%区间|两项调整区间|
|---|---:|---|---|
|independent|-3.1250|[-15.625, 6.25]|[-18.75, 9.375]|
|coverage|1.0417|[-10.416666666666666, 13.541666666666668]|[-11.458333333333332, 16.666666666666668]|

## 全部三臂六条件

|臂|条件|类1组错误率%|类1图像错误率%|四类BA%|其余类宏召回%|
|---|---|---:|---:|---:|---:|
|G|BASE|58.3333|60.0000|58.6737|64.8983|
|G|WITHIN|52.0833|55.0000|59.7645|64.6860|
|G|RANDOM|55.2083|55.0000|60.6165|65.8220|
|G|NEAR|55.2083|55.0000|60.0088|65.0117|
|G|FAR|54.1667|60.0000|59.0578|65.4104|
|G|ALL|52.0833|55.0000|60.1985|65.2647|
|P|BASE|42.7083|45.0000|62.8565|65.4753|
|P|WITHIN|42.7083|45.0000|62.1345|64.5127|
|P|RANDOM|30.2083|30.0000|66.3866|65.1821|
|P|NEAR|33.3333|35.0000|64.0791|63.7722|
|P|FAR|43.7500|55.0000|60.2909|65.3878|
|P|ALL|33.3333|40.0000|63.4697|64.6263|
|C|BASE|48.9583|50.0000|62.3563|66.4751|
|C|WITHIN|48.9583|50.0000|61.3694|65.1592|
|C|RANDOM|48.9583|50.0000|61.9930|65.9906|
|C|NEAR|45.8333|45.0000|62.7483|65.3310|
|C|FAR|50.0000|60.0000|59.5950|66.1267|
|C|ALL|39.5833|45.0000|62.3619|64.8159|

## P/C描述性对比

{
  "P": {
    "independent": {
      "benefit_pp": 12.5,
      "CI95_pp": [
        0.0,
        31.25
      ],
      "family2_CI_pp": [
        0.0,
        31.25
      ]
    },
    "coverage": {
      "benefit_pp": -10.416666666666668,
      "CI95_pp": [
        -20.833333333333336,
        0.0
      ],
      "family2_CI_pp": [
        -22.916666666666668,
        0.0
      ]
    }
  },
  "C": {
    "independent": {
      "benefit_pp": 0.0,
      "CI95_pp": [
        -6.25,
        6.25
      ],
      "family2_CI_pp": [
        -9.375,
        9.375
      ]
    },
    "coverage": {
      "benefit_pp": -4.166666666666667,
      "CI95_pp": [
        -19.791666666666668,
        12.5
      ],
      "family2_CI_pp": [
        -20.833333333333336,
        14.583333333333332
      ]
    }
  }
}

## 选组及留出审计

[
  {
    "fold": 0,
    "query_rows": 675,
    "class1_query_rows": 5,
    "class1_query_groups": 4,
    "query_group_overlap": 0,
    "transforms_fit_BASE_only": true,
    "selection_outer_train_only": true,
    "conditions": {
      "BASE": {
        "class1_rows": 6,
        "class1_groups": 6
      },
      "WITHIN": {
        "class1_rows": 8,
        "class1_groups": 6
      },
      "RANDOM": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "NEAR": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "FAR": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "ALL": {
        "class1_rows": 15,
        "class1_groups": 12
      }
    },
    "coverage": {
      "base_radius": 1.2009843466321055,
      "near_radius": 1.2009843466321055,
      "far_radius": 1.0612210002421865,
      "near_ratio": 1.0,
      "far_ratio": 0.8836260049667971
    }
  },
  {
    "fold": 1,
    "query_rows": 644,
    "class1_query_rows": 4,
    "class1_query_groups": 4,
    "query_group_overlap": 0,
    "transforms_fit_BASE_only": true,
    "selection_outer_train_only": true,
    "conditions": {
      "BASE": {
        "class1_rows": 6,
        "class1_groups": 6
      },
      "WITHIN": {
        "class1_rows": 8,
        "class1_groups": 6
      },
      "RANDOM": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "NEAR": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "FAR": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "ALL": {
        "class1_rows": 16,
        "class1_groups": 12
      }
    },
    "coverage": {
      "base_radius": 1.3415527144960113,
      "near_radius": 1.2406193805729093,
      "far_radius": 0.9958487948377749,
      "near_ratio": 0.9247637958371094,
      "far_ratio": 0.7423105958321519
    }
  },
  {
    "fold": 2,
    "query_rows": 599,
    "class1_query_rows": 5,
    "class1_query_groups": 4,
    "query_group_overlap": 0,
    "transforms_fit_BASE_only": true,
    "selection_outer_train_only": true,
    "conditions": {
      "BASE": {
        "class1_rows": 6,
        "class1_groups": 6
      },
      "WITHIN": {
        "class1_rows": 8,
        "class1_groups": 6
      },
      "RANDOM": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "NEAR": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "FAR": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "ALL": {
        "class1_rows": 15,
        "class1_groups": 12
      }
    },
    "coverage": {
      "base_radius": 1.1688824878082675,
      "near_radius": 1.1688824878082675,
      "far_radius": 1.0893112154392415,
      "near_ratio": 1.0,
      "far_ratio": 0.9319253447639313
    }
  },
  {
    "fold": 3,
    "query_rows": 637,
    "class1_query_rows": 6,
    "class1_query_groups": 4,
    "query_group_overlap": 0,
    "transforms_fit_BASE_only": true,
    "selection_outer_train_only": true,
    "conditions": {
      "BASE": {
        "class1_rows": 6,
        "class1_groups": 6
      },
      "WITHIN": {
        "class1_rows": 8,
        "class1_groups": 6
      },
      "RANDOM": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "NEAR": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "FAR": {
        "class1_rows": 8,
        "class1_groups": 8
      },
      "ALL": {
        "class1_rows": 14,
        "class1_groups": 12
      }
    },
    "coverage": {
      "base_radius": 1.5497850930691386,
      "near_radius": 1.4200728552468547,
      "far_radius": 1.0824838073159486,
      "near_ratio": 0.9163030807288213,
      "far_ratio": 0.6984734929745883
    }
  }
]

## 操纵与符号一致性

{"coverage_manipulation_pass": true, "mean_near_minus_far_radius_ratio": 0.14618285950711554, "head_sign_consistency": false}

## 成本

{
  "entry_wall_seconds": 18.54530143737793,
  "cpu_seconds": 76.16921,
  "peak_RSS_bytes": 1308884992,
  "GPU_initialized": false
}
{
  "worker_exit": 0,
  "stop_reason": null,
  "wall_seconds": 24.20479679107666
}

BASE归一化/变换固定；72头先封存再query。全部选择仅outertrain，新增组事先已知类1属于离线诊断。每类质量1，独立组数效应不等于已分离所有有效样本量因素；全局距离覆盖不等于患者/临床覆盖。16已知组的条件区间不构成独立患者泛化保证。未按结果改选组、距离、seed、alpha、维度、层或pooling。ALL仅参照非性能上界。协调时间包含入口/正文，不相加。
