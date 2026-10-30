# R87 保存端点动作效果转移审查

本次仅复用已封存的 Qa/Qb outcome tables；不拟合、不运行新 forward/optimizer、不读 VAL/TEST，不改变原科学门。全部6 stage×2 fold 保留；class1 fold1 的覆盖不足不补。比较由无标签动作共同存在确定的 stop、单动作及去掉顺序重复的合法双动作端点，不选择最佳动作。

```json
{
  "utility": {
    "defined_strata": 12,
    "positive_tau_strata": 12,
    "tau_min": 0.5521115320008979,
    "tau_max": 0.7371093904803245,
    "tau_mean": 0.6386616579467047,
    "endpoint_instances": 28876,
    "Qa_positive_to_Qb": {
      "counts": {
        "-1": 950,
        "0": 33,
        "1": 1100
      },
      "denominator": 2083,
      "positive_fraction": 0.528084493518963
    },
    "Qb_positive_to_Qa": {
      "counts": {
        "-1": 1289,
        "0": 65,
        "1": 1100
      },
      "denominator": 2454,
      "positive_fraction": 0.4482477587612062
    }
  },
  "macro": {
    "defined_strata": 12,
    "positive_tau_strata": 12,
    "tau_min": 0.5090795485616741,
    "tau_max": 0.7003908219570529,
    "tau_mean": 0.6089628783071587,
    "endpoint_instances": 28876,
    "Qa_positive_to_Qb": {
      "counts": {
        "-1": 1435,
        "0": 34,
        "1": 1626
      },
      "denominator": 3095,
      "positive_fraction": 0.5253634894991922
    },
    "Qb_positive_to_Qa": {
      "counts": {
        "-1": 1849,
        "0": 72,
        "1": 1626
      },
      "denominator": 3547,
      "positive_fraction": 0.45841556244713844
    }
  }
}
```

12层 protected utility 的 Kendall tau-b 均正，表明整体收益排序具有部分可转移信号；有益端点的跨片正向保持率明显不等于1，因此全局排序正相关不足以保证可靠识别少数有益修改。端点共享动作且跨stage复用样本，上述合计只是相关端点实例的描述，不是独立病例、统计显著性或GRPO效益。不能据此唯一归因策略失败，也不触发KL/LR/horizon/强制动作搜索；R87总门仍失败。

审查成本：CPU 6.914659225秒，核心wall 5.976795673370361秒，峰值RSS 438095872B。依赖导入、准备、文档与交付CPU未独立采样，不能计作0。原R87时钟未重置；没有新FIRST。
