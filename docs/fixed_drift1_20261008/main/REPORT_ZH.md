# 固定 FD10：共同平移与仿射搬运完整增量对照

两臂复用同一 T1 前缀、永久 fit/meta 划分与 seed74002；唯一方法差异是历史统计搬运。
无竞争项、原型证据项、策略预热、试探分支或策略更新；各臂新增200步，复用前缀276步。
按块记录的风险和梯度仅用于诊断，不选择动作；FD始终为10。

|设置|最终BA %|平均BA %|尾类 %|遗忘 pp|新两类 %|
|---|---:|---:|---:|---:|---:|
|main_SHIFT|56.1738|70.9045|58.7328|15.5420|54.7297|
|main_AFFINE|55.5860|70.6085|60.0461|13.8669|48.5018|

仿射相对平移初筛：最终BA至少+1pp、尾类不低于−0.5pp、遗忘增加不超过1pp、新类不低于−1pp。
本波结束后停止并分析；不因通过门槛自动增加种子、调度校准或RL训练。
官方val是反复使用的开发集，单种子与共享前缀不构成独立确认；test保持封存。
仿射矩公式对给定映射精确，当前类到旧类的漂移外推仍是假设。
FDRL3失败结论保持不变，本波由用户另行授权检查固定底座，不是恢复原RL主筛。

```json
{
  "cycle": "FIXED_DRIFT1",
  "primary_candidate": "AFFINE",
  "failures": [],
  "independent_confirmation": false,
  "test_accessed": false,
  "reference": "main_SHIFT",
  "wave": "main",
  "main_screen": {
    "main_AFFINE": {
      "ba": -0.00587831361625113,
      "tail": 0.013133208255159401,
      "forgetting": -0.016750967665601857,
      "passed": false,
      "new_recall": -0.062279670975323276
    }
  },
  "paired_predictions": [
    {
      "stage": 1,
      "a": "main_SHIFT",
      "b": "main_AFFINE",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 2,
      "a": "main_SHIFT",
      "b": "main_AFFINE",
      "n": 149,
      "prediction_agreement": 0.9798657718120806,
      "both_correct": 102,
      "a_only_correct": 2,
      "b_only_correct": 1,
      "both_wrong": 44
    },
    {
      "stage": 3,
      "a": "main_SHIFT",
      "b": "main_AFFINE",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 131,
      "a_only_correct": 2,
      "b_only_correct": 2,
      "both_wrong": 77
    },
    {
      "stage": 4,
      "a": "main_SHIFT",
      "b": "main_AFFINE",
      "n": 295,
      "prediction_agreement": 0.9322033898305084,
      "both_correct": 156,
      "a_only_correct": 8,
      "b_only_correct": 5,
      "both_wrong": 126
    }
  ],
  "screen_success": false
}
```
