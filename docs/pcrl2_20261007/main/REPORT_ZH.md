# 原型竞争预算 RL 验证

七动作小型组相对策略；相同预算 GREEDY 控制。旧类风险仅为共同平移二阶统计代理。
每个决策四个采样分支加一个 PC 参考，各进行最多两步真实适配器更新；全部计入总成本。
主路径采用更新后策略最大概率动作；GREEDY 才在采样分支与参考中选择最高奖励。
PCRL2：RESPONSE0 使用固定解析预测；RL_RESPONSE 加共享可学习残差；RL_SHARED 去掉固定评分；FIXED1 始终动作1。
主候选、复用对照和追加波以本周期 PROTOCOL_LOCK.json 为准。

|设置|最终 BA %|平均 BA %|尾类 %|遗忘 pp|最后新类 %|
|---|---:|---:|---:|---:|---:|
|main_R|56.8150|71.7165|58.2970|16.7473|55.5523|
|main_PC|58.1450|71.8622|56.8575|18.8339|61.2515|
|main_GREEDY|57.5259|71.9734|57.8191|16.5641|54.4653|
|main_RL|57.7976|71.8638|57.8191|15.8538|55.5523|
|main_FIXED1|57.7976|71.8638|57.8191|15.8538|55.5523|
|main_RESPONSE0|57.7976|72.0414|57.8191|16.5641|55.5523|
|main_RL_SHARED|57.7976|71.8638|57.8191|15.8538|55.5523|
|main_RL_RESPONSE|57.4851|71.7857|57.8191|16.2704|55.5523|

## 判定

```json
{
  "failures": [],
  "independent_confirmation": false,
  "test_accessed": false,
  "primary_candidate": "RL_RESPONSE",
  "reference": "main_R",
  "wave": "main",
  "main_screen": {
    "main_PC": {
      "ba": 0.013300666581419596,
      "tail": -0.01439455402870038,
      "forgetting": 0.02086629312239069,
      "passed": false,
      "new_recall": 0.056991774383078786
    },
    "main_GREEDY": {
      "ba": 0.007108933079903346,
      "tail": -0.004779169413315798,
      "forgetting": -0.001831842075744522,
      "passed": false,
      "new_recall": -0.010869565217391242
    },
    "main_RL": {
      "ba": 0.00982632438425124,
      "tail": -0.004779169413315798,
      "forgetting": -0.008935099179001621,
      "passed": false,
      "new_recall": 0.0
    },
    "main_FIXED1": {
      "ba": 0.00982632438425124,
      "tail": -0.004779169413315798,
      "forgetting": -0.008935099179001621,
      "passed": false,
      "new_recall": 0.0
    },
    "main_RESPONSE0": {
      "ba": 0.00982632438425124,
      "tail": -0.004779169413315798,
      "forgetting": -0.001831842075744522,
      "passed": false,
      "new_recall": 0.0
    },
    "main_RL_SHARED": {
      "ba": 0.00982632438425124,
      "tail": -0.004779169413315798,
      "forgetting": -0.008935099179001621,
      "passed": false,
      "new_recall": 0.0
    },
    "main_RL_RESPONSE": {
      "ba": 0.006701324384251306,
      "tail": -0.004779169413315798,
      "forgetting": -0.004768432512334914,
      "passed": false,
      "new_recall": 0.0
    }
  },
  "paired_predictions": [
    {
      "stage": 1,
      "a": "main_R",
      "b": "main_PC",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_R",
      "b": "main_GREEDY",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_R",
      "b": "main_RL",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_R",
      "b": "main_FIXED1",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_R",
      "b": "main_RESPONSE0",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_R",
      "b": "main_RL_SHARED",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_R",
      "b": "main_RL_RESPONSE",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_PC",
      "b": "main_GREEDY",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_PC",
      "b": "main_RL",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_PC",
      "b": "main_FIXED1",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_PC",
      "b": "main_RESPONSE0",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_PC",
      "b": "main_RL_SHARED",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_PC",
      "b": "main_RL_RESPONSE",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_GREEDY",
      "b": "main_RL",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_GREEDY",
      "b": "main_FIXED1",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_GREEDY",
      "b": "main_RESPONSE0",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_GREEDY",
      "b": "main_RL_SHARED",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_GREEDY",
      "b": "main_RL_RESPONSE",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_RL",
      "b": "main_FIXED1",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_RL",
      "b": "main_RESPONSE0",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_RL",
      "b": "main_RL_SHARED",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_RL",
      "b": "main_RL_RESPONSE",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_FIXED1",
      "b": "main_RESPONSE0",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_FIXED1",
      "b": "main_RL_SHARED",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_FIXED1",
      "b": "main_RL_RESPONSE",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_RESPONSE0",
      "b": "main_RL_SHARED",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_RESPONSE0",
      "b": "main_RL_RESPONSE",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 1,
      "a": "main_RL_SHARED",
      "b": "main_RL_RESPONSE",
      "n": 85,
      "prediction_agreement": 1.0,
      "both_correct": 83,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 2
    },
    {
      "stage": 2,
      "a": "main_R",
      "b": "main_PC",
      "n": 149,
      "prediction_agreement": 0.9463087248322147,
      "both_correct": 101,
      "a_only_correct": 5,
      "b_only_correct": 3,
      "both_wrong": 40
    },
    {
      "stage": 2,
      "a": "main_R",
      "b": "main_GREEDY",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 103,
      "a_only_correct": 3,
      "b_only_correct": 3,
      "both_wrong": 40
    },
    {
      "stage": 2,
      "a": "main_R",
      "b": "main_RL",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 103,
      "a_only_correct": 3,
      "b_only_correct": 3,
      "both_wrong": 40
    },
    {
      "stage": 2,
      "a": "main_R",
      "b": "main_FIXED1",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 103,
      "a_only_correct": 3,
      "b_only_correct": 3,
      "both_wrong": 40
    },
    {
      "stage": 2,
      "a": "main_R",
      "b": "main_RESPONSE0",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 103,
      "a_only_correct": 3,
      "b_only_correct": 3,
      "both_wrong": 40
    },
    {
      "stage": 2,
      "a": "main_R",
      "b": "main_RL_SHARED",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 103,
      "a_only_correct": 3,
      "b_only_correct": 3,
      "both_wrong": 40
    },
    {
      "stage": 2,
      "a": "main_R",
      "b": "main_RL_RESPONSE",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 103,
      "a_only_correct": 3,
      "b_only_correct": 3,
      "both_wrong": 40
    },
    {
      "stage": 2,
      "a": "main_PC",
      "b": "main_GREEDY",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 102,
      "a_only_correct": 2,
      "b_only_correct": 4,
      "both_wrong": 41
    },
    {
      "stage": 2,
      "a": "main_PC",
      "b": "main_RL",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 102,
      "a_only_correct": 2,
      "b_only_correct": 4,
      "both_wrong": 41
    },
    {
      "stage": 2,
      "a": "main_PC",
      "b": "main_FIXED1",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 102,
      "a_only_correct": 2,
      "b_only_correct": 4,
      "both_wrong": 41
    },
    {
      "stage": 2,
      "a": "main_PC",
      "b": "main_RESPONSE0",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 102,
      "a_only_correct": 2,
      "b_only_correct": 4,
      "both_wrong": 41
    },
    {
      "stage": 2,
      "a": "main_PC",
      "b": "main_RL_SHARED",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 102,
      "a_only_correct": 2,
      "b_only_correct": 4,
      "both_wrong": 41
    },
    {
      "stage": 2,
      "a": "main_PC",
      "b": "main_RL_RESPONSE",
      "n": 149,
      "prediction_agreement": 0.959731543624161,
      "both_correct": 102,
      "a_only_correct": 2,
      "b_only_correct": 4,
      "both_wrong": 41
    },
    {
      "stage": 2,
      "a": "main_GREEDY",
      "b": "main_RL",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_GREEDY",
      "b": "main_FIXED1",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_GREEDY",
      "b": "main_RESPONSE0",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_GREEDY",
      "b": "main_RL_SHARED",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_GREEDY",
      "b": "main_RL_RESPONSE",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_RL",
      "b": "main_FIXED1",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_RL",
      "b": "main_RESPONSE0",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_RL",
      "b": "main_RL_SHARED",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_RL",
      "b": "main_RL_RESPONSE",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_FIXED1",
      "b": "main_RESPONSE0",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_FIXED1",
      "b": "main_RL_SHARED",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_FIXED1",
      "b": "main_RL_RESPONSE",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_RESPONSE0",
      "b": "main_RL_SHARED",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_RESPONSE0",
      "b": "main_RL_RESPONSE",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 2,
      "a": "main_RL_SHARED",
      "b": "main_RL_RESPONSE",
      "n": 149,
      "prediction_agreement": 1.0,
      "both_correct": 106,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 43
    },
    {
      "stage": 3,
      "a": "main_R",
      "b": "main_PC",
      "n": 212,
      "prediction_agreement": 0.9150943396226415,
      "both_correct": 128,
      "a_only_correct": 6,
      "b_only_correct": 8,
      "both_wrong": 70
    },
    {
      "stage": 3,
      "a": "main_R",
      "b": "main_GREEDY",
      "n": 212,
      "prediction_agreement": 0.9150943396226415,
      "both_correct": 128,
      "a_only_correct": 6,
      "b_only_correct": 8,
      "both_wrong": 70
    },
    {
      "stage": 3,
      "a": "main_R",
      "b": "main_RL",
      "n": 212,
      "prediction_agreement": 0.9198113207547169,
      "both_correct": 128,
      "a_only_correct": 6,
      "b_only_correct": 6,
      "both_wrong": 72
    },
    {
      "stage": 3,
      "a": "main_R",
      "b": "main_FIXED1",
      "n": 212,
      "prediction_agreement": 0.9198113207547169,
      "both_correct": 128,
      "a_only_correct": 6,
      "b_only_correct": 6,
      "both_wrong": 72
    },
    {
      "stage": 3,
      "a": "main_R",
      "b": "main_RESPONSE0",
      "n": 212,
      "prediction_agreement": 0.9150943396226415,
      "both_correct": 128,
      "a_only_correct": 6,
      "b_only_correct": 8,
      "both_wrong": 70
    },
    {
      "stage": 3,
      "a": "main_R",
      "b": "main_RL_SHARED",
      "n": 212,
      "prediction_agreement": 0.9198113207547169,
      "both_correct": 128,
      "a_only_correct": 6,
      "b_only_correct": 6,
      "both_wrong": 72
    },
    {
      "stage": 3,
      "a": "main_R",
      "b": "main_RL_RESPONSE",
      "n": 212,
      "prediction_agreement": 0.9245283018867925,
      "both_correct": 128,
      "a_only_correct": 6,
      "b_only_correct": 6,
      "both_wrong": 72
    },
    {
      "stage": 3,
      "a": "main_PC",
      "b": "main_GREEDY",
      "n": 212,
      "prediction_agreement": 1.0,
      "both_correct": 136,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 76
    },
    {
      "stage": 3,
      "a": "main_PC",
      "b": "main_RL",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_PC",
      "b": "main_FIXED1",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_PC",
      "b": "main_RESPONSE0",
      "n": 212,
      "prediction_agreement": 1.0,
      "both_correct": 136,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 76
    },
    {
      "stage": 3,
      "a": "main_PC",
      "b": "main_RL_SHARED",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_PC",
      "b": "main_RL_RESPONSE",
      "n": 212,
      "prediction_agreement": 0.9811320754716981,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_GREEDY",
      "b": "main_RL",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_GREEDY",
      "b": "main_FIXED1",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_GREEDY",
      "b": "main_RESPONSE0",
      "n": 212,
      "prediction_agreement": 1.0,
      "both_correct": 136,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 76
    },
    {
      "stage": 3,
      "a": "main_GREEDY",
      "b": "main_RL_SHARED",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_GREEDY",
      "b": "main_RL_RESPONSE",
      "n": 212,
      "prediction_agreement": 0.9811320754716981,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_RL",
      "b": "main_FIXED1",
      "n": 212,
      "prediction_agreement": 1.0,
      "both_correct": 134,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 78
    },
    {
      "stage": 3,
      "a": "main_RL",
      "b": "main_RESPONSE0",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 133,
      "a_only_correct": 1,
      "b_only_correct": 3,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_RL",
      "b": "main_RL_SHARED",
      "n": 212,
      "prediction_agreement": 1.0,
      "both_correct": 134,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 78
    },
    {
      "stage": 3,
      "a": "main_RL",
      "b": "main_RL_RESPONSE",
      "n": 212,
      "prediction_agreement": 0.9952830188679245,
      "both_correct": 134,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 78
    },
    {
      "stage": 3,
      "a": "main_FIXED1",
      "b": "main_RESPONSE0",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 133,
      "a_only_correct": 1,
      "b_only_correct": 3,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_FIXED1",
      "b": "main_RL_SHARED",
      "n": 212,
      "prediction_agreement": 1.0,
      "both_correct": 134,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 78
    },
    {
      "stage": 3,
      "a": "main_FIXED1",
      "b": "main_RL_RESPONSE",
      "n": 212,
      "prediction_agreement": 0.9952830188679245,
      "both_correct": 134,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 78
    },
    {
      "stage": 3,
      "a": "main_RESPONSE0",
      "b": "main_RL_SHARED",
      "n": 212,
      "prediction_agreement": 0.9764150943396226,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_RESPONSE0",
      "b": "main_RL_RESPONSE",
      "n": 212,
      "prediction_agreement": 0.9811320754716981,
      "both_correct": 133,
      "a_only_correct": 3,
      "b_only_correct": 1,
      "both_wrong": 75
    },
    {
      "stage": 3,
      "a": "main_RL_SHARED",
      "b": "main_RL_RESPONSE",
      "n": 212,
      "prediction_agreement": 0.9952830188679245,
      "both_correct": 134,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 78
    },
    {
      "stage": 4,
      "a": "main_R",
      "b": "main_PC",
      "n": 295,
      "prediction_agreement": 0.9254237288135593,
      "both_correct": 161,
      "a_only_correct": 4,
      "b_only_correct": 9,
      "both_wrong": 121
    },
    {
      "stage": 4,
      "a": "main_R",
      "b": "main_GREEDY",
      "n": 295,
      "prediction_agreement": 0.9288135593220339,
      "both_correct": 161,
      "a_only_correct": 4,
      "b_only_correct": 6,
      "both_wrong": 124
    },
    {
      "stage": 4,
      "a": "main_R",
      "b": "main_RL",
      "n": 295,
      "prediction_agreement": 0.9322033898305084,
      "both_correct": 162,
      "a_only_correct": 3,
      "b_only_correct": 6,
      "both_wrong": 124
    },
    {
      "stage": 4,
      "a": "main_R",
      "b": "main_FIXED1",
      "n": 295,
      "prediction_agreement": 0.9322033898305084,
      "both_correct": 162,
      "a_only_correct": 3,
      "b_only_correct": 6,
      "both_wrong": 124
    },
    {
      "stage": 4,
      "a": "main_R",
      "b": "main_RESPONSE0",
      "n": 295,
      "prediction_agreement": 0.9322033898305084,
      "both_correct": 162,
      "a_only_correct": 3,
      "b_only_correct": 6,
      "both_wrong": 124
    },
    {
      "stage": 4,
      "a": "main_R",
      "b": "main_RL_SHARED",
      "n": 295,
      "prediction_agreement": 0.9322033898305084,
      "both_correct": 162,
      "a_only_correct": 3,
      "b_only_correct": 6,
      "both_wrong": 124
    },
    {
      "stage": 4,
      "a": "main_R",
      "b": "main_RL_RESPONSE",
      "n": 295,
      "prediction_agreement": 0.9355932203389831,
      "both_correct": 162,
      "a_only_correct": 3,
      "b_only_correct": 5,
      "both_wrong": 125
    },
    {
      "stage": 4,
      "a": "main_PC",
      "b": "main_GREEDY",
      "n": 295,
      "prediction_agreement": 0.9525423728813559,
      "both_correct": 164,
      "a_only_correct": 6,
      "b_only_correct": 3,
      "both_wrong": 122
    },
    {
      "stage": 4,
      "a": "main_PC",
      "b": "main_RL",
      "n": 295,
      "prediction_agreement": 0.9559322033898305,
      "both_correct": 165,
      "a_only_correct": 5,
      "b_only_correct": 3,
      "both_wrong": 122
    },
    {
      "stage": 4,
      "a": "main_PC",
      "b": "main_FIXED1",
      "n": 295,
      "prediction_agreement": 0.9559322033898305,
      "both_correct": 165,
      "a_only_correct": 5,
      "b_only_correct": 3,
      "both_wrong": 122
    },
    {
      "stage": 4,
      "a": "main_PC",
      "b": "main_RESPONSE0",
      "n": 295,
      "prediction_agreement": 0.9559322033898305,
      "both_correct": 165,
      "a_only_correct": 5,
      "b_only_correct": 3,
      "both_wrong": 122
    },
    {
      "stage": 4,
      "a": "main_PC",
      "b": "main_RL_SHARED",
      "n": 295,
      "prediction_agreement": 0.9559322033898305,
      "both_correct": 165,
      "a_only_correct": 5,
      "b_only_correct": 3,
      "both_wrong": 122
    },
    {
      "stage": 4,
      "a": "main_PC",
      "b": "main_RL_RESPONSE",
      "n": 295,
      "prediction_agreement": 0.9525423728813559,
      "both_correct": 164,
      "a_only_correct": 6,
      "b_only_correct": 3,
      "both_wrong": 122
    },
    {
      "stage": 4,
      "a": "main_GREEDY",
      "b": "main_RL",
      "n": 295,
      "prediction_agreement": 0.9966101694915255,
      "both_correct": 167,
      "a_only_correct": 0,
      "b_only_correct": 1,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_GREEDY",
      "b": "main_FIXED1",
      "n": 295,
      "prediction_agreement": 0.9966101694915255,
      "both_correct": 167,
      "a_only_correct": 0,
      "b_only_correct": 1,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_GREEDY",
      "b": "main_RESPONSE0",
      "n": 295,
      "prediction_agreement": 0.9966101694915255,
      "both_correct": 167,
      "a_only_correct": 0,
      "b_only_correct": 1,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_GREEDY",
      "b": "main_RL_SHARED",
      "n": 295,
      "prediction_agreement": 0.9966101694915255,
      "both_correct": 167,
      "a_only_correct": 0,
      "b_only_correct": 1,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_GREEDY",
      "b": "main_RL_RESPONSE",
      "n": 295,
      "prediction_agreement": 0.9932203389830508,
      "both_correct": 166,
      "a_only_correct": 1,
      "b_only_correct": 1,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_RL",
      "b": "main_FIXED1",
      "n": 295,
      "prediction_agreement": 1.0,
      "both_correct": 168,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_RL",
      "b": "main_RESPONSE0",
      "n": 295,
      "prediction_agreement": 1.0,
      "both_correct": 168,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_RL",
      "b": "main_RL_SHARED",
      "n": 295,
      "prediction_agreement": 1.0,
      "both_correct": 168,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_RL",
      "b": "main_RL_RESPONSE",
      "n": 295,
      "prediction_agreement": 0.9966101694915255,
      "both_correct": 167,
      "a_only_correct": 1,
      "b_only_correct": 0,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_FIXED1",
      "b": "main_RESPONSE0",
      "n": 295,
      "prediction_agreement": 1.0,
      "both_correct": 168,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_FIXED1",
      "b": "main_RL_SHARED",
      "n": 295,
      "prediction_agreement": 1.0,
      "both_correct": 168,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_FIXED1",
      "b": "main_RL_RESPONSE",
      "n": 295,
      "prediction_agreement": 0.9966101694915255,
      "both_correct": 167,
      "a_only_correct": 1,
      "b_only_correct": 0,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_RESPONSE0",
      "b": "main_RL_SHARED",
      "n": 295,
      "prediction_agreement": 1.0,
      "both_correct": 168,
      "a_only_correct": 0,
      "b_only_correct": 0,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_RESPONSE0",
      "b": "main_RL_RESPONSE",
      "n": 295,
      "prediction_agreement": 0.9966101694915255,
      "both_correct": 167,
      "a_only_correct": 1,
      "b_only_correct": 0,
      "both_wrong": 127
    },
    {
      "stage": 4,
      "a": "main_RL_SHARED",
      "b": "main_RL_RESPONSE",
      "n": 295,
      "prediction_agreement": 0.9966101694915255,
      "both_correct": 167,
      "a_only_correct": 1,
      "b_only_correct": 0,
      "both_wrong": 127
    }
  ],
  "screen_success": false
}
```

初筛：RL 相对匹配 R 最终 BA ≥ +1 pp、尾类 ≥ −0.5 pp、遗忘增加 ≤ 1 pp、新两类 ≥ −1 pp；还需最终 BA 严格高于协议列出的全部对照。
通过才追加 seed74003/74004 的全部四臂。每个新 seed 对 R 的 BA > 0 且保护通过，三 seed 平均 BA 差 ≥ 1 pp，平均候选 BA 高于全部预注册稳健性对照，才进入固定模块消融。
共享 T1 前缀不算独立重复；meta 属于训练来源且参与任务末 refit，官方 val 为反复复用的开发集。test 封存。
历史代理风险不是真实召回或遗忘保证；短视 contextual bandit，不宣称长期 RL 收益。全负结果、失败、分支及策略更新均保留。
HK 迁移需先核验完整23类资产、泛化评估及成本并冻结附录；未经此步骤不会启动。
