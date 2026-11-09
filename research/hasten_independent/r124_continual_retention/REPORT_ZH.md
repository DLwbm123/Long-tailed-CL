# R124 replay-free生成式连续RL机制实验

**单顺序开发VAL、非独立患者；首任务共同SFT贡献不记RL。**

|arm|post平均macro %|post平均tail %|平均forgetting pp|
|---|---:|---:|---:|
|GRPO|27.4443|11.8912|0.0000|
|TARS|27.4443|11.8912|0.0000|
|RAPO|27.4443|11.8912|0.0000|
|TARS_RAPO|27.4443|11.8912|0.0000|
|TARS_CE|13.4776|7.0631|44.8909|

原九门：{"post_macro_over_TARS_ge_1pp": false, "post_tail_over_TARS_ge_2pp": false, "initial_task_head_drop_ge_minus1pp": true, "worst_old_tail_over_TARS_ge_minus1pp": true, "post_macro_over_RAPO_positive": false, "post_tail_over_RAPO_positive": false, "post_macro_over_TARS_CE_positive": true, "post_tail_over_TARS_CE_positive": true, "classwise_forgetting_less_than_TARS": false}
全部通过：False

CE控制为匹配TARS权重、同query和更新次数，但无G8 rollout/KL，计算更少。
所有原正负结果与旧门保留，不把新连续条件改写为旧实验成功。
25行含5行共享首任务读出；实际21个读出序列，不是25独立重复。原生cache后端存在不等于可读取旧/未来TRAIN；accessor日志与seal先后已核。
校准保存了aggregate正确计数与数学表，未保存逐例校准预测；可核样本访问/类别分母/权重计算，不能事后独立重评分该校准正确计数。不追加生成补造。

## 实际学习与保留的区别

|arm|post尝试|非零参数步|最终参数等首任务SFT|正确RL响应|
|---|---:|---:|---|---:|
|GRPO|512|0|True|0|
|TARS|512|0|True|0|
|RAPO|512|0|True|0|
|TARS_RAPO|512|0|True|0|
|TARS_CE|512|512|False|不适用|

四RL臂全部新增任务共131072响应、0正确奖励、0实际参数更新；最终逐参数等于首任务共同SFT。组内优势恒零，RaPO的sigma变化没有产生学习。
零遗忘与较高seen平均macro来自停留在首任务模型，不是主方法学会新类、保留奖励的净增量或持续学习成功。主对TARS/RAPO增量0；匹配CE可学习部分新类但严重忘旧，也不是整体成功。

原九门通过数：4/9。旧R117等结果与原失败门保持。
RaPO原文附录分类任务提供累计seen类别列表，本冻结适配开放category-only提示没有该条件；未证明它是零奖励的单一原因，不把本轮当作者原样复现或所有RL无效。详见TASK_INTERFACE_AUDIT_ZH.md。
完整完成后仅允许另立一次共享累计seen词表的资格实验；重新generic共同首任务起点、匹配全部控制、不提供未来或逐例答案，不同时改SFT/推理长度/奖励。接口收益和监督收益不记RL或创新。
所有182226响应、2866曲线、360类行、当前TRAIN/已见VAL访问与封存顺序核对通过。CPU软件和历次saved-only审计另计，body计时不是完整进程CPU。
