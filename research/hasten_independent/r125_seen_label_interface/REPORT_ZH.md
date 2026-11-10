# R125 replay-free生成式连续RL机制实验

**单顺序开发VAL、非独立患者；首任务共同SFT贡献不记RL。**

|arm|post平均macro %|post平均tail %|平均forgetting pp|
|---|---:|---:|---:|
|GRPO|16.1930|1.7262|35.5672|
|TARS|17.0042|1.7262|35.1009|
|RAPO|22.9985|4.7727|16.7780|
|TARS_RAPO|22.9707|3.5823|15.1628|
|TARS_CE|14.7715|4.2888|40.7386|

原九门：{"post_macro_over_TARS_ge_1pp": true, "post_tail_over_TARS_ge_2pp": false, "initial_task_head_drop_ge_minus1pp": false, "worst_old_tail_over_TARS_ge_minus1pp": true, "post_macro_over_RAPO_positive": false, "post_tail_over_RAPO_positive": false, "post_macro_over_TARS_CE_positive": true, "post_tail_over_TARS_CE_positive": false, "classwise_forgetting_less_than_TARS": true}
全部通过：False

CE控制为匹配TARS权重、同query和更新次数，但无G8 rollout/KL，计算更少。
所有原正负结果与旧门保留，不把新连续条件改写为旧实验成功。
25行含5行共享首任务读出；实际21个读出序列，不是25独立重复。原生cache后端存在不等于可读取旧/未来TRAIN；accessor日志与seal先后已核。
校准保存了aggregate正确计数与数学表，未保存逐例校准预测；可核样本访问/类别分母/权重计算，不能事后独立重评分该校准正确计数。不追加生成补造。

## 实际学习与保留的区别

|arm|post尝试|非零参数步|最终参数等首任务SFT|正确RL响应|
|---|---:|---:|---|---:|
|GRPO|512|512|False|16756|
|TARS|512|512|False|15984|
|RAPO|512|512|False|11105|
|TARS_RAPO|512|512|False|10969|
|TARS_CE|512|512|False|不适用|

原九门通过数：4/9。旧R117等结果与原失败门保持。
本轮全方法统一累计seen词表、重新generic首任务SFT；历史R124开放问法与本轮的起点也不同，不能把历史差异全记成post提示单组件因果。接口与监督贡献不记RL或创新。
这是事先冻结的唯一seen词表资格实验；若不启动或原九门未过，不扫描prompt/答案格式/seed、追加reasoning或监督救分。不称作者原样复现、独立患者证据或所有RL无效。
所有182226响应、2866曲线、360类行、当前TRAIN/已见VAL访问与封存顺序核对通过。CPU软件和历次saved-only审计另计，body计时不是完整进程CPU。

## 首个新增任务的启动资格（不替代原九门）

|arm|正确响应|mixed准确率组|非零参数步|启动资格|
|---|---:|---:|---:|---|
|GRPO|5241|461|128|True|
|TARS|5193|469|128|True|
|RAPO|3943|446|128|True|
|TARS_RAPO|3891|448|128|True|

后处理首次prompt校验误用CSV原始连字符类别名；训练与CE目标始终沿用冻结R114引擎的hyphen→space规范化。校验器改为同映射后重新做CPU-only保存验证，原失败源码/记录保留，0新模型调用/更新，未重跑或修改科学结果。类别身份/顺序不变；这是协议中canonical类别表的实际字符串规范化澄清，非事后更改提示。
