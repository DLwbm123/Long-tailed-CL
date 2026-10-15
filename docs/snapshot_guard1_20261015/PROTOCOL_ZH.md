# 快照守卫正式四轨迹：前瞻冻结协议

用户直接授权“请立即开始实验”，本轮由四条已交付真实零更新预检推进到冻结的正式训练及全训练封存后的统一development val。仅GPU0/1；test/reserved封存，RL暂停，不用val调参，不增加seed、臂、预算，不读旧训练图像或新增回放/校准。

## 源码与准入

生产数学/worker/scope/eval及所有既有依赖逐文件与预检科学源16ad480完全相同（原生产数学7a077b6）；原8项生产CPU与4真实零预检复用，回执公开df79726d。SOURCE_REUSE保留一致性证据。本轮仅新增调度入口及无模型CPU调度检查，不改数学或再次预检。调度核对原始私有配置中全部科学字段、四PASS/梯度/状态不变/最大旧bank标量；只有运行路径、授权、source provenance和deadline可不同。旧预检仍为已终止已交付轮，不重新操作/收费。新source commit用于正式checkpoint元数据，不能冒称四预检在此提交重跑。

## 固定范围与比较

顺序ISIC Core+PMT+guard、ISIC LT+PMT+guard、HK Core+PMT+guard、HK LT+PMT+guard，共4轨迹18stage1288更新。主四格Core/LT/两守卫臂；另复用两背景无约束PMT贡献对照。旧8控制轨迹36stage不重训；只在本轮统一评价明确允许配对控制推理并计费。贡献协议和BUDGET_SPEC复制原生产冻结版，其中“未来/另授权”现由本次直接授权生效，数值与比较不改。

每臂ISIC468(100/72/296)、HK176(98/4/2/20/18/34)，2epoch/batch64/workers4/AdamW .0003/WD .01/clip5/FD10/ridge .001/proximal .01/beta .5/原cosine eta1e-5/seed74002/order/fitmeta/cached权重/医学变换不改。54守卫事件，复用原canonical/captured，零额外图像提取。全dual无cap，所有阈值/840秒守卫软资源门保持不变；失败停止后续依赖，保留其他正在执行的已派发worker，禁止同轮重试、调参或改健康源。

保护仅固定encoder/frame的epoch初始化及boundary最终head保存前的完整current-fit true-vs-all margin；每optimizer步证书失效；meta仅原统计。投影detach，中间原梯度未改。权威double头/兼容float槽/guard-aware checkpoint与evaluate严格按原生产契约。same-state OFF/proposal不代替独立控制，精度差异单列。训练点保护不是val无损/旧域外推或医学有效保证。

## 屏障与停止

部署后先target compile及独立CPU调度检查（mock进程，不是新科学CPU或GPU预检），再NAS nfs/free>=40GiB/小probe和GPU01各>=24576MiB准入。可见中性父子入口。实际launch起24h含等待/train/eval，不重置；超时只终止本轮自己的进程组，绝不处理其他任务。

调度只派发train/evaluate，无preflight派发。全部4条TRAINED且每条steps/stages/policy、所有GUARD_TRAINING_SEAL和PMT TRAINING_SEAL的source/version/order/seed/budget精确一致后才EVALUATE；原worker再次全轮检查后才能打开val manifest。未齐/失败停止依赖，不读部分val。正常4train+4eval=8次新增GPU驻留，旧4pre费用已闭合不再加。

## 费用、监测和交付

GPU驻留逐次收费，CPU守卫/传输/操作包含其中不叠加；父wall另列，未测null。旧闭合GPU393758.130153168秒只加本轮新增一次，不加cumulative。预检峰值不是多task真实峰值或24h适配证明。CPU调度小检查驻留/user/system/peakRSS另列。

沿用lca每半小时最多一次合并PROGRAM/activeSTATUS/必要私有尾日志/完整psNVML。正常健康安静，失败先私有核对并匿名记录；不持续轮询、不改健康源。终态active空且父worker退出后防重一次collect，只匿名聚合/标量/匿名混淆；禁止权重、特征、统计映射矩阵、逐样本ID/predictions/images/privateconfig/rawlogs/traceback。

完成完整主四格/无约束贡献/交互/六指标Acc F1/31类及全部stage/新旧混淆救回破坏间隔/PMT精度差异/守卫可行性和成本/所有正负失败/AIR差距。缺失null，不单臂配对，不以单次development val当顶级、临床、独立患者或显著性证明。源码协议与完整结果按显式代理GitHub push/remoteSHA/精确REPORT HTTP200再NAS独立public_delivery/DELIVERY/HANDOFF双端核实；启动先做独立startup交付，不能称结果已完成。
