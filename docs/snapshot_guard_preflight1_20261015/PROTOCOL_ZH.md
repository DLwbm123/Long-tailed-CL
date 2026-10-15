# 四条真实零更新 GPU 预检：前瞻冻结

用户本次仅授权 GPU0/1 四条真实零更新预检，正式训练和评价均未授权。沿用已验收生产数学源7a077b69f16d843573b2bc7af5a46d9701c29bb0（CPU交付aaefd9cc）；训练、守卫数学/阈值/参数文件完全不改。新代码只是独立preflight-only入口与资源测量，新增小合成CPU验收通过才派发GPU。原生产CPU不重跑，旧已交付轮不操作。

固定顺序 ISIC Core+PMT+guard、ISIC LT+PMT+guard、HK Core+PMT+guard、HK LT+PMT+guard；最多GPU01并行两个。seed/order/fitmeta/预训练/图像变换/batch64/2epoch配置/AdamW.0003/WD.01/clip5/FD10/ridge.001/prox.01/原cosine/PMT ridge.001/guard固定全dual不改；预检沿用workers0/expected_steps0。真实训练run在第一batch backward后返回，不调用optimizer.step，wrapper再硬性拦截AdamW.step。四项全PASS后也直接STOPPED_NO_TRAIN；没有train/eval派发分支。

每条先用真实当前首task图像原canonical和首batch完成生产调用链、原guard epoch证书和反传，双pool梯度必须有限非零；全参数和buffer在模型初始化与所有预检结束后精确逐tensor比较。首task没有旧bank，不能宣称覆盖真实旧漂移。原PMT full-input-rank1536/ISIC6或HK21旧类synthetic shadow维持；另外复用已冻结CPU boundary fixture（ISIC Nfit9434/allcurrent11835/K8/rank1536，HK1038/1298/K23/rank1038）检测guard最大旧bank：将实际生产Session输入bank/offbank/region/R/pairs/fullfit特征暂驻该GPU，调用原CPU projection/完整full-dual求解与checkpoint校验，真实模型和首batch梯度同时驻留。旧统计均合成，不读取旧医学图像或旧医学checkpoint，不写正式统计。CPU合成生成和GPU传输/求解时间都计入本条预检驻留；不能称真实最大旧类训练或真实旧域效果。

new入口严格拒绝非preflight/非零expectedsteps/正式训练授权/评价；禁止val/test/reserved manifest及旧控制checkpoint访问。全参数buffer不变须在shadow后再验一次；记录真实batch梯度、两个pool范数、真实guard证书、PMT shadow及guard shadow标量、最大allocated/reserved、实际GPU驻留/operation墙钟、失败与全部费用。原PREFLIGHT含controller的完整私有文件保留NAS，不外传；只收匿名纯标量，不导出features/映射统计矩阵/逐样本或IDs/影像/模型/rawlog/traceback/privateconfig。

执行前显式代理公开源码协议，remoteSHA和精确PROTOCOL匿名HTTP200；NAS真实nfs/free/probe，GPU01每卡>=24576MiB准入，不借GPU23或动他人任务。target解释器compile gate和新增harness小CPU检查（禁train/eval/optimizer及CPU传输wrapper/小boundary证书）必须通过。CPU/GPU各自新源/目录冻结，不改健康源、不重启、不自动重试。真实preflight launch开始24小时资源等待及全部预检，不重置；这是预检上限，不是正式训练轮或其deadline。父子中性入口/tmp/q247c.py与/tmp/q247w.py，解释器/tmp/q107v/bin/python；数据/log/tmp/cache仍NAS。启动核实完整ps和NVML。

所有四条独立驻留各计一次，失败额外保留，不按operation/job去重，不加旧cumulative。旧闭合GPU391490.24710153416秒只作历史基数，当前活跃另列。只要单项失败/未运行整体不PASS，不补0、不放宽门或同源重跑；其余独立预检照原队列保留，无训练依赖。新CPU小检查资源另记，不和嵌套operation相加。没有GPU峰值或全轮耗时先验保证。

终态active空且父/worker退出后防重复收集一次，公开代码/协议/全部正负失败/REPORT/RESULTS/AUDIT/费用，代理push remoteSHA/精确REPORT匿名HTTP200后NAS独立public_preflight/回执/HANDOFF。沿用同一lca半小时只一次合并状态/必要log尾/完整ps NVML，不持续轮询。四PASS也不授权训练，须等用户新的直接指令；val不用来调参数，test/reserved始终封存，RL暂停。
