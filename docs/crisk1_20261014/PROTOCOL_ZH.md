# CRISK v4 生产凭据、主四格与贡献统一评价

本轮已获授权：原18阶段checkpoint与72个方向候选只读复用；旧训练图像仅作推理统计，无训练回放/optimizer/policy更新。仅GPU0/1，test/reserved封存；不重训、不改变方向/四步长或冻结风险参数、不用val调参。主配置固定local_transport+CRISK。设计源845a665f7ec13fb325dbb46cee89be9f0d9cf5ee及其四项单因素贡献保持。

## 顺序与权限屏障

唯一顺序：4 CPU元数据准备→4真实64张训练图像零更新预检→4 GPU训练侧配对统计→4 CPU模型/风险准备与选择封存→4 GPU统一development val评价。每阶段四轨迹全部完成才进入下一阶段；任何失败保留错误/全部费用并停止依赖，无自动正式retry。固定队列ISIC_base、ISIC_local_transport、HK_base、HK_local_transport，每批两slots在0/1运行。24小时从真实launch起，含等待/全部准备与评价，不重置。预算到期只停止本轮自己的进程，不干预其他任务。

val manifest只在EVALUATE且四元数据PREPARED、四真实预检PASS、四统计STATS_READY、四选择RISK_READY与三个封存时间均核对后读取。每条预检用最终stage 32旧+32新训练图像，current/previous encoder配对；参数/state精确不变、CPU/GPU OFF、四固定候选与原守卫有限一致，无分类指标收集。不存在测试入口。

## 凭据的精确生产定义

当前原seen-prefix全部train在固定原eval Resize224/Normalize(.5)、batch64、workers0、shufflefalse下推理。task>1再用上一冻结encoder对相同规范行推理，当前新类均衡共同shift重建后扣除旧类配对残差。只临时保留特征，46维以内投影均值/协方差/配对二阶矩留NAS私有，不外传影像、样本IDs、逐样本分数或矩阵。

风险模型采用当前训练图像的逐类联合分数均值与ML协方差（P=[原float32头,C1量化方向]），只用于风险先验，绝不写回bank/Q/R/头。四步长为二进制缩放，先逐元素核对原sidecar与scale×C1完全相等；不允许近似替换方向。统计bank累计残差与相邻配对漂移分列，不再次修改当前训练事件计数。

模型为逐类shared-scale Student-t，df=目标类n−1、协方差乘(n+1)/(n−1)，65536数值节点、batch4096、seed1074005。每个模型共用一组OFF/ray节点计算四步长，原argmax tie与新类守卫保持。供体使用当前投影协方差并重新以目标均值为中心，trace缩放到目标；即使供体n大，Student-t的df和预测放大仍用目标n。借用不增加目标真实支持、质量或独立单位，不更换目标类别身份。

每类模型额外记录OFF错误节点数，再计算r=rescued/wrong、h=harmed/correct。条件分母0时发布null与不可识别标记；仅供API占位的.5不会作为已识别概率，关闭该条件的模型伪计数且不可识别供体分支不注入先验。实际目标训练计数的Jeffreys条件后验仍保留未观察事件的不确定性。节点数不是医学样本支持，也不是临床置信区间。

供体提案基于目标所有竞争边的[W_c−W_j,ray_c−ray_j]：normalized projected shape的Frobenius差z_b，提案权重n_d/(n_d+16)×exp(−z_b²/.1)。可信度rho按设计中的支持、年龄和z_b/z_d折扣；z_d为该供体直接配对残差在同一边界投影下RMS相对目标边界散度。当前新类没有运输，其漂移项为0且来源标记当前观测。常数/零散度条件不捏造形状，浮点容差内PSD裁剪及不可识别标记保留。

旧类RMS范围由本类相邻encoder配对残差二阶矩投影的对角线平方根给出，系数固定1。通过完整OFF/守卫ON分数盒与tie规则传播，生成OFF错误概率和联合救回/破坏外范围；条件比值用分母范围传播，分母可跨0时范围为[0,1]。s_q/s_r/s_h为点模型到这些范围的最大距离，给定模型的敏感项，不是跨患者概率证书。累计bank残差仅诊断，不与相邻范围重复相加。

每类目标OFF正确训练样本保存所有竞争边相对侵蚀的sum/square-sum/count，严格匹配C支持；据设计的固定预算 .25 n/(n+16) 检查全部旧类。仍只选择完整原候选或identity，不拼接类别列、不放宽新类守卫。首task原方向零，全部选择严格identity。所有五选择器须在val读取前完整封存。

## 对照与完整报告

主四格base/local_transport/base+CRISK/local_transport+CRISK；四个贡献为NO_RESERVOIR、NO_DRIFT、NO_DONOR_TRUST、NO_EDGE_GUARD。NO_DRIFT只去除效用敏感项，供体rho保留漂移可信度；NO_EDGE_GUARD只旁路旧类边界准入，不改分数或新类守卫。全部四候选与五选择器共享特征/分数，报告所有正零负、identity及拒绝原因。

报告各自控制的完整六指标/Acc/可得macroF1、全部AIR差距、local_transport交互、18stage/全部31类n与recall、救回破坏/旧旧旧新流、RAW与守卫、新类间隔、旧类侵蚀与保护牺牲、训练/val估计偏差、条件支持及供体/漂移/数值误差。选择不在val阶段重新计算。原控制OFF逐stage与已公开结果精确核对，旧F1缺失不填0。当前训练内统计仍不独立；方案受既有development诊断启发，重复val不是独立测试、临床或显著性证据。

正常GPU驻留12条（4preflight+4statistics+4evaluate），CPU驻留8条（4metadata+4risk）；失败全部另计。训练统计219448曝光、真实预检512、val15966，分别单列；模拟概率节点不算图像曝光。复用原控制训练/旧方向/AIR历史费用不重复计。全过程结束且进程退出后仅收匿名聚合一次，完整源/协议/正负或失败/report/audit代理GitHub核实，再独立NAS public_delivery/DELIVERY；不只保留NAS，不自动追加科学轮。
