# 多类零点损失记忆：前瞻冻结

上一低秩序列8af319a工程完成但分类门失败：ISIC中间stage与HK stage2–6损害旧类，候选与完整旧支持local oracle较接近，近似准确不保证分类好。当前OVR sum-head BCE会随共同头平移改变，而argmax分类不变。本轮检验多类耦合、共同平移不变的softmax CE目标；不加节点/谱秩/优化次数/半径或难度网格。结构理由不是效果保证。

## 数据和四臂预算

直接复用低秩fixture：r=2K再嵌入d，small8/32、ISIC16/1536、HK46/1536；N512/9434/1038、K4/8/23、原比例、seed263/264/265、独立eval offset10000。信号/噪声/混合/任务序列不改。新目标轮不是重新运行旧OVR结果，不重新评价旧NAS。

四臂oracle_wide(norm<=2)、oracle_local/moment_local/quadratic_local(后三norm<=.125)，全部用同一个新softmax目标；每stage48全batch head步、lr1/16、新head零初始化、逐head投影和class-balanced平均不改。CE与OVR梯度尺度不同，数值lr相同不等优化进度相同；不得把跨轮效果全部归因平移对称性。

旧moment仍原48 Adam.02、16节点+权重、ball投影，使用归一化节点；候选仍mu d/Q15方向15d/15特征值/遗漏谱rho1，共16d+16float64。当前支持内压缩，不降baseline，候选eig无optimizer不称计算等价。旧memory append不变、head可变不等无遗忘。支持统计可能含低样本信息，仅合成不授权医学原始features回放/隐私保证。

## 多类二阶近似和保守界

对于K个head w，定义b_j=w_j-mean(w)，R=max_j||b_j||，beta8，局部域仍每个原head norm<=.125硬拒绝。类c精确风险是E[logsumexp(beta*x^T w)-beta*x^T w_c]，class-balanced平均。

候选S_c=logK-beta*mu_c^T b_c+beta^2/(2K)*sum_j b_j^T Q15,c b_j。

grad_j=beta*(1/K-1[j=c])*mu_c+beta^2/K*Q15,c*b_j。

共同平移w的风险/梯度和预测不变，仅当平移前后都在原域内才能合法调用。域约束本身未改成平移不变域，不声称解决全部几何兼容性。类别增长时K和b重新算，memory只保存mu/Q，不含未来支持或头。

loss误差界 beta^2/(2K)*rho*sum||b_j||^2 + (4/3)*beta^3*R^3。

逐head gradient误差界 beta^2/K*rho*||b_j|| + 4*beta^3*R^2。

证明范围：unit x下沿t*b，logsumexp第三导数是u_j=beta*x^T b_j在softmax概率下的第三中心矩，其绝对值<=range(u)^3<=8beta^3R^3。Taylor三阶余项除6。梯度为beta*x*p_j，p_j''=p_j*((u_j-Eu)^2-Var(u))，绝对值<=(5/4)*range(u)^2，积分余项给逐head上界(5/2)beta^3R^2；实现使用更保守4beta^3R^2。加PSD遗漏谱rho的二阶截断误差。界可能很宽，必须完整报告；这不是医学保护或广义adapter证书。

## 实施、检查、评价

新worker只在隔离进程临时绑定shared.exact/quadratic/train_head和已有lowrank fixture，finally恢复；旧依赖源码与旧部署不改。train_head原48步直接复用，新增训练后一次风险/梯度计算（不step），报告实际训练目标及projected gradient mapping norm=16||W-project(W-grad/16)||。48个方法stage各一次诊断，成本在驻留内；小残差只局部驻点指标，不是分类效果/全局误差证书，也不新增迭代追收敛。

四CPUcase：新CE autograd/quad FD、零anchor精确、共同平移与类排列、正遗漏谱误差界、越域拒绝、绑定恢复/首task无旧memory一步精确（2合成更新）；三个完整合成序列。阶段2+1+1、4+2+2、13+2+2+2+2+2，共2304合成head更新、moment压缩576、候选压缩0，不是正式医学预算。全四臂全stage训练封存后才生成独立合成评价；旧fullsupport仅oracle对照或封存后诊断，候选训练仅当前支持与旧memory。

严格新目标门：工程全通过，两个高维各每stage candidate BA/oldBA/newBA>=moment_local，且各至少一严格收益。首stage oldBA=null，不调容差/挑stage/换主对照。原完整query gate=null，全部旧失败不改判。48方法stage/540类recall、宽域/局部oracle代价、训练目标/残差和所有正零负全部保留。即使过门仍非医学/GPU-ready，缺encoder变化/正式生产预算/新正式协议与真零预检。

## 费用与交付

CPU2线程，每case1800秒/总5500秒仅资源界。所有逐次wall/user/system/RSS和失败保留，父wall另列，内部train/fit/eig/diagnostic/eval/operation含驻留不加。此前19轮74CPU5546.389817275573秒；本轮只加一次。GPU历史400036.74851310626不变。0医学image/features/checkpoint/val/test/GPU；test/reserved封存。先完整GitHub/NAS公开交付，再按机制决策，不能无限补丁追门。
