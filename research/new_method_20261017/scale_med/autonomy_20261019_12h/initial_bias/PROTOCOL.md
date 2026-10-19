# 初始阶段原有bias适配：分类比较协议（实现/预检前封存）

状态：只有协议封存，尚未实现或启动分类运行。必须先完成同包装器合成聚合/失败回执、实际训练步/恢复、head-only特征不变、增量统计和delta映射预检；预检通过后源码公开封存，再启动。bias_binding_v2的工程探针和合成delta不得作为初始化。原全部失败门不修改。

## 独立问题和范围
固定表示上的CE/OvR/循环/核/协方差及统一读取都没有解决类1风险，不等于任务监督进入图像编码器没有价值。本轮只检验一个已知强基线：在初始2555训练行上更新原有预训练bias，之后编码器永久冻结。不是新算法；BitFit及初始任务适配已有相关工作。未来不继续编码器训练，不新增模块，不换层/池化/width/rank，不加入范数/路由/蒸馏/融合。目标是判定这一个固定训练过程的实际风险，不声称优化了全部非凸函数族。

## 固定训练规则
原APART生产路径和锁定预训练权重；既有原环境/合格原reader像素。bias集合由权重文件非head的全部.bias键定义(74个checkpoint tensor、原加载器映射98个native tensor，102912参数)，不更新pool、routing/assigner、original_backbone或任何weight。不按梯度挑子集。用原有head/head_few的前四类logit训练；这些临时head是原网络既有部件，训练后丢弃，用原ridge重新读出，不增加新模块。

固定10个完整epoch，batch8，AdamW lr=.003、weight_decay=0、betas(.9,.999)、eps1e-8，无scheduler、AMP/精度搜索、clip或早停选择。10epoch/.003沿已有APART配置的固定参考值，不宣称适用于视觉bias的最优值。encoder与原routing保持pointwise/eval，允许目标bias和临时head求梯度；禁用原推理路径中本来关闭的dropout，保留所有层与池化。原encoder初始化seed沿历史lock；数据shuffle独立seed74162连续生成每epoch排列，不增seed。

原组图像平衡q每类质量1，初始总质量4；目标F=sum(q_i*(CE(head_i)+CE(head_few_i))/2)/4。均匀无放回minibatch按N/4乘batch均值(q*两head平均CE)，尾batch用真实batch长度；不作图像复制/重采样。初始验证681不进入训练/epoch选择，未来验证442/8不用。每步loss/所有被优化梯度/参数须finite，恰好完成2555×10行训练曝光；不宣称全局最优或梯度收敛，只对预定有限日程的模型作比较。未完成、nonfinite或更新越界记未决，不看测试，不以增加epoch/lr/seed补救。

## 强简单控制和G/P/C流程
主候选预定为G_Bias（最简单全局臂），不是见结果选C/P最优。同一适配编码器同时产生G/P/C完整表示：g走原APART双分支FP32生产，h走同原像素的现代FP16 block6，bias经原加载器确定无损键转换共享；原width1535/1375/1318、rank128、PERM74010及残差惩罚.01不变。P_Bias/C_Bias是预定表示控制，完整报告但不能替换主G。

冻结编码器+同一临时head训练是HeadOnly强控制。可在初始冻结raw pre_logits上缓存计算等价训练，先预检cached/fresh head logit和梯度一致；训练同10epoch/同排列/同q/同优化器，仅head更新。预检证明head不会影响g/h特征，之后ridge的G/P/C可直接复用decoder_matched的G/P/C_Matched精确预测，不能把它算作未运行的独立复验。HeadOnly损失与bias训练损失全部保留，不用训练损失代替风险。

Bias与HeadOnly从同一初始head状态开始，独立重置optimizer/排列；HeadOnly学习后的head不得传给Bias。真正bias训练初始化只来自历史预训练seed。头训练后encoder_delta仅存最终偏差，未来冻结；包括head/optimizer的研究恢复checkpoint计入20GB全窗，但不当作部署状态免费内存。

初始适配完成且封存encoder后，重新提取该固定编码器的初始g/h，P/C全部norm/U/V/输出norm只用2555训练拟合。仍只让初始G的681验证BA在alpha[.0001,.01,1]选一个共同值，同分大者，未来/P/C冻结该值；不是强制新G仍选旧.01。未来每阶段只访问当前到达支持，按原q增量充分统计；旧输入退休，不因NFS物理文件存在而回读拟合。全部9头和变换/alpha封存后才允许测试像素、标签和旧预测进入评估。原划分/顺序不变。

## 主指标、统计家族和判定
继承冻结G/P/C_Matched为直接控制，全部原54方法原样报告。主G_Bias对三个冻结G/P/C的未来J=(stage1BA+stage2BA)/2，以及G_Bias对冻结G的类1三个阶段召回，合计6个预设主对比。10000次seed74163类内已知组配对bootstrap，相同draw跨方法和阶段，报告95%与6项Bonferroni区间[.05/12,1-.05/12]。主要BA/召回仍图像等权，组作为重采样单位，附报组等权。已知组不是已认证患者。

成功必要门：J对冻结G>=1pp、对冻结P/C点差>0且三个调整下界全>0；类1三个阶段对冻结G各>=5pp且三个调整下界全>0；两个未来BA对冻结G不下降；类4出生/转旧及类5出生召回对冻结G不下降；旧四类宏、最终前五类宏成本各<=1pp；类0三个阶段成本各<=1pp。不得宏平均掩盖类损失。全通过仅称INITIAL_SUPERVISED_ADAPTATION_HEADROOM，method_success=false（已知基线、探索性已暴露测试、非独立临床证据），不是原创算法成功。

完整但不通过记STOP_FIXED_INITIAL_BIAS_NO_EXTENSION；缺失/数值/隔离/容量/训练曝光失败记UNRESOLVED_INITIAL_BIAS。P/C辅助最佳、训练loss下降、工程预检、单个未调整区间均不能替换主门，不追加lr/epoch/seed/参数子集或新读出。

## 持久预算、运行空间与停止
所有方法使用同原width的对称A无损上三角持久态，公平控制不改变预测；每臂必须包含六类A/B/W、全部norm/U/V/PERM、最终canonical bias delta及完整元数据，实际artifact<=19,058,800bytes。不能把新delta当免费共同预训练参数，也不借packing增加width。bias_binding_v2只证明模板G/P/C约10.07/10.47/11.14MB可容纳；真实训练后仍需完整回读/字节验证。

所有运行数据在本窗NFS，直接只读复用decoder_matched的pixels_0..3，禁止再生成/复制3.68GB像素。stage0训练只取~valid行，物理chunk包含验证像素时必须先按原mask筛训练后再进入loss；验证只供后续G选alpha，不选择epoch。当前stage退休后当前learner的逐例输入访问守卫关闭。预计本轮新增<=512MiB（含恢复checkpoint、g临时输出、统计/头/报告及NFS分配余量），写前至少保守预留1GiB；当前全窗约12.55GB，需实际扫描而非估计放行。单文件512MiB/17GB软停/全窗20GB硬限。可重写本轮自己的恢复checkpoint，不得覆盖旧失败/历史资产，不清理旧模型环境来腾空间。

只GPU0，先显存/NFS挂载容量/写读probe，中性主子入口，现场完整ps/GPU核验。新启动硬截止2026-10-20 03:29:18UTC，无时间/CPU/GPU时长上限；到期不新启、不强杀健康任务。成本累计并避免父子/正文重复计算。完成公开源码/协议/全聚合/成本/失败/实质中文报告，push核SHA匿名访问并存DELIVERY；禁逐例/图像/特征/模型/私有路径/日志。
