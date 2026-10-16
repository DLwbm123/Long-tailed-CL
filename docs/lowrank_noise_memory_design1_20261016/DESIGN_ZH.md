# 低秩噪声下的共同零点记忆分类验证：前瞻冻结

上一轮 a927baf 的高维四臂全阶段满分，严格收益门失败，小规模旧类下降。这不能支持分类收益。本轮不改候选方法，先检验原高维单位噪声集中是否掩盖非平凡分类取舍；不扫描难度或在结果后改门。

## 唯一分布改变

原生成器的单位 Gaussian 方向噪声分布在 ambient d 维，各坐标二阶矩 1/d。新任务将整个生成过程放在预声明 r=2K 个坐标内，再零填充到原 d；r 是原公式已有的 K 个类中心坐标加 K 个混合坐标数量，不按数据/查询/效果选。噪声二阶矩变成固定坐标投影 P/r，独立于 ambient d。信号 .45、噪声系数 .85、非高斯混合 .6*(.8 或 -.2)、类比例、样本数、seed263/264/265、独立评价 offset10000 全部保留。small r8/d32；ISIC规模 r16/d1536；HK规模 r46/d1536。真实 rank 可能低于 r。

这是新合成分布，不覆盖或重跑旧结果；不得跨轮直接作严格配对检验或推医学因果。低秩也改变 Q15 的截断精度，不将结果全部归因于难度。噪声方向有限样本 realizations 不与旧样本逐点配对。未来不根据结果调整 r、信号强度、seed 或半径。

## 原方法与四臂严格复用

冻结依赖 check_shared_zero_memory.py 原样复用。新 wrapper 只在独立 worker 的 run_scale 调用期间替换 fixture 函数，并 finally 恢复，磁盘旧源码不变；unit 验证恢复。四臂 oracle_wide(norm<=2)、oracle_local、moment_local、quadratic_local(后三<=.125)，每阶段48全batch梯度步，lr1/16，class-balanced sum-head one-vs-rest BCE，新增head零初始化，逐head投影。

候选仍每类 mu(d)、Q15(15d+15)、最大遗漏谱rho(1)，16d+16 float64；S(v)=log2+4mu·v+8vTQ15v，共同zero anchor。原 loss/gradient 有界误差和 .125 域硬拒绝不变。旧moment48 Adam.02/16节点权重不降低；candidate eig无optimizer，不称计算等价。只当前类支持传入压缩；旧 fullsupport 只供合成 oracle 对照/封存后的误差诊断；不授权医学旧图或原始特征回放。旧memory append逐tensor不变，head可变不等无遗忘。

三个序列同旧协议：2+1+1、4+2+2、13+2+2+2+2+2。每scale全部四臂训练封存后才生成独立合成评价集；非医学 val/test。48行stage方法/540类recall，包含宽域与局部oracle、moment和候选所有正零负。

## 工程与新分布门

四CPUcase：新fixture单位范数/零尾/同seed跨ambient精确/不同seed/配置不变、fixture绑定恢复与无旧memory首步精确（2合成head更新）；三个完整序列。尺度总2304合成head更新、moment当前压缩576更新、candidate压缩0。复用已通过的旧算法算术，不再重复旧科学轮。

主门保持：两高维各每stage candidate BA/oldBA/newBA>=moment_local，且各至少一处严格收益，加全部工程与界检查通过。首stage oldBA=null。无容差、挑stage或改主对照。即使新分布过门也不改变旧失败。small结果、局部对宽域oracle代价、候选对局部oracle差逐项保留。原完整adapter query门=null。本轮不训练encoder，不验证医学/生产预算，不GPU-ready。

## 运行与交付

2线程CPU、每case1800秒/总5500秒仅资源界；4子进程每次wall/user/system/RSS全留。内部训练/构造/eval包含驻留不相加；父wall另列。此前18轮70CPU5240.123274484184秒，本轮只加一次；历史GPU400036.74851310626不变。0医学images/features/checkpoint/val/test/GPU。失败先完整交付，不在健康源修或重试。全部代码/协议/匿名标量/全stage/class正负/费用GitHub与NAS回执交付。
