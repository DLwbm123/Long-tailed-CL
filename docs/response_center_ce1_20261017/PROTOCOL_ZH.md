# 参数保护与类别间响应保护：冻结的最小对照

## 依据、唯一改动与复用

用户明确授权在q398分析交付后，保持预算与判定标准，执行最小对照。q398已完成并交付：两组收敛，ISIC/HK均双片失败；HK Qfit +5.563192pp但Qgate −5.799805pp，ISIC旧类损伤伴新类收益。小参数位移未保证这两个端点的预测稳定，尚不能推断响应保护有效。

固定原ridge W0、相同CE、H+新S、初始化、.064系数、beta8、优化器。只将参数二次度量||W−W0||²替换为类条件、类别间响应二次度量。每数据集只训练1个response头；参数保护头直接复用q398已交付端点，封存本轮候选后才加载。历史源码/结果不改。无调参网格、插值、额外零中心罚项、样本选择或原型迁移。

## 响应定义和尺度（评价前固定）

令d=1536、K为已见类数，p_i=1/(K*n_yi)，deltaW=W−W0，s2=sum_i p_i||z_i||²。使用与CE完全相同的全部合法H+新S，而非另选旧类子集。对每个真实类c与所有j!=c，保留z_i^T(delta_w_c−delta_w_j)；含旧—旧、旧—新及新—新竞争。不只固定最强竞争者，不挑开发获益类。

Rraw=sum_i p_i sum_{j!=yi}[z_i^T(delta_w_yi−delta_w_j)]²/(K−1)。它的二次型矩阵trace为2*s2；class-contrast子空间维数d*(K−1)。因此预先取tau=2*s2/[d*(K−1)]，Rnorm=Rraw/tau，即Rnorm=d/(2*s2)*sum_i p_i sum_j gap_ij²。系数仍为.064/2，目标CE(8zW)+.064/2*Rnorm；等价U=8W下.001系数。仅支持数据决定s2，无查询或拟合终点调节。

这是平均曲率尺度对齐，不是每方向约束强度相同、相同端点位移或相同实际FLOPs。响应保护不罚所有列共同平移；该平移也不改变CE。全支持上的类别差异若处于响应零空间，则支持CE也不变；从相同W0起始，不额外利用零空间或添加ridge地板。它仍可能保护错误，且支持响应小不保证未见查询错误小。本轮不选择性过滤正确样本，也不据此宣称新方法或泛化保证。

## 数据边界和对照

相同1536DT归一化特征、零adapter、类均衡H+新S：ISIC2354、HK264。完整均值重构maxabs<1e−12、完整岭正规方程res≤1e−7、counts/finite/model一致检查保留。原centroid/ridge/affine三份绑定加q398参数头DELIVERY绑定。拟合前q398头及其结果JSON也禁止读取，只有交付元数据可读；候选写name.ce.private.pt并seal后再读参数头和query/ref。

评价三头改为raw_ridge、parameter_ce(q398)、linear_ce(response)，以参数头替换原次要均值头，因此仍2155共享查询、6465个三头logit样本。原ridge对原affine报告、参数头对q398报告的逐类整数精确复现、CE1e−10。主判定完全不变：收敛且每片BAall严格改善、old/new各下降≤1pp，两片均过才datasetpass；主对照仍raw_ridge，不因相对q398改善切主或部署。另完整报告参数→响应及ridge→响应的逐样本配对、所有类/组指标和新类学习效果，不把更少学习当保护成功。

原H从原centroid herding直接读取，新S同前，无bank复制。禁原旧S、teacher、保留E、原图、新标签；两片已曝光开发，不构成独立测试或患者级证据。此响应在既定压缩支持上实现，不是无逐样本记忆的正式新算法。

## 检查、输出、预算

共享脚本增加可选response_scale；旧协议默认为parameter。工程自检在每正式fit前运行：非零中心解析梯度、显式样本/竞争类求和对照、共同logit平移不变、度量trace=d*(K−1)、W/U目标与梯度等价、默认旧目标兼容、类内复制后目标/梯度/scale不变，以及二维拟合对照80步标量根并检查收敛。二维对称对照与q398相同最优解；2工程fit，工程gradient数closure+3每数据集，不额外fit或反传。

2CPU任务每个900秒、总3600秒、2线程，0GPU。每数据集1正式fit，总2fit；L-BFGSlr1 max_iter200 max_eval250 history20 strong_wolfe tolgrad1e−8 tolchange1e−14、closure硬上限300/fit，总≤600加2终点梯度。终点grad∞≤1e−6收敛，否则只记预算终点，不延长/重试。计算响应会增加单次closure工作量，计入同一时间上限与实际CPU成本，不能称严格等FLOPs。

2完整支持矩重构、0基线重拟合/正式solve。共享支持2618/查询2155/三头6465logit；2新头32256个float64=258048字节，无额外持久推理模块。已有参数头临时加载用于比较，峰值RSS另记。封存后额外对两种候选在相同支持上各计算一次响应诊断，共5236个支持delta-logit样本，不重复提取特征，不算梯度或拟合；这项及实际工作内存/CPU明确计入费用。

保存完整fit/unit轨迹、初末CE/目标/参数距离/实际保护值/收敛/次数、response_scale、逐类原始和归一化支持响应、support/headstats、全class/summary/CE/errorsoldnew、margin/rank/top2/3/ties/competitors、两对配对、原主gate与失败成本。0encoder/adapter/teacher/memory更新。全部负结果保留，完成后公开报告并依据结果推进老路线，不承接新方法会话。
