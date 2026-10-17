# 冻结头在旧记忆张成空间内外的 margin 分解

## 依据与范围

q399/q400/q401在同一预算内均未收敛，进一步求解所得query收益不一致。q401响应头在旧H上的归一化响应变化ISIC约.02315、HK约.16360，但ISIC Qfit旧类BA仍下降12.77369pp。参数头在支持上变化更大，也更充分拟合新类。这提示值得检查旧记忆覆盖与query变化的关系，而非继续轮换求解器。

本轮不训练、不改头、不改正则/坐标/判定。复用已交付的原ridge、q398参数头和q401响应预算端点；q401仍未收敛且完整主判定失败，不因新诊断重定义为成功。只解释已有端点，不选择新模型、不部署。

## 支持空间与数据访问

使用同一零adapter归一化特征，取全部合法旧H（ISIC64/HK169）做一次thin SVD，Q为保留右奇异向量。保留阈值max(H.shape)*float64机器epsilon*最大奇异值，是固定数值秩规则，完全不使用query/结局选秩；不截选子集、不扫描阈值、不加正则。Q只作工作内存，不新增推理模块。

query之前验证Q正交误差<1e−10、H投影相对重构误差<1e−10，并封存basis元数据（维度/秩/阈值/残差）。禁止在此前读query或q401结果参考；头与来源固定、无拟合。四份centroid/ridge/parameter/response交付绑定；三头shape1536×K、DT/finite、零adapter及模型一致性验证。禁止旧S、teacher、E、raw、新标签和新数据。H原文件直接读，不复制旧历史。

同一合法新S仅用于支持覆盖诊断，与H合计ISIC2354/HK264，精确逐类计数保持。支持各类及query各类报告||z−QQ^Tz||²/||z||²，不据结果调整Q。已曝光Qfit/Qgate均为开发资料，患者分组未知。

## 类别竞争与精确分解

对每个参数/响应头W1相对原ridge W0，令ΔW=W1−W0，z_parallel=zQQ^T，z_out=z−z_parallel。margin使用beta8缩放的真实类分数减最强非真实类分数。基线竞争者j0与候选竞争者j1分别记录，不能只固定旧top1而忽略切换。

固定j0时，margin变化等于beta*z_parallel*(Δw_y−Δw_j0) + beta*z_out*(Δw_y−Δw_j0) + 候选竞争者切换校正；校正为候选实际margin减候选对j0的gap，恒≤0。固定j1时，margin变化等于相应parallel/out分量 + 基线对j1的gap减基线实际margin，校正恒≥0。两种等式与完整logit差逐样本验证最大误差<1e−10。

两头、两query、全部类保留all/harmed/rescued/both_correct/both_wrong五种分区；后四类互斥且穷尽。每行保留count、竞争者切换数、|outside|>|parallel|数量、覆盖比例及各有符号分量min/median/mean/max。零计数分区保留null，不删除。导出后可作按样本加权和按类汇总，明确分母；不通过挑受损类建立未经预设的判定。

三头全部逐类count/correct/errors与q401原输出精确匹配，CE误差<1e−10。包含参数臂作为参照，但没有新增性能通过门。若空间外分量大，只说明相对旧H线性张成空间的分解；完整响应目标还保护新S，空间外不等于全目标零空间或完全未保护方向。相关性也不证明压缩损失、因果机制或可恢复的全历史收益。

## 工程、预算与交付

二维固定例子验证分量、竞争者切换和零更新恒等式；无工程fit、autograd或HVP。正式2个thin SVD，无新头/梯度/优化/基线重训，2CPU各900秒、总3600秒、2线程、0GPU。支持2618与query2155特征前向；三头querylogit6465，另parallel/outside诊断logit8620样本计算明确计成本，不冒充免费。仅重用归一化特征流程，不运行encoder训练或访问原图。

0adapter/encoder/teacher/memory更新、0新标签、0持久推理参数。预计support coverage21、query coverage42、三头class126、分解420行含零分区。所有运算计CPU/wall/RSS，从q401累计；不声称严格等FLOPs。

先公开协议源码后启动，完成退休后交付完整汇总和报告（无个体标识、特征/头向量或服务器路径）。根据空间内外与切换项的实际证据决定是否存在下一步可检验改进；若没有，明确报告瓶颈，不为持续运行空转。
