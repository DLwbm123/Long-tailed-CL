# GAIN 四候选训练侧配对诊断

当前授权：只读原18阶段checkpoint和72个封存候选；旧训练图像仅作冻结推理诊断，不作optimizer/FD/统计训练回放。方向和0.125/0.25/0.5/1步长不变，无学习率、预算、seed或val调参；不运行RISK准入，不选最佳候选。

ISIC/HK × base/local_transport四轨迹。原科学source、order、seed74002、全部train计数、训练budget和DELIVERY逐项核对。候选直接复用gain_joint1的533562177c4fa39f18dcbe981e2c0d0f596a09ac sidecar，不重新求解或修改源checkpoint。复用原数学/高维原头兼容验证，不重复收费。

CPU合成生产检查通过并代理公开冻结source/protocol后，一次launch：四CPU元数据准备完整封存；四真实64图preflight依次在GPU0/1读取当前/相邻前一encoder，参数精确不变、CPU/GPU候选与原守卫有限一致、0optimizer/policy，且不收分类指标。全部4 PASS才统一DIAGNOSE。GPU2/3禁止，保护其他任务，四小时含准备/预检/诊断，不重置、不自动正式retry。

每stage只提取seen-prefix train影像，固定原eval Resize224/Normalize(.5)，batch64、workers0、shufflefalse。每批current encoder特征同时算OFF和四候选。task>1在同一规范样本顺序下读取上一冻结encoder；当前新类的class-balanced common global shift同批重建，旧类残差扣除此shift；不把区域位移减到不同维度全局特征上。current矩阵与before矩阵只在stage内存使用，不保留旧feature或逐样本分数。

直接观测旧类在当前encoder下的train分布，比较bank运输代理均值、投影协方差与实际竞争边界：均值残差/分数残差、配对残差均值及二阶矩标量、各边界proxy gap/actual gap/负间隔计数、原型年龄。46维成对矩阵和shift仅NAS私有。重建运输误差如实报告，不静默把原bank替换成观测统计。

按完整train identity/group、seed74002分固定两fold，每个stage和类完整报告n、救回、破坏和缺失fold。只作分类器训练内诊断，不能作为独立校准；患者独立性未验证，重复图片/stage/增强不增加独立n。不拟合新分类器或校准器，不反馈训练，不填independent=true。

预测复用封存point/Student-t三借用情景的pre-val计算标量；不读取旧val分类结果或新val manifest。与本轮train配对救回/破坏逐类对照，未知质量保留，只报告known-part与完整观察之差，不能称完整总体偏差。四候选正零负、所有stage/31类、旧旧/旧新混淆、新类守卫前后变化、两fold、漂移和估计误差全部报告，不以训练内结果选步长/改门。

正式诊断影像曝光219448（ISIC110412/HK109036，两控制合计），真实预检512曝光（4×64×2encoder），不复制/下载影像。正常4CPU prepare + 4GPU preflight + 4GPU diagnose，每次驻留分别计费，失败额外保留。合成CPU和并行wall单列，pipeline包含在驻留内，旧训练/方向/评价0新增计费。

源码/protocol先GitHub代理push、remoteSHA和精确匿名HTTP200。启动后中性coordinator/worker及全父子ps/NVML一次核对；超过30分钟则可靠后台运行，不默认持续轮询。真实结束且子进程退出后仅匿名JSON汇总collector一次；完整或失败部分结果/全部cost/report必须代理GitHub核实再独立NAS public_delivery/DELIVERY，missing不填0、不自动追加实验。模型、影像、IDs、privateconfig、rawlogs、tracebacks、features、均值/矩阵/shift都不外传。
