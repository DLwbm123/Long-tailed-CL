# MEASURE_RESPONSE_DECOMPOSITION 类0损害与两折差异的独立诊断

用户明确要求分析METRIC_MEASURE_ALIGNMENT的类0损害与折间差异，核旧控制后独立可证伪机制，不扫权重/温度/偏移救门。本轮仅解释原冻结G/I两个封存端点，不产生候选头、不插值/移除分量预测或训练。method_success=false，不改变STOP_METRIC_MEASURE_ALIGNMENT。

## 已知证据与新问题

旧聚合：类0两fold都净-10，24harm中19转类2；fold0 BA-.178397pp而fold1+3.819238pp。不能把折间BA差说成只有一折类0损害，或由support TV相近就判风险相同。

已核readout_support（组缺席/重数）、support_completion（补组/近远）、query_readout_cross_audit（换特征/头的路径分解）、readout_margin_audit/decision_anchor（参考margin/方向）等控制。它们不是本轮固定表示下权重改动的正规方程响应分解。负结果全部保留，不称独创算法。新问题：在同一个新I Hessian下，哪类support重加权的响应贡献了0-v2 margin移动；标签交叉统计变化和weighted Gram变化分别是什么。不是学习保护方向或确定可部署改良。

## 精确代数及强对照

固定X=[z,1]、G/I样本权p_G/p_I、同alpha.01仅罚非截距。H_I=A_I+alpha D。

H_I (W_I-W_G) = (B_I-B_G) - (A_I-A_G)W_G，容许原G正规方程残差。

按support来源类c和组相对增/减权桶b分解R_cb=sum_{i in(c,b)}(p_I-p_G)_i x_i(y_i^T-x_i^T W_G)。每桶DeltaW_cb=H_I^{-1}R_cb，12桶总和应重构原DeltaW。

桶按整数G_c*m_g-n_c的负/零/正号固定，表示减权/不变/增权；无阈值/搜索。每类三桶求和构成4来源类强对照。另同时计算target项H_I^{-1}DeltaB与Gram项-H_I^{-1}DeltaA W_G，两者和重构原DeltaW，区分两类统计作用，不把weighted Gram变化冒称encoder表示变化。

这是以新H_I为锚的算法路径分解，不是唯一因果归因。所有分量共享H_I，不能解释成分别改一类权后真实重训的效果。严禁将任何分量单独加到旧头做argmax评分或从结果挑桶/挑类构造新头。

## 数据与隔离

仅原producer PLAN cached_g2555/既定两fold、原Fsupport-only teacher及metric_measure_alignment两个I完整artifact。原681开发/future/test/rawimages/旧learned T/P/C/D禁止。每fold仅support行/标签/组进入RHS与统计，原F mapper保持exact，不fit。两个fold全部分量先COMPONENT_SEAL，才用query标签做endpoint replay与margin描述。query不入分量构造/方向/选择。封存原输入可物理含query，但计算分量只传support切片。

0新分类器fit、0encoder/mapperfit/梯度/GPU/模型更新。恰好2个真实多RHS线性系统求解（每fold一次、84列：12桶×6列+target6+Gram6），属于解析响应应用而非新头拟合；另合成1次已知系统求解，总solve调用3。不把所有线性系统调用说成0。不保存/下载分量或特征张量，只保留RAM与公开聚合。

## 可证伪描述门与类1全类风险边界

主假设：在两fold的全部类0及G原正确类0上，0-v2 margin平均变化均<0，来源类2贡献均为最负，且其负贡献占四来源类负贡献总和>1/2。四单元全过才CLASS2_SOURCE_DOMINANT_PATTERN，否则MIXED_OR_NON_CLASS2_SOURCE_PATTERN；任一缺失/零负质量为UNRESOLVED_PATTERN。不做bootstrap或显著性，不将已用于选假设的query当独立验证。类2竞争的19/24只是既有动机，不是新机制证明。

全部fold×真实类×固定rival，共24对，分别ALL/G_correct/I_new_harm三子集全报（72cell，空子集均值NA）；每单元报告原/新margin、总移动、4来源类、12桶、target/Gram贡献的均值/分位、加和残差。I_new_harm子集是事后描述，主描述使用前两子集；不能据它挑方向。另仅用原公开计数给出逐类BA贡献恒等拆解，完整类1和四类风险/RH/混淆与原G/I回放exact，原F query分数parity<=1e-8。

本轮不产生新性能候选，原风险门完整保留：BA>=+1pp、类1>=+1正确/5pp、逐fold BA和类1不退、F原正确类1H0、其他类成本各<=1pp。原I已失败，不重新启动bootstrap或改为更宽门；任何描述门通过也不能部署/升主或自动加保护。未来不同方法必须另封相同或更严类1/全类/两fold门、强控制及预算，不能挑类/桶/权重/温度/偏移救本I端点。

## 校验与预算

先合成：已知SPD系统的分量和/target-Gram闭合，零分量/主门通过失败缺失（1toy solve）；实际两fold G/I的support A/B回放误差<=1e-10、正规方程残差<=1e-7；12桶RHS和与DeltaB-DeltaA W_G误差<=1e-10；求解重构DeltaW、target+Gram与query margin闭合maxabs<=1e-8；finite/完整cell覆盖/endpoint计数exact。数值先写回执再assert。失败UNRESOLVED_MEASURE_RESPONSE_DECOMPOSITION，无重试、不放宽容差、不补未知值。

独立首次actual含合成/导入/子任务/等待<=600s，协调590s TERM、5s后KILL；CPU300s、RSS4GiB、单文件16MiB，新remote+localGit/helper16MiB严格<128MiB，预计remote<4MiB；max(logical,allocated)，旧窗17GB软停/20GB硬限，default1040384+localGit1GB、报告另16MiB预留。只NFS cache/tmp，实际挂载/容量/写读probe；0GPU，无像素/模型/环境复制，不删历史或监测旧退休PID，中性主子argv。成本协调wall完整一次、wait4worker/coord own CPU各一次，阅读准备网络未知项保留。唯一30min心跳原位更新。公开源码协议全部聚合/失败/NA/成本中文报告，代理push核ref/匿名HTTP200，DELIVERY双端，身份/逐例/features/Gram/模型/原私有日志/PDF不下载公开。
