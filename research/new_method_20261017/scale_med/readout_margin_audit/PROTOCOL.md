# READOUT_MARGIN_AUDIT：封存后读出与margin失效定位

在实际访问新诊断输出和执行synthetic前封存。独立CPU描述性诊断，承接已失败DECISION_ANCHOR，不修改其结果或重置其时钟。依据：2026-10-20用户分享分析后的本会话数学核对，以及持续自主研究授权。

问题：D的类2新增损害，在固定F读出作用于D特征时是否已存在，还是更换自身读出后出现？mapper只含共同坐标的平移与标量尺度，因此记录原始g坐标下alpha*s²，避免把同alpha误当同有效尺度。分解是依赖中间状态的描述，不作独立因果归因，不升级L主候选。

只读初始inputs/0（2555训练+681已暴露开发）、decision_anchor四已封存FEATURE_SEAL缓存/四artifact、protection教师和MODEL_SEAL、已公开aggregate。严禁读取图像/pixels、future/test、旧E/I/O学习模型、重新编码、重训、重新拟合ridge、调参或选新部署模型。F教师必须复演其已选.01读出correct55/2/22/430；四自身读出必须复演原全部计数，否则停止未决，不救parity。

固定F/E/L/R/D；每臂两个split。13个预定组合：五个自身读出；F读出作用E/L/R/D特征；E/L/R/D读出作用F特征。覆盖全部4类/固定3竞争类，不只择类2损害。每个组合报告BA/recall/counts；每个端点报告以F原正确为分母的中间/最终四正确性格；固定真实类y与竞争类k的margin变化按F读出表征项+读出更换项精确分解，报告全体均值/分位[0,.1,.5,.9,1]、F原正确子集/自身新损害子集（空集NA），完整分母保留。额外报告真实margin=min各成对margin，但不将变化竞争类当固定对归因。

对每臂原F正确训练/开发样本，报告Q_D投影norm、每对margin变化界||w_y-w_k||*||delta Q_D||、全部3对都满足严格界小于F原margin的充分证书计数，以及实际固定F读出保持计数。证书只针对固定F读出，不约束重拟合头。不新增阈值/检验家族/CI或显著性声称，未选样本不丢弃；统计开发已暴露，method_success=false。

mapper一般精确化简z=g_I/s + offset；有效原始权重W/s、原始截距b+offset@W；以原始分数回放认证，不假设浮点中心恰为0。相同坐标下alpha_eff=alpha*s²为ridge目标的代数解释，尺度变化不自动证明风险因果。F/D与全部控制均报。

运行前synthetic：一般两级标量mapper与原始坐标分数一致、平移和尺度回放、成对margin分解与方向界、自由截距ridge的alpha*s²等价，容差1e-10；实际分数等价/分解/界容差1e-8（界允许1e-8数值余量）。原自身计数exact，F parity exact，finite全覆盖。任何失败保存FAILURE及退出回执，禁止隐式重跑/放宽门。

硬预算：本独立诊断首次实际运行至退出<=600秒（synthetic和正式共用一次原始时钟），新增远端+外部保守16MiB<128MiB，单文件16MiB、worker RSS4GiB，CPU300秒，无GPU和网络；原窗口17GB软停/20GB硬限保留。只复用NFS资源，不搬运features/模型。独立协调每秒查RSS/round/时间，超限终止本轮worker、5秒后kill，EXIT/FAILURE/资源/协调回执；不影响他人。中性父子入口，独立guard不依赖SSH/Codex存活。先真实挂载容量小probe；不删除历史。

交付完整源码/协议/全部聚合/计数/finite/parity/资源成本与中文结论，git通过本机代理推送核SHA与匿名HTTP；私有逐例只NFS（本轮默认不另存逐例，临时内存用后释放），不下载模型/feature/身份/原日志。只复用唯一scale-med，健康不新增轮询。完成后分析新假设，不自动启动训练。
