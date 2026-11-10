# R127：固定权重下菜单扩展与最终首task head损失的归因诊断

本协议在实现/新模型调用前冻结。它不是RL训练、continual重新训练、prompt搜索或R126缩短矩阵。R126时间未准入的原20h钟、失败源码/费用全部保留。本诊断保守沿用更早1794271927.5872638作为first preparation，4h硬截止1794286327.5872638（Nov10约12:52:07+08），所有准备/软件/native/失败/子进程共钟，不重置；只pro5000物理GPU1。

## 新问题与边界

R125首task head绝对变化同时改变了模型参数与累计seen菜单。此前未测同共同SFT权重在最终菜单的旧类读出。本轮只解释最终首task head/old5读出中，两种变化各贡献多少；相对同菜单的主方法与对照差异不能被共同菜单效应解释。不得撤销R125任何原门/结果，也不把菜单收益记RL。共同首task模型已拟合首task TRAIN；旧VAL仍只是组件级开发集，非患者独立或临床验证。

全部任务已经完成/到达，这是回顾性冻结仪器诊断。读同一R125 COMMON_PREFIX_SFT及5个stage4臂的权重，只读不修改、不载optimizer；不构成新在线learner的防未来数据起点，不把首task模型看到最终schema伪称当时在线可得。所有输入图像限R125 VAL中canonical0..4全部624个组件，不选样本/类别、不看标签改变条件，TRAIN图像/feature读取0。只复用R114 generic冻结视觉VAL cache后端，逐次accessor拒绝所有非这624行，缓存存在不许读其余VAL像素。只读共享原模型/feature/权重，不复制大文件。

## 唯一7个读出

COMMON_INITIAL_MENU：R125共同首task SFT+原stage0的5类模板，先做第一16行native parity，再完整624行与R125已存共同首task预测逐一核对，必须全部相同；否则关闭本诊断，保留费用，不调prompt、batch或processor救分。
COMMON_FINAL_MENU：同一共同SFT+原stage4的23类模板。
GRPO/TARS/RAPO/TARS_RAPO/TARS_CE_FINAL_MENU：分别R125各臂stage4冻结权重+同原stage4模板。

只用R125原始stage0与stage4 prompt binding，类名继承Engine hyphen→space，ID升序分号连接、原问题完整文本；不生成新菜单、不更换类别顺序/格式/长度/重采/硬mask/reasoning，不测learned模型配5类限制菜单，避免oracle task-ID部署解释。batch16、624旧VAL原顺序相同，manual greedy/max24/EOS完全同R125；7读出以列出顺序一次完成，全部权重前后逐参数不变。不按结果追加读出或选择菜单。

learned在原R125全seen VAL时的batch组合可能不同。因此5个最终模型也在完全相同的旧5类batch上重新读出，不能直接把旧保存读出与新基线的差全归参数。保存每臂旧历史预测与新matched-batch预测的差作为单独残差，不替换R125原数字。所有私有逐例预测只留远端。

## 主估计与完整报告

A=共同权重/原5类菜单；B=共同权重/原23类菜单；C_arm=各最终权重/原23类菜单，全部同624图同batch。对head4(1..4)、old5 macro、micro和每类报告：menu=B−A，parameter_at_final_menu=C−B，matched_total=C−A；严格加法恒等式核对。旧R125历史读出H_arm另报 readout_residual=H−C，且 historical_total=menu+parameter+readout_residual。残差若非零仅记录，不能未经实验归因BF16/并行/采样某单因。

相对控制 C_main−C_control 的菜单项精确抵消；报告main与全部控制的head与每类差，不借共同菜单解释相对保留差异。主要输出是有符号贡献，不是成功门。预先用原1pp head阈值描述menu效应是否超过原容差；只在matched_total<0时给menu loss与总loss比例，允许负或>1且说明抵消，不裁剪、不用比例修改原gate。尾/tiny原失败不由本head诊断概括；首task只有tail class0、无tiny，不作全23类tail改善结论。

## 原生准入、费用与停止

CPU检查原stage菜单字符串绑定、相同readout集合/旧VAL准入与加法分解。GPU模型来自R114实际同Engine；只恢复R125 COMMON权重、全部参数requires_gradFalse，不创建optimizer。native首两缓存VAL feature分别经full-model hidden和手动同路径hidden/同形head核对<=.05，rawfullhead差保留；这是已缓存视觉语言路径，不声称新原图视觉encoder parity。额外4个cached feature-forward行、0新encoder。

native用旧VAL首16行分别作COMMON initial/final菜单greedy，两组32响应保留复用对应正式读出，不重复收费或挑样本。first16初始预测必须同R125保存；以两个16row真实wall的max做全部7x39batch、乘1.5+600reserve后小于4h原钟剩余才继续，不能凭R126或估算fake通过。完整初始624 parity也必须通过后才作其余读出。每次native/readout/失败/原数据已有成本、实际input/generated tokens/wall/CPU均报。

正常4368新响应(7x624，含native复用32)、4372 cached feature-forward行(加parity4)、0新视觉编码/0optimizer/0TRAIN图像feature读取。本实验固定7读出/35类行（7x5），5个臂归因行、每臂每类分解另25行，所有den均原624旧组件且类分母不变，非7独立样本集。保存readout/inputtokens/generatedtokens/time与private逐例预测/访问；CPU saved-only独立整数计数/加法恒等式/weights immutable/配置模板/4368-4372费用核对后交付，不再生成。

旧根R114–R126保守6995677184B+本机200MB+新root预计<150MB，累计严格<10,000,000,000B、不按新诊断重置。所有HF/Torch/CUDA/Triton/tmp/pycache计新root；新权重/大cache副本0。独立2s后台guard计全部旧基线，在9GB/deadline−120/RSS48GB/free1GB预停；filecap256MiB/core0/CPU4/allocator56GiB；真实挂载写读probe/GPU1free/RAM核后才启动，不动他人/不my-gpu。任何失败保留源码/日志/成本/原钟，不忙重试。仅原tars-rl监测，不新任务。

实际CONTROL_EXIT0后只本R127 CPU-only finalize_saved；safe source-only code/config/聚合REPORT/CLASS_RESULTS/ATTRIBUTION/ALL_ATTEMPT_COSTS公开，显式HTTPS7897proxy/NO_PROXY空/精确allowlist/no-force，remoteSHA与匿名200才完成；runtime与postprocessSHA分开。禁止身份/图像/私有索引/逐例预测/ACCESS/weights/features/私有路径/原日志/PDF/凭据。不改变任何R125/R126结果或协议。
