# R128：当前完整标签的on-policy对比辅助监督资格实验

在实现/新modelcalls前冻结。first preparation保守采用本轮入口2026-11-10T02:19:39.063Z（1794277179.063），23h硬截止1794359979.063，严格<24h；准备/软件/native/失败/重试/子进程共钟不time.now重置。原R126旧反馈候选原20h失败与R127完成记录不改、不poll旧PID。新问题是current-label对比监督，不读旧buffer，不减少原R126矩阵以绕时间。

## 起点/到达/数据

只读复用R125按generic/B0/seed74005、首task0..4自然2epoch训练出的唯一COMMON_PREFIX_SFT（306updates/4868queries）；学生接口/架构完全同R125，故不重复该SFT，也不追加SFT救分。它是已经验证的合法无未来数据前缀，不是全TRAIN SFT或VAL最好checkpoint。所有4臂从同此前缀分别建立自己的model/AdamW/sigma/teacher，post optimizer首次task2创建2e-5wd0，之后step128/256/384/512连续，不共享策略或梯度。历史SFT和624首taskVAL成本另列，计入完整链但不伪称本轮新调用；共享首task读出直接复用，不重新生成。

canonical真实到达5+5+5+4+4，scheduler知道schema/顺序/manifest元数据，learner不看未来图像/feature/反馈/目标。科学TRAIN仅当前task；旧及未来TRAIN由accessor拒绝，历史TRAIN replay0、无buffer。只读R114 generic冻结视觉cache作后端，不允许全历史访问，不复制模型/feature。stage1..4全部4臂seal后才统一seenVAL审计，不进权重/teacher或选择；输入类别表仅累计已到达类，hyphen→space，同R125固定模板/greedy评估/manualtemp1 G8 max24/EOS，无hardmask/reasoning/formatreward/学生提示搜索。

每新task每臂128attempt/8query/G8，current自然replacement共享tape seed12474005+stage、学生采样seed12480000+1000*stage+k同臂。只当前TRAIN各臂自己greedy一次TARS表，公式/medianfloor1e-8/min1/naturalmean1同R125，任务内固定，不乘KL。所有臂保留同R125 RaPO反馈：detach D=max(meanmask(behaviorTF−own_previous_stage_anchorTF),0)，ret=exp(-20D)，total=exactaccuracy+.5ret，query组内减mean再除持久sigma（初1，先.999sigma+.001std64total），clip.2/KL.04/gradclip1/Adam2e-5。全错不等PG0，ret差异可产生优势。行为/currentTF同micro2未更新ratio必须精确1。

## 四个必须完成的臂

BASE_TARS_RAPO：原骨干。
CE_AUX_TARS_RAPO：同骨干再加coefficient1.0的标准TARS-weighted canonical CE；8个同query标签，每batch一次合并梯度/optimizer，不额外更新，强完整答案监督而非弱系数控制。
POS_KD_TARS_RAPO：同骨干加coefficient1.0的正教师sampled-token partial-KD系数，sg(logp_teacher_positive−behaviorTFlogp)。这不是SDFT的EMA/full-vocab原样复现，明确称机制控制。
CONTRAST_TARS_RAPO主：同骨干加SIPO正/负教师证据sg(lp_pos−lp_neg)，κ=.2、δ=.2在任何新结果前一次选定、不搜索。每token先构造z=baseAdv+κclip(e,−δ,δ)，再按baseAdv符号投影：正取max(z,0)，负取min(z,0)，零保留z。连续RaPO奖励不满足作者二元1/G最小幅度推论，只依赖显式投影，不冒充作者原样算法或符号裕量证明。其余臂baseAdv扩展为token系数；POS_KD为baseAdv+sgΔ，不额外符号投影。所有PG含aux项乘同class TARS，KL从不乘，按8queries/64responses及每response有效token均值归一；CE同8query均值。监督与distillation收益不自动记纯RL必要性或创新。

## 教师/负例/信息边界

POS_KD与主分别保存自己periodic teacher snapshot，每global post update前g%10==0与当前actor同步，g跨任务累积不重置；checkpoint含teacher及last_sync_step。历史stage anchor是另一份自己上一task权重，不被teacher覆盖。teacher只评分当前生成的相同tokens，不再生成新文本，stopgrad；单独restore teacher再restore actor并核参数相同，不能改变ref buffers。teacher评估统一micro1，正负同形调用；不得把它与micro2 student的数值差假称bitexact。

正/负使用同一固定训练-only模板 `A reference answer is: {answer}. {student_question}`，只替换answer。positive是当前query canonical groundtruth完整类别答案，非独立临床解释/免费信息。negative：有效错误回答用它自己；正确回答取组内有效错误答案众数，平票最低canonical ID。有效须EOS完成、normalize后为当前已到达类别、且不等gold。无合法错误（全对、只有invalid或future-schema输出）不造负例；invalid response保留BASE/RaPO/KL但teacher项0，不丢query/重采。全错组只有可判定有效wrong response才可额外有teacher项。POS_KD对所有自己的sampled prefixes正教师评分，CE对全部gold canonical TF，不弱化两控制。BASE也获得同标签用于reward/TARS，不假称它没见标签。

只对科学训练保存远端私有normalized rollout text/done/parsed labels/negative IDs，以CPU独立核exact rewards、组与teacher eligibility/tie规则；不公开或fetch逐例输出。每类aggregate/teacher token/opportunities/实际L2/sigma与费用公开。未保存logp/逐类梯度就不补造variance/效果，teacher项非零不算学习或泛化证据。

## 原生准入与预算

工程stage0仅合法首task图，恢复合法COMMON prefix。原生processor tokenIDs exact、matchedhidden/同形head<=.05；rawhead/sampleTF差保留。同优化TF未更新ratio1。四真实RL/CEaux/KD/contrast反向optimizer路径与一条SFT-CE16 query工程路径，全部丢弃/还原前缀/anchor/teacher/sigma，不算新SFT。首task已见词+neutral Category padding覆盖最长学生/teacher schema，只CPU算未来长度不送未来词给模型。额外worst-case teacher profile：16个现有firsttask feature-row/tokens24的positive和negative TF各一次，每response micro1，实际32 teacher TF rows/0生成；只是容量/时间，不是正确奖励。

各工程RL2query路径计总wall及其中teacherwall；全矩阵forecast=1.5*[512*4*sum(各臂non_teacher_native2query_seconds)+512*4*(worst_positive_profile_seconds+worst_positive_profile_seconds+worst_negative_profile_seconds)+17548/16*greedy16_seconds+22812/16*greedy16_seconds]+1800。CE_AUX native路径已含2query CE梯度，SFT16工程路径只验证不用作科学SFT外推。每臂non_teacher计时仍含optimizer构建/参数步等成本，没有扣除启动开销赌更快；worst教师全行计费不按低eligibility减预算。完整forecast<原23h剩余才科学启动；失败保留，不扫κ/δ/周期/提示/seed/预算/臂救启动。

科学2048次post optimizer/131072 RL响应；TARS校准17548、VAL22812、native80=171512新生成响应，工程4TRAIN encoder/0新VALencoder。CEaux4096 teacherforced queries，positive teacher科学最多65536行（KD32768+主32768），negative最多32768行，nativeprofile额外32TF行，实际数/输入tokens/有效生成tokens/含pad处理tokens/wall/CPU全报。新curves2048，加历史306=fullchain2354，不能称2354新步。20stagearm表含4共享首task旧读出=17unique序列其中16新；288classrows/72current类group行。0科学旧TRAIN、0新model/featurecopies。private parsedoutputs/logs/checkpoints计账。

## 预先九门/完成与停止

主相对BASE四post平均macro>=+1pp/tail>=+2pp；最终首taskhead相对共同前缀drop<=1pp（完整菜单/参数全流程，不据R127调门）；任一post oldtail对BASE>=−1pp；主对CE_AUX与POS_KD各平均macro/tail>0（4门）；classwisepeak-finalforgetting少于BASE。全部macro/head/tail/tiny/old/new/micro/分母、每类组/teacher有效率、PG/KL/梯度范数/L2非零步、teacher同步和actualcost全报。不据中间VAL/reward改参、丢组、停臂，完整后决定。

单顺序开发VAL、共用历史前缀，非新独立seed或患者临床证据。R125原4/9、R126时间失败、R127条件菜单分解均保留，不宣称组合顶会创新。不追加SFT/seed/resolution、λCE/KD/κ/δ/teacher周期、解析器/提示长度或旧保护器救分。只有全部效能门过也只是构建资格，RL与辅助监督贡献分别评判。

整个研究新增物理<10GB，旧R114–R1277085854720B+本机200MB+新root预计<650MB，峰<8GB；2s独立guard计旧基线+本机余量+新root，在9GB/deadline−120/RSS48GB/free1GB预停，filecap256MiB/core0/CPU4/allocator56GiB。所有HF/Torch/CUDA/Triton/tmp/pycache同root计账；只pro5000/GPU1，中性main.py/argv，不my-gpu/不动他人/不删历史失败模型原数据。实际mount写读probe/GPU1free/RAM/native后启动。实际CONTROL_EXIT0后本R128 CPU-only finalize_saved核前缀/访问/seals/tapes/own连续optimizer/sigma/teacher周期/负例eligibility/数学/九门/2354链与2048新curves/288类行/所有成本，不重生成；只safe code/aggregate公开，显式7897proxy/NO_PROXY空/allowlist/no-force/remoteSHA+匿名200，runtime/postprocess分SHA。禁止私有索引/rollouttexts/predictions/ACCESS/weights/features/paths/图像/原日志/PDF/凭据公开或fetch。