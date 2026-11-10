# R129 冻结协议：生成式旧语义分布归一化资格

假设：旧policy在当前图像上的canonical答案关系可形成不同于当前完整答案teacher的保留目标；按旧任务归一化相对全部旧类归一化，可能减少已有任务间偏差固化。并不保证旧图准确率或跨任务质量，必须全阶段实测。不是SS-IL/COPR完整复现、RL必要性或最终创新。只改变训练目标，不改变学生prompt/候选顺序/部署解码/输出长度。

四臂：BASE_TARS_RAPO；SEM_GKD_TARS_RAPO；SEM_TKD_TARS_RAPO(main)；SEM_TKD_TARS_CE。三RL均原TARS+RaPO/G8；CE同current8query/128updates和自身TARS表、同TKD与同anchor信息，但没有G8/KL，计算更少，强teacher监督控制不能省略。

所有臂复用唯一R125 generic/B0首task0..4合法SFT前缀，无新SFT/旧任务后SFT；历史306/4868queries/624VAL单列。arrival固定5+5+5+4+4。fullcache只读后端：TRAIN每次仅当前任务；旧/未来TRAIN禁止；无旧buffer；VAL仅seen且全4臂该stage seal后。scheduler知道manifest/schema，不向模型给未来图像/反馈。首task旧VAL共享4行，新增stage4×4读出；20表行17unique/16新，288classrows。

每新task128optimizer尝试/8current自然有放回query，同tape与response seed沿R125；独立连续AdamW lr2e-5wd0/clip.2/KL.04/gradclip1，step128/256/384/512不重置。RL sigma1起/.999EMA跨task，accuracy+.5exp(-20max(meanmask behaviorTF-anchorTF,0))先于组均值，teacher不进sigma。每臂currentTRAIN greedy一次标准rarity×error^.75/medianfloor1e-8/min1/naturalmean1，仅乘PG或CE，不乘KL/KD。

语义分数S(c,x)=sum logπ(name_c+EOS|x,current_seen_prompt)，无长度归一化、无硬mask、无gold前缀。anchor为自己上一task末policy，与当前prompt相同。三个KD臂到达时只用currentTRAIN once计算全部old canonical答案S并私有保存，禁止旧图/未来类；不重算随训练teacher。GKD在全部旧类单一softmax；TKD分别在每个旧task类别组softmax后对任务取均值（明示不同于论文求和，保持任务总强度不随数目增加）。温度2/系数1/乘T²=4，一次固定不扫。损失mean_x mean_group KL(q_anchor||p_actor)。当前gold只用于RLaccuracy/CE，不参与softtarget。旧类名是已到达schema元数据，不当旧图真实准确率反馈。

内存安全的精确一阶KD：先no_grad计算actor S，解析得到dL/dS=T*(p-q)/(batch*n_groups)，再相同micro2 TF逐块回传sum sg(dL/dS)*S，最后与RL/CE梯度共同一次optimizer步。该向量是标准KL一阶梯度，不是近似奖励/teacher采样；CPU与真实native检查梯度等价、actor前后logp一致、anchor不改。所有TF kind/rows/prefill+padding token cells与active response tokens实记。首个post仅一个oldtask，GKD与TKD完全相同，必须stage1权重和逐例VAL一致，不称两个独立机制胜负。

九门：main对BASE四post平均seenmacro>=+1pp/tail>=+2pp；最终首taskhead对共同前缀drop<=1pp（全菜单流程，R127不换门）；任一post oldtail对BASE>=-1pp；对GKD/强CE各平均macro/tail>0四门；classwise peak-finalforget少于BASE。所有类/组/oldnew/head/tail/tiny/micro/正确响应/非零L2/漂移/sigma/费用完整。单order旧seed共享前缀是开发条件，不患者独立。负不扫温度/λ/组划分/seed/提示/格式/长度/额外SFT/旧保护阈值救分。

science：3RL×512×64=98304生成；校准17548/VAL22812；native64，合138728，CE科学4096teacherqueries。语义anchor TF science131694 rows；actor no_grad与backward各150528；native每kind额外68 rows（3aux路径各2图×5答案=30，最长19答案×2图profile38）。TF不算生成但全费用/时耗入预算，完整encoder native4、VAL新0。2048新curve+历史306=2354chain，RL类group54行。

firstprep1794310608.4689498（含来源/入口工作）/独立23h deadline1794393408.4689498，绝不重置/超过24h上界1794397008.4689498。实际GPU1native分别测3RL和CE反向、semantic anchor/actor eval/backward最坏19答案×2图（合法首task词+neutralpadding覆盖最长schema/答案长度，不给未来词），全部工程更新丢弃恢复前缀。完整预算=1.5*(512×4×sum四臂2query非semantic秒 +131694×anchor38行秒/38 +150528×(actorEval38+actorBackward38秒)/38 +(17548+22812)/16×greedy16秒)+1800，须严格小于原remaining才science。绝不缩G/update/删控制或放宽margin。未来长度仅CPU tokenize；native来自首task合法TRAIN。

全窗口闭合基线7605948416B+本机200MB+新root预计<650MB，全峰<8.5GB/严格<10GB。2s独立9GB/deadline-120/RSS48GB/free1GB预停，filecap256MiB/core0/CPU4/allocator56GiB；实际/data挂载写读probe/GPU1free/RAM/native准入，仅pro5000/GPU1/中性argv，不my-gpu/他人/删历史失败数据。所有新cache/tmp/pycache计root。正常退出0后仅本轮CPU saved-only finalizer独立核访问/seals/tapes/语义targets/KD数学/首stage身份/own状态/九门/全部费用，0新生成；失败保留源码日志成本原钟，先诊断不盲覆盖。

source-only GitHub显式7897proxy/NO_PROXY空/精确allowlist/no-force/remoteSHA+匿名200；私有语义target、索引、逐例rollouts/VAL预测、ACCESS、STATE/权重/图像/features/paths不fetch/public。
