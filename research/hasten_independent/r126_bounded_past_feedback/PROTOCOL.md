# R126：有界真实旧任务反馈的机制资格实验

本协议在新实现/native/科学模型调用前冻结。首次准备原钟1794271927.5872638已包括上一轮来源和候选准备，20h硬截止1794343927.5872638（严格<24h）；准备、CPU、native、失败/重试、全部子进程共享，不重置。仅GPU1/pro5000，累计整个窗口<10,000,000,000B。R125完成验证和安全交付de537a799909ce84eafcb04a0ff1694275338567，4/9失败门不变，不修改任何旧模型/协议/结果，不轮询旧PID。

## 问题、用途与控制

R125中真实RL保留配方明显优于无保留RL，但首task head仍损41.4883pp；加TARS到RaPO未获平均净增量。下一诊断检验：在同旧图像/同生成预算下，把真实旧任务正确奖励用于PG，能否超越只在旧图像上维持旧policy的代理，并不牺牲尾类/新类？CLEAR提供真实旧经验反馈先例，但这里重新采样on-policy回答，非其V-trace旧轨迹算法复现；GEM/旧投影不加入。这是主线机制资格诊断，非最终创新方法，不学习R113记忆编辑actor或复活R105全候选路由。

五臂：CURRENT_TARS_RAPO（8current、无旧访问）、OLD_PROXY（4current+4old，old准确率不进入PG）、OLD_TASK_TARS_RAPO（主；同4+4，old准确率进入PG）、OLD_TASK_RAPO（同4+4、不加TARS）、OLD_TASK_CE（同4+4、当前TARS权重的监督CE，oldunit）。OLD_PROXY与主才是同图/同预算的真实旧正确反馈对照；CURRENT改变曝光分布，只作诊断参考。所有模型都可能从已到达数据保存历史标签；不能把OLD_PROXY说成从未获得旧标签，区别是优化信号。CE得到答案形式监督，少G8/KL/前向，不能宣称严格同反馈或等计算/RL必要性。

## 到达与有界旧证据

固定canonical ID0..4/5..9/10..14/15..18/19..22，5+5+5+4+4；scheduler已知schema/manifest元数据，模型不接触未来图像/feature/反馈/目标。generic Qwen2.5VL3B、taskLoRAB0、224/64tokens/frozenvision/qvLoRAr8alpha16、seed74005，不载R125或其他任务SFT。仅首task2434TRAIN自然2epoch/4868queries/batch16micro2/306updates，AdamW2e-4wd0clip1/cosine306共同支持；贡献不记RL。其余task无额外SFT。

每class实际到达、全部当前task模型seal后、该task任何VAL之前，用独立CPUgenerator12654005+c对该class当前TRAIN索引randperm，保存first2组件（若不足2则全部；现manifest各类>=2），不看模型、正确率、VAL或任何未来像素。冻结视觉feature从**当前**TRAIN合法读取复制到CPU BF16 birth shard，连同ID/label保留；全类最多46行/12058624B本体，torch/索引开销另计。birth-shard各类只写一次，不重复全历史snapshot。所有臂共享固定外生buffer选择/冻结视觉输入，各臂model/Adam/sigma独立，绝不共享策略反馈。

原完整TRAINcache仅作为current-only只读后端；旧输入必须从自有有界feature shards/RAM读取，严禁从原TRAIN mmap读取旧/未来行。accessor分别检查TRAIN当前class、BUFFER selected且birth_stage<current且arm非CURRENT、VAL仅seen并在全部5臂seal后独立审计。记录私有访问step/source/ID，buffer读不算新encoder但计历史访问；buffer复制46行明确计新存储，原cache创建费用不当免费。模型输入是单图+累计seen类名，无跨图免费窥视/无限历史访问。

## 查询、随机流与连续状态

每newtask128attempt、8query；4RL G8，CE同query/attempt但少前向。current完整128x8自然replacement tape用12474005+stage；CURRENT使用8列，mixed臂用前4列。old128x4 tape用12674005+stage，先uniform已到达old类、再uniform该类两个buffer组件。所有mixed臂共享old tape与current前4，动作回答fresh on-policy采样。response12480000+1000*stage+k同RL，无额外重采/滤组。old均匀类采样是显式分布变化，不能把对CURRENT差异单独归因old奖励。

各臂首task同共同SFT；post AdamW仅task2创建一次lr2e-5wd0，之后自己的momentum/step128/256/384/512与sigma连续。每stage anchor/ref为自己上task终点，在同当前seen prompt下评分。共同prompt同R125，类别官方名按冻结Engine hyphen→space规范化、canonical ID升序分号连接，max24/manualtemp1/正常EOS/exact canonical，无hard mask/reasoning/format奖励。未来类别只CPU scheduler算最长长度，不送模型。所有校准/生成/TF/anchor/VAL用相同stage绑定。

每臂任务初始仅对完整**当前TRAIN**自己policygreedy一次，standard rarity*error^.75/medianfloor1e-8/min1再除当前自然均值Z。CURRENT/OLD_PROXY/main/CE只给current slot这个TARS权重；旧slot预固定1，OLD_TASK_RAPO全部1。old校准正确率、old诊断反馈不能回流到PROXY权重/选择或任何控制；不从old重拟TARS表。自然current的平均权重1、old固定1，mixed期望总强度1，无事后权重扫描。

## 精确奖励与归一化

actual r_acc是每query G8回答的exact类别正确+EOS。used r_acc=actual，但OLD_PROXY的old4query used r_acc恒0，必须在组内减均值和sigma更新之前屏蔽；保留actual仅供隔离日志，不入损失、权重或采样。两者oldlabel在历史中可得，但改变old正确性不会改变PROXY的给定轨迹surrogate梯度；CPU fixture验证。混合臂记录actual与used成功数、current/old角色，不能把无old准确率PG称无任何old信息。

所有RL均RaPO：D=max(meanmask(behaviorTF LP−anchorTF LP),0)，detach ret=exp(-20D)，total=used_acc+.5ret，按query组内减mean，除跨task CTANsigma（初1；先.999sigma+.001std(64total)；epsilon1e-6）。与R125相同clip.2/KLbeta.04/gradclip1。TARS只乘PG不乘KL；loss与CE都按全部8query/64response归一，old PROXY并不删除old KL/retention项，不另增当前PG强度。恒定组合法零优势；准确率全错也可有retention差异，非零PG不自动有用。behavior与currentTF同micro2初始ratio必须1，samplingTF BF16差如实披露。

## 事先九门与报告

新九门只用于本R126，不替代或撤销R125失败：main相对OLD_PROXY四post平均seenmacro>=+1pp/tail>=+2pp；最终首task head相对共同SFT损失<=1pp；任何post oldtail对OLD_PROXY损失<=1pp；相对OLD_TASK_RAPO与OLD_TASK_CE各平均macro/tail>0（4门）；classwise peak-finalforgetting少于OLD_PROXY。CURRENT差异、所有old/new/head/tail/tiny/micro/分母、actual/used reward机会、每类组组成、drift/sigma、真实L2非零步和成本完整报告，不额外造事后门/挑阶段。全部5臂stage模型seal后才统一seenVAL，首task只读一次共享到5臂，不据中间reward/VAL改数据/提示/参数/臂或停止。

若PROXY旧图本身已解决而old正确奖励无增量，报告该归因；若仅普通replay/CE有效，不包装为RL必要或创新；若仍头/尾失败，不扫buffer容量/比例、lambda/alpha/sigma/beta、提示、seed或SFT救分。只从完整证据决定合理后继。单开发order/小tiny分母非独立患者或临床安全。

## 工程、全部成本与停止

CPU fixtures验证当前/旧/未来访问拒绝、selected-old buffer独立存取、每类2/46容量、old-correctness屏蔽的PG和EMA不泄漏、全错合法零组/TARSmean1。native仅首task合法两图：processor tokenIDs exact、samehidden/same形head<=.05，rawhead/sampleTF差保留；4RL和CE真实反向/optimizer，OLD_PROXY等old角色用首task合法数据上的**工程模拟role flag**检验，非真实旧任务或科学学习证据，全部丢弃还原generic/ref。CPU buffer fixtures与随后真实stage访问记录验证历史接口，不伪称native已见过去task。profile使用首task词汇+neutralpadding覆盖最长schema257tokens，1.5全矩阵+1800reserve实测forecast<原钟余量才科学启动。

正常仍306SFT+2560post=2866curve/360classrows/25stagearm含5共享首task=21unique readout。4RL2048steps/131072responses，CE512/4096teacherqueries；post总current12288query、BUFFER8192query，calibration21935当前响应、VAL29139、native80=182226生成，4TRAINnative encoder/0新VALencoder。另buffer-copy46行/实际文件字节，非新视觉编码。GROUP诊断219个stage×RL×class/role行（CURRENT仅current18，3mixed各seen67）。所有科学尝试与实际非零步分开，bufferrole日志与used/actual奖励分开；所有失败/CPU软件/native/postprocess费用保存，未知计时标未知非0，不删账/重置钟。

R114–R125旧根6905499648B+本机200MB+新root预计<700MB，窗口峰值保守<7.9GB且严格<10GB。独立后台2s guard含全部旧根+本机余量+新root，在9GB/deadline−120s/RSS48GB/freeFS1GB预停；filecap256MiB/core0/CPU4/GPUallocator56GiB，真实/data ext4/读写probe、GPU1实际空余和RAM必须准入。HF/Torch/CUDA/Triton/tmp/pycache全部新root计账。只有GPU1/pro5000、不my-gpu/不动他人，中性入口argv，无新模型或整份cache复制，不删除历史/失败/原数据。

实际退出0后本R126 CPU-only saved-only finalize，核bufferbirth/46shardrows/selected旧访问/current/seenVAL/全seals、共享tapes/独立连续state/role-mask奖励数学、所有计数/九门/费用，生成safe聚合报告并source-only公开GitHub显式7897proxy/NO_PROXY空/allowlist/no-force，remoteSHA与匿名200。禁止身份/图像/特征/权重/私有索引/逐例预测/ACCESS/QUERY_TAPE/BUFFER私有shards/原日志/私有路径/PDF/凭据。故障只在核身份后根因修复，保存源码/全部attempt/原clock，checkpoint可恢复不盲覆盖。仍仅原tars-rl心跳。
