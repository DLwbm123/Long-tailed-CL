# R125：累计已见类别接口的单次资格实验

本协议在新实现、native或科学模型调用前冻结。首次准备1794248541.2337618（R124已交付后资源检查），20小时硬截止1794320541.2337618；准备/工程/失败/重试/子进程共原钟，不延长、不重置。仅pro5000物理GPU1，累计整个研究窗口<10,000,000,000B，非按实验重置。R124已经完整结束并通过CPU-only独立验证，9fad51183de2ed614778ee072cbcd57b16f9b382已验证remoteSHA与匿名200。四RL共131072回答正确0/参数更新0，因此满足此前另立任务接口资格实验的条件；不撤销R124九门4/9失败、不把旧类不变称保留机制有效。

## 唯一接口因素与可否证问题

R124固定开放类别问法没有列已到达候选；RaPO原文附录B分类显式提供累计seen类别。本实验全5臂、所有SFT/校准/rollout/TF/anchor/VAL统一用固定模板：

What is the endoscopic finding or anatomical landmark shown in this image? Choose one category from the categories seen so far: {seen_categories}. Answer with the category name only.

seen_categories按canonical ID升序，以分号空格连接该阶段累计到达类别的official_class_name。不给当前例答案、未来类别、future图像/反馈，不提供class-count/正确率等提示；不是硬logit闭集约束，输出仍原manual采样/greedy，exact canonical+EOS评分不变。不得改模板/类别顺序/同义词/parser/max24/temperature/formatreward、SFTepoch/lr、RaPOalpha/lambda/sigma/beta或task顺序救分。

重新generic/taskLoRAB0、首任务同seed74005自然2epoch306updates建立共同起点，不复用R124/其他任务SFT。模板也用于首任务，故历史比较包含首任务接口和已学起点的变化，不是隔离post提示的单组件因果；首任务监督收益和接口本身收益不记RL。自身共同起点、普通GRPO、TARS、RAPO、TARS_RAPO主、匹配TARS_CE才用于同条件机制归因。已知方法适配服务主线，不宣称简单组合已创新。

## 其余科学协议保持

完全继承R124的固定到达0..4/5..9/10..14/15..18/19..22、TRAIN只当前task/历史样本0/future禁止、每阶段全部5模型seal后才审计seenVAL、各臂独立连续model/Adam/sigma、各阶段自己前policy anchor、当前TRAIN自己的greedy TARS表和mean-strength1/PG-only、每新任务128×8query/G8、CE匹配自身TARS权重但少rollout/无KL、同种子query12474006..009和response12480000+1000*stage+k、相同2e-5/KL.04/clip.2/max24、RaPO exp(-20D)/lambda.5/CTAN.999初1。core原九门完全不改（平均seen macro/tail对TARS、头类和oldtail保护、对RAPO/CE增量和forgetting）；完整5阶段后统一判定，不中途按结果改臂或停止。

首个新增任务逐RL臂单独报告correct sampled responses>0、mixed accuracy groups>0、实际非零参数步>0三项启动资格；通过仅说明非零学习机会，不是效能、保留、临床或创新。所有阶段逐类组组成/奖励/实际L2/attempt/old-new/micro/tiny及分母全报，不将错误retention梯度当有效任务反馈。若仍无可用反馈或原联合门未过，保存并关闭该单次接口变体，不扫prompt、追加seed或附加reasoning/监督救分。只有原联合门与匹配控制收益有证据才另立后继确认；无有效机制则记录依据，不盲跑。

## 动态提示与真实工程准入

每stage进入时绑定当前seen模板/token IDs/64视觉token/grid1x16x16，所有5臂共享。冻结视觉缓存只只读复用，图像信息不随提示变；生成、teacherforcing和reference必须用同一stage prompt。native首task合法两图原生processor/fullforward与手动cache注入samehidden/same-shape last head<=.05；记录rawhead与sampledTF差、不声称全bitexact。tokenizer64image-pad展开必须与原生processor逐ID相等。所有stage模板由scheduler metadata检查不含未来类，但真实模型只在该stage到达后看到对应seen名单。

native性能使用只包含首task已见词汇的重复padding，使prompt token长度至少覆盖所有未来stage模板长度（未来schema长度可由调度器知道；未来词汇不得送模型）。4RL/CE工程反向和16querygreedy在该保守长度计时；全部丢弃并还原generic/ref及正常首task prompt。不新增科学超参/训练查询。native允许合法全错零步，科学奖励资格另报告。完整306SFT+2048RL+512CE+校准21935+VAL29139的实测1.5margin+1800reserve须低于原钟剩余才启动科学SFT；否则保留成本、停止本次准入，不把预测当实际耗时。

正常科学/工程调用数仍为306SFT/4868queries+2560post（4RL2048步131072回答、CE512步4096teacherqueries）+21935校准+29139VAL+80native=182226生成响应；4TRAIN工程视觉编码/0新VAL编码，25表行含5共享首task、21unique读出、360类行、2866曲线。模型不见未来像素/历史重放，schema和到达元数据已知，static全TRAINcache不授权无限访问。单order开发VAL非独立患者。

## 累计资源与交付

R114–R124保留6358048768B+本机200MB+新root（含全部tmp/HF/Torch/CUDA/Triton/log/失败），预计新<650MB，保守全窗口<7.3GB、严格<10GB。2s独立后台guard在9GB、deadline-120s、RSS48GB、FSfree1GB预停；单文件256MiB/core0/CPU4/GPUallocator56GiB，actual/data挂载/读写probe、GPU1当前显存与原生准入必须核实。新模型/feature复制0，无需删除任何历史/他人文件。neutral入口argv不含身份/项目/方法信息，不干预他人。

退出0后本实验专属CPU-only saved-only finalize核动态提示、currentTRAIN/seenVAL/无历史重放、全部seals/independent连续state/optimizer/sigma/tapes/组数学/实际位移/9原门+启动资格/全部attempt费用与原钟。校准只存aggregate正确计数，不能补造逐例校准重评分。所有失败保留，原钟不重置。source-only安全公开GitHub显式7897代理/NO_PROXY空/allowlist/no-force，remoteSHA与匿名200；不抓取公开身份/图片/特征/权重/逐例预测/ACCESS/QUERY_TAPE/原日志/私有路径/凭据/PDF。只一个tars-rl心跳，R124及更旧PID不再轮询。
