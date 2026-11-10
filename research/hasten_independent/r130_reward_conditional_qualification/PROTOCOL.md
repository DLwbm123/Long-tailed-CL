# R130：RaPO目标分解与旧条件关系资格（非完整continual）

一次固定首新增任务0..4→5..9的机制资格。继承最早准备1794310608.4689498及23h截止1794393408.4689498，不改名重置钟。原R129未运行草稿封存，不运行旧worker；不复活R126旧buffer问题。新问题是无旧图时，旧类内部关系约束能否比完整候选KL和匹配CE更少阻碍新类获得。不能先假定新图旧policy目标有用，诊断不是最终顶会方法。

## 信息、起点、动作与访问

唯一R125 generic/B0首task自然SFT前缀（历史306步4868query/624VAL另计），所有臂相同起点；没有新SFT/支持样本/答案teacher。真实schema只绑定首新增task的累计10类，原固定模板/hyphen→space/canonical+EOS/temp1/max24；无prompt/顺序/格式搜索、硬mask或reasoning。图像编码器和generic共享cache只读，learner仅当前5..9 TRAIN行，不许旧或未来行，VAL只封存后审计。动作是原VLM完整自由生成并优化LoRA policy，无memory actor或旧buffer。单转换不伪称完整5task连续训练。

## 固定五臂与损失

A RAPO: accuracy+.5retention，组中心化后除own CTAN；B ACC_CTAN:同own totalreward统计更新CTAN，但PG只accuracy中心化/该sigma。A/B每臂独立model/Adam/sigma，不能互用状态。它隔离保留PG的条件贡献，不单独识别CTAN全部效应，B≈A不证明等价或全收益来自CTAN。

C ACC_FULL_KL=B+全10类候选KL(q||p)；D ACC_OLD_COND=B+qO KL(q(.|old5)||p(.|old5))；E CE_OLD_COND=强CE1+同通用rollout KL.04+同D项。E仍G8生成，用于同样retention/CTAN诊断及rollout KL，不把PG加回E；CE为gold tokens含EOS每序列平均后query均值。C/D/E相同完整候选打分和每个候选反向路径，包括D/E新类零系数；有意义梯度强度不保证相同，E额外CE费用单列。无TARS校准，查询自然不重采。

q为冻结首task前缀在相同当前10类prompt的完整canonical+EOS序列sum logp，经temperature1候选归一化；qO=全部10候选中旧5总mass并detach。所有q detach，student p可微；old条件不约束old/new质量。显式regularizer系数1，与PG/KL合并一次AdamW更新；不把query常数KL当group奖励。有限候选概率不是full free-generation语义概率，不约束非法/别名字符串质量。q不看current gold或VAL；CE拿相同current gold，信息监督差异明确。

各臂32updates×8自然query/G8，同seed12474006 tape前32batch；response12481000+k，模型分化后响应不假定逐例相同。AdamW2e-5/wd0/clip.2/tokenKL.04/gradclip1，各臂新optimizer是单转换资格起点，不冒充R125状态延续。CTAN初1，先.999sigma+.001std64(accuracy+.5ret)；ret=exp(-20*max(meanmask(behaviorTF-anchorTF),0))detach。正确比错误totalreward至少.5仍可能有错误正优势，全错仍可能retPG。参考为该臂同初始前缀/同当前prompt，固定本转换。

## 预固定有限测量（先于资格，但不能按测量改五臂）

仅transition1。菜单×参数2×2用R125 COMMON前缀和RAPO stage1保存权重，对相同624旧VAL、batch16，原stage0/1菜单F00/F01/F10/F11全新同批读出；训练和选择均不读结果。保留交互F11−F10−F01+F00，不称唯一因果份额，不以旧菜单部署择优/改原head门。回顾性已到达后读取，不谎称前缀训练时见未来schema。

TRAIN梯度探针用固定R125 RAPO stage1保存权重和own stage0 anchor、其保存Adam/sigma；当前自然tape第0batch8query/G8、独立seed13081000，只采一次64响应。不按混合/全错重采。保存原始token/mask/text/done/accuracy/h/total/CTAN、分解的accuracy PG、retention PG、token KL梯度norm及两两cosine；另用R125保存的当前task TARS表作诊断性PG权重（不是RAPO训练表，也不进五臂），记录加权梯度。组合梯度裁剪前后norm和一次带原Adam状态的counterfactual实际L2位移，更新全部丢弃、恢复检查点/anchor；属于测量optimizer成本，不作科学学习。没有旧token/grad的事后补造。

## 评估、资格门和停止

五臂全部32步seal后才VAL，另先封存一个不可训练前缀并在当前菜单读一次；共6×964=5784responses。所有macro/head/tail/tiny/old5/new5/micro正确整数分母全部报，class peak与原九门不在此短实验伪判完整通过。原R125/R128失败门保持。

D进入完整CL候选须同时：D−C seenmacro>=1pp且tail>=2pp；D oldmacro>=C且D newmacro>=B−1pp；D−E macro>0且tail>0；全部5臂32实际非零步、合法访问和状态/计数核过。任何一门失败关闭此固定配方，不改系数/seed/updates/G/菜单或tail类；通过也只是单转换开发资格，之后完整5task128/G8全关键控制仍要新冻结/真实准入，不能根据短VAL缩控制。A/B差全量报告，不设事后等价门。tiny很小组件分母不作患者独立/临床证据，不以阶段峰值当稳定保留。

## 全部预算和真实准入

正常生成18680=10240资格rollouts+64探针+2496菜单+5784资格VAL+96native；4TRAINnative编码/0VAL新编码。160科学optimizer+5native丢弃+1probe丢弃=166；E科学CE256query。C/D/E科学anchor/eval/backward各7680候选TF行；nativeprofile、探针组件重算和反事实更新所有TF/input/decodecells按kind计费。无校准/新大模型feature副本，历史SFT费用不免费。firstprep包含来源/分析/实现/CPU/native/失败/重试/子进程共原钟。native只首task词，neutral Category覆盖最长当前stage1 prompt，5真实梯度/optimizer路径丢弃恢复，额外候选最坏答案长profile；同形hidden/head<=.05保留raw fullhead和采样TF误差，不放宽。按各臂2query路径外推×4×32，加nativeprofile额外probe/菜单/VAL全部费用，乘1.5再+1800reserve<原remaining才科学启动。否则保存时间拒绝不重跑赌更快，不进入任何科学训练/诊断新读出。

闭合R114–R1287605948416B+本机200MB+新root预计<350MB，全窗口严格<10GB，不删历史。独立2s guard9GB/deadline−120/RSS48GB/free1GB预停；file256MiB/core0/CPU4/allocator56GiB，实际/data ext4写读probe/RAM/GPU1free检查。仅pro5000物理GPU1，中性argv，不碰my-gpu和他人。所有cache/tmp/pycache计新root。退出0后仅本次CPU-only保存审计和整数重算，不再生成；全部失败/未知CPU不填0。公开仅source/安全聚合，不取私有逐例/索引/模型/feature/路径/原日志。
