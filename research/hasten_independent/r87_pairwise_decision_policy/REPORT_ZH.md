# R87：两步分类仲裁策略与GRPO额外价值

固定R81/R84保存CLS/NCM分数，双fold、双方向、3条件stream、六阶段；288policy snapshots/36864controlleroptimizersteps，0视觉优化/编码/VALTEST。分类动作不是生成，最多两个不重复竞争对的分数交换；真实类不在top2不能纠正。

GRPO特异必要门 **False**；全8项同正阶段 **3/6**；tiny门 **False**；最终期望与modal对NCM门 **False**。

|G减参照|heldout期望宏pp|heldout期望尾pp|
|---|---:|---:|
|U|+0.795298495|+0.461902856|
|CE|-0.202766856|+0.037047581|
|E|-0.233530421|-0.162659780|
|H|+0.811767015|+0.469249665|
|NCM|-0.671623814|-0.283160213|
|balanced|-0.930440628|-0.367578407|

U为同legal tree的祖先uniform，CE为同网络精确监督CE目标，E为同网络精确终点reward梯度，H为reward循环打乱GRPO。均同128step/stage，实际policy计算量不等；没有GPU效率声明。若只胜CE/U不胜E，只能支持目标差异，不能宣称GRPO特异收益。

原Q有历史开发暴露，本次heldout只指未进该方向policy loss的component；另一方向独立参数。class1fold1缺双slice，必要少类门仅fold0class1覆盖，未补独立证据。期望准确率是随机策略的精确期望，非实际独立正确图像数；modal为最大joint概率轨迹，不按测试reward选。全Q含反馈行仅描述。全部stage/fold/direction/stream/类方向完整保留，不挑方向、阶段或stream。

## 1. 核心结论：有奖励对齐信号，没有GRPO额外识别价值

G相对uniform与打乱奖励H的宏/尾增量均正，在两个方向都保留；说明在本固定任务和预算下，正确reward配对相对这些控制有条件作用。但是G相对同网络监督CE宏为负；相对精确奖励梯度E宏/尾均负。它还低于不作任何仲裁的原NCM，因此不能把“胜过随机修改”写成提升识别能力。

全局heldout期望增量（pp）：G-U +0.795298495/+0.461902856；G-H +0.811767015/+0.469249665；G-CE −0.202766856/+0.037047581；G-E −0.233530421/−0.162659780；G-NCM −0.671623814/−0.283160213；G-R84balanced −0.930440628/−0.367578407。仅3/6stage全部8项正（要求>=5）；正的是stage3/4/5，但它们仍低于NCM，不挑后期报告成功。

前向G-CE宏/尾−0.255830275/−0.019345858pp，反向−0.149703436/+0.093441019pp；两方向宏均未超过CE。相对E两个方向宏/尾均负。最终stage G-NCM期望宏/尾−0.003903433/−0.002371082pp，modal两项均0，未过预定最终门。

## 2. 不是“少类动作完全无效”，也不是已经获得少类稳定收益

|类|covered heldout cells|G-NCM期望recall pp|G-U|G-CE|G-E|G-H|
|---|---:|---:|---:|---:|---:|---:|
|1|36（仅fold0）|+1.603072501|−1.466038872|+0.393758074|+0.570245442|−1.447090620|
|2|72|+0.273839519|−0.006644283|+0.050676767|+0.117051997|−0.005830581|
|12|72|−0.096480634|−0.090672294|−0.328366155|−0.101386567|−0.085932356|

class1相对NCM在36个coveredcell期望recall均增，但比uniform更低；class2有20正/16负/36同，class12有16正/11负/45同（相对NCM）。class12均值负导致tiny门失败。期望recall是对随机策略所有轨迹精确积分，不是纠正了新的独立病例。class1fold1无双component覆盖，不填数也不作其可靠性声称。

全部23类的stage、full/heldout、各臂expected/modal方向保存在CLASS_STAGE_METRICS与CLASS_RESPONSES；全部144个stage/fold/direction/stream/query的宏指标保存在CELL_METRICS。旧头/旧尾/新头/新尾含空组的未定义值完整保留在OLD_NEW_HEAD_TAIL。fullQ含训练反馈，仅描述，不能当新泛化确认。

## 3. 实际策略行为：减少随机修改，并未形成确定性纠错

|臂|heldout第一步停止概率均值|期望swap次数|modal停止单元|
|---|---:|---:|---:|
|G|0.649636|0.645447|72/72|
|CE|0.284833|1.340777|72/72|
|E|0.713400|0.509861|72/72|
|H|0.014178|1.957279|72/72|
|U|0.015028|1.954734|72/72|

所有臂的最大joint概率轨迹在全部72个heldout及72个fullcell均为stop，modal分类等于NCM。短stop轨迹在uniform树中原本就更可能成为joint mode，因此不能仅由modalstop断言“学会拒绝”；G/E停止概率显著高于uniform这一描述才显示策略分布更保守。G减少随机修改，但其heldout期望utility相对NCM均值−0.0145435574，72/72cell没有正增量；E/CE也分别为−0.0083004433/−0.0118729800，均0/72正。

这不能唯一归因训练不足、KL、动作定义或输入特征不足。E是精确梯度、不是全局最优解证明；有限128steps/stage和固定uniform-KL是本次方案的一部分。G的组标准化与E全树标准化不同，不能称相同raw utility的无偏梯度比较。已提供同网络、同数据/合法树、同更新步数的强控制，但不宣称计算量相等或GRPO在所有任务普遍无用。原top2以下的真类不可救，且两方向/同阶段/三stream不是独立患者/模型seed。两动作顺序可能产生同一分类终点，轨迹数不是独立模型或独立证据数。

## 4. 完整执行、成本与保存

正式288个policy-stage快照（4learnedarms×6stage×2fold×2direction×3conditionalstream），每个128步，合计36864实际controlleroptimizersteps。G/H/CE/E各72快照、9216steps。共9216个group记录；所有policy+AdamW先seal，随后才评估该方向heldoutlabeloutcomes。核对训练/留出ID严格分离、轨迹概率与reward/count重建、AdamW累计stage步数，保存校验通过。源视觉/分类器不再forward，不作视觉优化或新图像编码，不读VAL/TEST。

使用已保存NCM/CLS分数，另做576次最终policy distribution评估：288heldout+288full；历史字段名heldout_policy_distribution_evaluations实际包含两类，按此披露。共享counterfactual class-trajectory算术评估7,784,805，不是这么多新图、独立端点或encoderforward。最大训练合法动作125、合法轨迹15295。全部精确概率、策略参数、优化器、train/eval action输入和outcome基底保留私有，没有删除科学端点。

|臂|fit CPU秒|fit wall秒|训练distribution评估次数（含最终）|
|---|---:|---:|---:|
|G|415.908796|104.561493|11592|
|H|415.547879|104.441343|11592|
|CE|365.169442|91.874836|9288|
|E|365.496055|91.993155|9288|

G本次没有CPU节省：其额外旧策略采样/回放工作使fit CPU高于CE/E。完整worker wall489.427577秒、CPU1646.321136秒；原clock到退出897.738868秒（含准备与所有软件检查）。worker RSS峰773754880B，2秒抽样group峰791343104B；新磁盘峰310234704B，低于事前6GBcap。CPU-only，CUDA_VISIBLE_DEVICES空，无GPU请求，0root/remote spill，无OSswap操作。

本机两次软件检查各5toycontrollersteps，CPU1.202659+1.112806秒；服务器5toy CPU2.621426秒（已含workerCPU），共15toy不混入正式36864步。首次本机pass后补了精确tie处理及fixture，发生于任何真实分数/native/科学拟合之前。saved-score native2真实TRAIN行CPU工程通过，不是pixel/GPU native。后置完整切片和行为描述仅读保存数据，CPU2.178510秒/wall1.560241秒/RSS399400960B，0新policy/encoder/classifierforward、优化或VAL/TEST。文档/准备/发布CPU未独立采样不是0。

RAM准入64GiB；worker28GB/group29GB每2秒、总30GB上限与原20h clock始终保留。磁盘初稿2GB在真实score/native/fit前按完整两步树最坏存储量修订为6GB（sum M²=695638，含NFS保守余量），未因中间效果改预算；入场recent28079098722B+6GB+512MiB<50GB。实际仅约310MB，不能把6GBcap说成实际占用。单attempt退出0，无科学失败、部分优化器恢复、重训或clock重置。

## 5. 主线判断与下一步边界

这次将动作直接绑定分类决策，并设置了同容量CE、精确reward梯度、uniform和打乱控制，已完成用户要求的GRPO额外价值检验；答案在本固定设定下为否。不是生成式VLM，不新增SFT预热；固定源分类器来自既有监督适配，控制器从uniform开始。RLAP-CLIP的RLPO只提供任务设计启发，没有被当成标准GRPO复现或成功保证。

不采用该仲裁策略作为主线性能提升，不强制非zero动作，不根据modalstop去掉stop/改KL/改部署温度，不选择后3stage，不扩大top-k或继续扫lr/horizon/动作幅度。普通CE/E的条件表现也不证明换成一个监督小技巧就达到投稿创新要求。R87是机制诊断，不是最终完整方法。

后继必须提供能在独立component上识别“何时改动确实比保持NCM好”的不同依据，并在相同监督控制下证明增量；仅有决策自由度、奖励非随机或损失下降都不够。它还需要保留不修改的可靠参照，不能靠削弱uniform或只比较随机有害修改制造成功。当前没有新run/FIRST，整体研究授权保留，但其他暂停分支未解禁；不把失败自动变成新一轮KL/温度/模块循环。

R80原最终SFT联合门失败、R81未超uniform、R82/R83识别门失败、R84/R85/R86的有限监督/信息收益与原门失败均保持。单HK74002反复开发、历史TRAIN查询暴露、episode-transductive无标签输入均披露；不是三新seed、独立患者、VAL/TEST或临床确认。

## 6. 公开边界

公开自有source/protocol/全部stage/class与双方向三stream的宏聚合、expected/modal、成本和正负解释；部署路径已替换为中性占位。图像身份、逐例数组、私有Q风险/margin、prototype/features、policy/optimizer权重及原日志不公开。完整数组和学习端点仍保存私有。RLAP-CLIP来源：[官方论文记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc65fab891d83433bd3c8d966edde311-Abstract-Conference.html)；用户提供PDF的RLPO方法式和表3作为定向设计参考，未声称对匿名代码的采样/stopgrad实现作过确认。
