# R95 BiomedCLIP上的RLAP机制主线诊断：冻结训练方案

2026-11-02。用户要求快速训练，采用刚通过D94真实双prompt接口检查的BiomedCLIP；不恢复R93/R91/D92。这里是RLAP医学骨干适配、非标准GRPO或最终创新已成立。原R88已闭合不重跑；历史LAION数字仅背景，因骨干、文本接口/参数、ownmemory选择不同不能声称纯骨干因果效应。

## 完整科学配置

原HK seed74002、23类13+2+2+2+2+2、TRAIN6821/最终VAL1698、0TEST；同原order/名字/template。三个必要arm FULL/CE_WEIGHT/UNIFORM各完整6stage×20epochs，不减步数/微批/对照。batch16/micro2，原AdamW lr1e−4/wd.01，无新增schedule/retention/macro项。正式预计31260optimizersteps、18最终task模型+optim端点。每arm同初始化/数据/顺序/标签/epochs；共享底座只读原Biomed pretrained，不接R67 SFT/LoRA或CPC权重。

完整继承R88论文重建的双prompt、每任务pv/pt、旧分支冻结、共享easy/hard专家/attention/policy、difficulty CE+微批对称对比loss+.01balance，FULL再+.1RLPO，CE_WEIGHT无RLPO但policy可CE学习，UNIFORM权重恒均匀。RLPO奖励owncos−maxwrongcos−.1proto-collapse、stopped batch标准化adv、−adv logw+.01KL、reference任务边界EMA.9，非采样oldratio/clippedGRPO。reference_uniform首任务，保存任务端点后更新reference。

每类oldmemory min20 distinct centroid近邻，冻结原始Biomed unprompted表示选，全部arms共享，tiny不复制；当前新TRAIN完整+旧仅memory。bank每epoch刷新当前branch；旧branch编码固定可精确缓存。单模板A photo of a {name}，不依据VAL换模板。候选类birth路由而非真label路由、projection/fusion等论文重建假设与R88保留；不冒称作者code。

Biomed视觉prompt4×768/BERT文本4×768，D94接口保持原pool/CLS/mask/proj；文本每branch比R88多1024参数如实披露，不能伪称严格容量相同。联合维度仍512，所有其他结构相同。文本尾4必须padding，不能截有效token；None真NOOP。

## 只做等价加速

1. 同一stage/arm旧branch参数与底座已冻结，预处理是确定性val-transform；对本stage合法trainids+memory编码一次、以FP32缓存旧视觉向量，class/index不变，保存private缓存不丢历史。只缓存冻结编码，不缓存更新中的新branch、experts、policy或原型。
2. 每stage旧text branch冻结输出只算一次；当前text在同一optimizer组内参数不变，每组编码一次。其输出detach为可积累梯度leaf，保留原micro2每次contrastive/CE/balance及实际末组缩放；微批反传后把合计leaf梯度一次反传到原text图。不融合成batch16对比损失。
3. 同optimizer组bank/policy不变，proto/RLPO只计算一次，proto用同类leaf累计classification梯度，再与FULL RLPO项合并反传；CE_WEIGHT/UNIFORM同原语义，无新reward。

这些是计算图/冻结编码复用，不改目标/样本/步数。先对FULL/CE_WEIGHT/UNIFORM三个arm真实TRAIN微组slow与fast输出/loss/完整参数梯度/一次AdamW更新核验，score/loss绝对误差<=2e−5，grad与updated params逐元素<=2e−4*(1+参考最大绝对值)。记录实际最大差，非bitwise承诺。数据预处理无Random、base eval、所有旧branch frozen为缓存前提，任一不成立禁缓存。不用早期视觉prefixcache。

## 科学门（继承R88）

全部18task端点seal后一次VAL，0TEST，不早评/选epoch/选arm。FULL对CE_WEIGHT与UNIFORM六stage平均宏/尾均正，>=5/6stage四项全正；最终FULL宏/尾对同预算boundedNCM均正；tiny1/2/12相对boundedNCM均非负且>=2正（zero1e−8pp），合取necessary gate。NCM_ALL信息预算额外诊断单列；完整micro/macro/tail、class/stage、old/new head/tail、正负及cost保留。均值正不绕final/tiny。原R67 SFT等历史协议不同，不宏微混比或归因GRPO。

## 工程/完整预算准入与硬边界

以D94最早实现start为24h原起点，包含该工程、等待、本轮实现/软件/native/profile/训练/分析/发布，不另重置；D94自身1800s工程已闭合，其费用纳入关联准备一次，不重复计。FIRST文件为权威。

GPU1only、64GiBRAM/14GiBfree准入、reserved<12GiB、worker28/group29每2s总30GB、CPU4/400000s/file1GiB、disk8GB+512MiB早停/每server<50GB、remote-home/NFSprobe/独占memorylock。模型只读既有资源，不下载复制底座、无rootspill/swap/quantization/动他人。

前置profile固定初始13active文本/最终23class2active且5old缓存两种最坏形状，batch16/micro2各两pass，固定第二pass，取最大完整step时间；first/second都计工程成本，不挑最快。fullstep含梯度和AdamW。完整投影=最坏step×31260×2 + 实测冻结前向每row最坏×所有cache/bank/eval/NCM合法row上界×2 +7200秒I/O/保存/分析余量。依所有stage真实row/memory预算计算cache与evaluate次数，GPUreserved必须实测合格；投影<原剩余−180s才正式fit，否则0science拒绝，不删控制/缩轮数/减安全救援。工程optimizer和真实forward单列，正式模型从原seed重新初始化，不用工程更新状态。

独立guard不依赖chat/SSH。先核receipt防duplicate launcher；formal failure先审原clock/所有成本/actualsteps/seal，不盲partialresume或重训。只有savedverify/analyze故障可saved-only修复。完整结束exit0/18sealed/TRAIN/EVAL/VERIFICATION/ANALYSIS/STATUS COMPLETE才完整交付；负门同样交付。安全ownsource/协议/全class-stage聚合和费用经本机显式7897proxy、原deliverygit/branch、exactremote/匿名REPORT200/source+localDELIVERY；私有图/逐例/ID/features/weights/optim/rawlogs/privatepaths不公开。唯一此chat30min monitor原位更新，不新建。
