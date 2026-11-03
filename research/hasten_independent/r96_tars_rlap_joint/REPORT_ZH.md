# R96：TARS–RLAP联合主方法完整结果

目标端完整运行exit0，训练/评估/保存状态验证均完成；**预定效果门未通过**。北京时间2026-11-03 01:30:48结束，无活跃worker。不是工程失败，也不是科学成功。所有模型、优化器、缓存、逐例私有读出及原服务器中断历史均保留。

## 全部方法结果

单位为百分比。阶段平均是六阶段宏指标的算术平均，不是混合全部图像的准确率。macro与micro严格区分。

| 方法 | 平均micro | 平均macro | 平均tail | 最终micro | 最终macro | 最终tail |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 69.973442 | 50.831865 | 33.672476 | 58.303886 | 44.129909 | 32.483562 |
| JOINT | 69.090665 | 51.852207 | 36.366268 | 61.660778 | 44.076862 | 27.784529 |
| NCM | 70.651077 | 50.957004 | 33.248921 | 69.611305 | 49.184760 | 29.495421 |
| NCM_ALL | 73.421422 | 52.818489 | 33.733129 | 72.850410 | 51.422314 | 30.026540 |

## 联合主方法是否有效

联合方法相对同次监督起点：平均macro+1.020342pp、平均tail+2.693791pp，但平均micro−0.882778pp；最终macro−0.053047pp、最终tail−4.699033pp，最终micro+3.356892pp。只有3/6阶段macro/tail同时非负，要求至少4/6。平均宏+.5pp及平均tail不降通过；最终宏+.5pp、最终tail+1pp和阶段稳定性门失败，performance_pilot_gate=false。

index1/2/12最终SFT与JOINT均为0/1、0/1、0/2；新增原正确病例损害0，但原先没有正确病例，因此该tiny保护通过是空基准，既没有rescue，也不能声称修好了tiny。最终joint macro44.076862/tail27.784529亦低于bounded NCM49.184760/29.495421及all-TRAIN NCM51.422314/30.026540。

## 真正GRPO及信用统计

结构融合RLAP双Prompt/可学习原型权重/难度专家attention，去除RLPO；每阶段先10epoch监督，然后copy做2epoch真正categorical G8 GRPO，两次optimizer更新共享固定行为logits/actions/advantages。TARS gamma.75的类别权重只乘surrogate，不乘KL.04；另有.01routing balance。没有把连续prototype权重目标改称GRPO。5210监督+2084GRPO=7294正式更新，6监督参考+6联合主方法端点；RL delta不carry到下一阶段监督训练，属于stagewise posttraining。

区分三个层级：1042个optimizer batch rollouts，每个batch内逐query独立G8；共16588个真正采样的query-level G8 groups和132704个采样动作。两次更新使用同一批动作，因此33176个query-group update exposures和265408个action update exposures不是新独立采样。后者中26460组为零优势（79.756450%），裁剪动作更新暴露16次（0.006028%）。这提示二值奖励下更新信号稀疏，**不是失败病因的因果证明**，不能据此追加reward/G/LR搜索救援。详见GRPO_STAGE_STATS.csv。

TARS权重的错误率来源是合法stage TRAIN query的固定SFT读出，bank和query可重合，非原R67的crossfold信号。原始类计数是元数据，旧类实际仅memory replay。原型bank每epoch刷新，固定SFT reference和TARS权重不跟随更新；GRPO后的修改不进入下阶段SFT carry。这些信息预算和阶段语义均需披露。

按用户指令只先跑联合主方法，消融/普通GRPO/exact/错配等独立对照后置。SFT是必需KL参考和同次前后读出，不是新增独立训练对照臂。当前结果既未通过完整效果门，也不能隔离TARS、适配结构或随机GRPO各自贡献，更不能由均值正就声称顶会创新成立。

## 评估、校验与边界

六阶段全部计划端点先封存后统一VAL，最终1698个VAL实例，TEST0。原程序saved-state verification通过（12端点），本次仅依据已保存aggregate JSON重算全部class/stage/old/new/head/tail及冻结门，无新encoder/classifier/policy forward、optimizer或VAL/TEST图像读取。macro/tail重建最大误差1.42e-14pp；micro的FP32计数转百分比舍入最大差5.34e-06pp，未改读出路径或评估。不是又做了一次完整权重/逐图身份审计。

CLASS_STAGE.csv完整披露各阶段每类正确数与分母；GROUP_STAGE.csv给出old/new/head/tail及四交叉组，STAGE_METRICS.csv给出所有臂六阶段结果。所有正负结果保留，不选最佳epoch、阶段或臂。不含患者/病例身份、逐例scores、原图、features、权重、optimizer、rawlog或私有路径。

HK seed74002单开发种子，非多seed/患者独立/临床验证。原R67全历史BiomedLoRA SFT平均macro66.840041/finalmacro64.876782/tail44.924938仍是强历史参照，本轮未超过这些绝对数值；训练结构、memory与监督预算不同，不能因果归结GRPO或RLAP。TARS历史macro不能与任何RLAP micro错比。

## 迁移及全部成本

my-gpu原运行由用户要求停止，未评未完成部分；原代码、原环境、训练数据、完整已存阶段模型/优化器/日志全部保留，并复制静止快照到pro5000 data盘。目标从原种子完整重跑，**保留原FIRST/deadline，未因换机器重置24小时**。共享唯一空闲GPU6、实际只用1张，未超过最多2张的限制；所有大文件在data盘，未动其他进程/删除历史。

原端worker CPU5783.782177秒/wall4436.386617秒；目标worker CPU7883.400993秒/wall4253.567836秒；两worker CPU合计13667.183170秒。目标原clock到退出11171.283068秒（3.103小时），不能仅报目标worker时间作为整个实验成本。两服务器都发生了真实正式训练；原端未完成次数不与新端完整31260/7294步骤混成一条连续optimizer轨迹。

目标峰worker RSS2475732992字节/group2480164864字节，TRAIN_COMPLETE记录的GPUreserved峰2820669440字节，science实际占用600481792字节，均在冻结界限内。CPU/native/profile/训练core已含对应worker，不能重复加；输入/缓存行是累计暴露，非独立病例。

源端/目标端native/profile均须分开保留并计工程暴露；原生CPU只作为worker内分解，不再加进worker总和。独立软件检查、controller、SSH/venv安装/资产提取/文档/发布CPU非零，其中部分未完整独立采样。COSTS.json包含完整已采样原端/目标端收据和计数。迁移报告保留8,519张TRAIN/VAL提取、TEST提取0、复用既有Biomed权重、Torch2.7.1 Blackwell相对原Torch2.1.2必要环境变化及全部准入过程，不宣称bitwise轨迹相同。

另有原排队controllerCPU0.15秒、原正式controllerCPU12.173647898秒、目标controllerCPU154.833175101秒；原端外部软件coreCPU0.522055684秒另列（其导入未独立全采样），新端软件检查已包含目标worker。所有工程参数丢弃后才正式初始化。

## 当前结案

本轮完整负门结果保存交付，不按VAL追加调参、选择臂或训练轮数救援，也不自动把简单修改包装为主线创新。当前两项均无活跃训练；用户已取消定时监测，保持取消，未创建新monitor。必要独立对照仍未运行；本次没有启动后继实验。后续若推进，先针对已明确的主线问题冻结有区别的机制/合法反馈/完整预算，不能以重复旧reward/selector扫描替代研究。
