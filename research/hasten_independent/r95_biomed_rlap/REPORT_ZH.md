# R95：BiomedCLIP版RLAP完整结果

目标端完整运行exit0，训练/评估/保存状态验证均完成；**预定效果门未通过**。北京时间2026-11-03 04:43:57结束，无活跃worker。不是工程失败，也不是科学成功。所有模型、优化器、缓存、逐例私有读出及原服务器中断历史均保留。

## 全部方法结果

单位为百分比。阶段平均是六阶段宏指标的算术平均，不是混合全部图像的准确率。macro与micro严格区分。

| 方法 | 平均micro | 平均macro | 平均tail | 最终micro | 最终macro | 最终tail |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 71.644966 | 52.970561 | 36.276680 | 64.664307 | 46.528690 | 30.027668 |
| CE_WEIGHT | 69.602748 | 49.758797 | 31.860091 | 56.419319 | 38.334171 | 22.249690 |
| UNIFORM | 73.020659 | 52.458821 | 32.736287 | 66.725555 | 47.719662 | 29.640313 |
| NCM | 70.651077 | 50.957004 | 33.248921 | 69.611305 | 49.184760 | 29.495421 |
| NCM_ALL | 73.421422 | 52.818489 | 33.733129 | 72.850410 | 51.422314 | 30.026540 |

## 结论与事前门

FULL对CE_WEIGHT的平均macro/tail为+3.211763/+4.416589个百分点；对UNIFORM为+0.511740/+3.540394个百分点，平均四项增益门通过。只有stage1/2共2/6阶段同时对两消融的宏/尾四项全正，未达5/6。最终FULL对UNIFORM宏/尾为−1.190972/+0.387355pp，对bounded NCM为−2.656070/+0.532247pp，对all-TRAIN NCM为−4.893625/+0.001128pp。最终宏优势不成立，不能只选CE_WEIGHT较弱对照来宣称整体有效。

极小类index1/2/12最终FULL、两个消融、NCM和NCM_ALL全部0/1、0/1、0/2；相对NCM没有至少两类正提升，tiny门失败。因此全局均值门true，但阶段稳定性、final-vs-NCM、tiny门false，necessary_gate=false。尾类均值提升不能遮盖这些负结果。

方法仍是论文RLAP的明确重建：Biomed原pretrained冻结底座、双Prompt、stage-specific旧分支冻结、prototype权重、easy/hard专家与attention、FULL加RLPO。不是官方作者代码或标准GRPO。FULL/CE_WEIGHT/UNIFORM各6阶段×20epochs，总31260正式更新/18完整模型与优化器端点。新全TRAIN、旧每类最多20distinct memory；任务reference EMA及全部模板/训练配置未按结果修改。

与已闭合LAION R88的绝对数值仅可描述性列示：原FULL平均macro54.233843/final48.472684/tail31.372347，本轮52.970561/46.528690/30.027668。骨干、BERT接口、每branch文本参数量、基于各骨干的memory选择及硬件框架均有差异，不能把跨run差异因果归RLPO或宣称医学骨干整体更优。

## 评估、校验与边界

六阶段全部计划端点先封存后统一VAL，最终1698个VAL实例，TEST0。原程序saved-state verification通过（18端点），本次仅依据已保存aggregate JSON重算全部class/stage/old/new/head/tail及冻结门，无新encoder/classifier/policy forward、optimizer或VAL/TEST图像读取。macro/tail重建最大误差2.84e-14pp；micro的FP32计数转百分比舍入最大差5.68e-06pp，未改读出路径或评估。不是又做了一次完整权重/逐图身份审计。

CLASS_STAGE.csv完整披露各阶段每类正确数与分母；GROUP_STAGE.csv给出old/new/head/tail及四交叉组，STAGE_METRICS.csv给出所有臂六阶段结果。所有正负结果保留，不选最佳epoch、阶段或臂。不含患者/病例身份、逐例scores、原图、features、权重、optimizer、rawlog或私有路径。

HK seed74002单开发种子，非多seed/患者独立/临床验证。原R67全历史BiomedLoRA SFT平均macro66.840041/finalmacro64.876782/tail44.924938仍是强历史参照，本轮未超过这些绝对数值；训练结构、memory与监督预算不同，不能因果归结GRPO或RLAP。TARS历史macro不能与任何RLAP micro错比。

## 迁移及全部成本

my-gpu原运行由用户要求停止，未评未完成部分；原代码、原环境、训练数据、完整已存阶段模型/优化器/日志全部保留，并复制静止快照到pro5000 data盘。目标从原种子完整重跑，**保留原FIRST/deadline，未因换机器重置24小时**。共享唯一空闲GPU6、实际只用1张，未超过最多2张的限制；所有大文件在data盘，未动其他进程/删除历史。

原端worker CPU15682.521896秒/wall12202.215576秒；目标worker CPU29353.416383秒/wall15842.982812秒；两worker CPU合计45035.938279秒。目标原clock到退出30711.315245秒（8.531小时），不能仅报目标worker时间作为整个实验成本。两服务器都发生了真实正式训练；原端未完成次数不与新端完整31260/7294步骤混成一条连续optimizer轨迹。

目标峰worker RSS2434789376字节/group2455887872字节，TRAIN_COMPLETE记录的GPUreserved峰2850029568字节，science实际占用1195536384字节，均在冻结界限内。CPU/native/profile/训练core已含对应worker，不能重复加；输入/缓存行是累计暴露，非独立病例。

源端/目标端native/profile均须分开保留并计工程暴露；原生CPU只作为worker内分解，不再加进worker总和。独立软件检查、controller、SSH/venv安装/资产提取/文档/发布CPU非零，其中部分未完整独立采样。COSTS.json包含完整已采样原端/目标端收据和计数。迁移报告保留8,519张TRAIN/VAL提取、TEST提取0、复用既有Biomed权重、Torch2.7.1 Blackwell相对原Torch2.1.2必要环境变化及全部准入过程，不宣称bitwise轨迹相同。

关联D94准备workerCPU27.062167秒另计一次，因此两训练worker加已知D94准备共45063.000446 CPU秒；D94core含于D94worker，不能重复加。其他未独立采样工程开销不是0。

## 当前结案

本轮完整负门结果保存交付，不按VAL追加调参、选择臂或训练轮数救援，也不自动把简单修改包装为主线创新。当前两项均无活跃训练；用户已取消定时监测，保持取消，未创建新monitor。必要独立对照仍未运行；本次没有启动后继实验。后续若推进，先针对已明确的主线问题冻结有区别的机制/合法反馈/完整预算，不能以重复旧reward/selector扫描替代研究。
