# 快照守卫生产接线与 CPU 验收：前瞻冻结

本轮只完成生产入口和 CPU 检查，禁止 GPU、医学图像/checkpoint、val 读取或调参；test/reserved 封存，RL暂停。不启动协调器/正式训练，不把CPU准备当真实GPU零更新预检。

## 主四格及运输贡献

主四格固定 Core、local_transport、Core+PMT+守卫、LT+PMT+守卫。Core/LT复用既有阶段checkpoint；无约束运输贡献对照复用science21660bf的Core+PMT及LT+PMT_global_regional，保留原单seed/order/训练预算和公开正负结果，不重训对照、不新增global_only或强度臂。八个复用轨迹、36个stage。未来若另获GPU授权只新增四守卫轨迹、18stage、1288更新、4真实零更新预检；正常4pre+4train+4eval=12次GPU驻留。复用对照必须关闭精确回归并核对完整来源、数据、参数、checkpoint/version/order/step；CPU未过不保证可复用。

原2epoch/batch64/workers4/AdamW.0003/WD.01/clip5/FD10/ridge.001/proximal.01/cosine eta1e-5/seed74002/split74002/taskorder/fitmeta/预训练和图像变换不变。ISIC每臂100/72/296=468更新，HK98/4/2/20/18/34=176；不降低Core、不val网格、不加seed/预算档位、不读旧训练图像。

## 生产插入点与失效

在run_module_additions.run添加有条件hook，关闭时原路径无新数学或RNG调用。epoch static_PC choose后守卫ON头；OFF为同状态bank的原common_shift/原local_memory/原native metric与competition头，非独立训练控制。仅保护已注册的完整fit行，不能用meta/子集替代。原canonical/captured复用，不新增图像提取。support frame/encoder version须匹配；每次optimizer.step立刻注销证书。旧守卫head可作为原proximal previous继续影响训练，但此时不再有完整支持保护承诺。中间minibatch的原native求解、loss、梯度全部保留，投影detach，不声称使用正确的持续约束envelope。

boundary维持meta加入原all-current统计，不作为独立校准/保护点；最后head重建之后、checkpoint之前，再对fit完整投影。旧矩PMT与regional独立运输保留、当前实际统计不运输、LT原区域prior不重复shift。

## 头表示与 checkpoint/eval

新guarded checkpoint保存权威float64 head、force=Atλ、reference、proposal、source/task/steps/seen/frame/version和当时全点KKT标量凭据。旧head.float槽只作格式兼容；guard-aware evaluator明确用double head，拒绝将无约束重建头冒充guarded头。重建H^-1(B+force)并核对stationarity/metadata及兼容float槽；不保存原fit特征/逐点dual供eval，不声称无支持可独立重验全部margin。

原对照公开指标及存储float头保持原样；same-state OFF与无约束proposal均在同一encoder特征下用double比较。新增PMT配对控制推理会同时记录其存储float头与已有double头预测差异；精度变化不得隐去或全部归因guard。guard vs 独立控制、guard vs 独立PMT轨迹、same-state OFF/proposal三个问题分开报告。全四条训练及所有stage seal/source/steps核对通过，才允许统一development val；eval入口先做barrier，再manifest/model访问。未来需要三次encoder推理/每stage，非零费用。

## CPU 预声明检查

八独立CPU子进程：①使用science21660bf原训练文件作参照，Core/LT关闭及PMT关闭精确回归、全部task真实run调用、四模拟零更新入口、全部guard checkpoint和合成完整eval、scope/budget/早val拒绝；②真实12层预训练ViT、两张合成图、CPU反传、0optimizer/参数buffer不变、双pool非零有限梯度；③–⑧1536维ISIC最大阶段epoch/boundary各Core/LT，以及HK最大旧bank boundary各Core/LT，完整fit约束、metadata/评分/失效和实际资源。所有high case为合成，不读医学特征；使用公开N和counts、固定.8signal/.6noise/seed241+K/已知affine(2.8,.65...)，不代表旧域拟合质量。

CPU小encoder替代和.synthetic step count仅检查调用链；真实ViT CPU backward不冒充最大旧bank/完整模型训练。高维矩阵及完整约束单独验收，不把两者混成同一次真实医学反传。solver完全沿用ce45194全dual固定阈值/maxiter2000/maxfun10000/maxcor10/ftol1e-20/gtol1e-12/maxls40、软840秒。子进程硬1800/整体14400秒只CPU资源门，不是新正式预算。每次驻留/user/system/peakRSS、import与pipeline包含关系、失败与未解全部保留；不放宽门或默默identity，不自动同轮重跑。

## 预算和证据

完整解析预算见BUDGET_SPEC。四未来轨迹54个snapshot守卫事件（36epoch、18boundary）；原canonical54次、153234图forward，student81436、FDteacher81436，总图像load曝光234670不增加。guard额外H求解/CPU-主机传输/私有证书字节/实际checkpoint bytes逐次记录，CPU嵌入GPU worker驻留，不重复加成本。配对控制推理费用单列。当前没有正式launch/deadline/GPU峰值，所有未知null，不拿旧CPU时间推造GPU预算符合。

全部CPU/工程正负、源码协议和费用必须GitHub显式代理remoteSHA/精确REPORT匿名HTTP200及NAS独立public_design/回执双端核实后称交付。保护仅当前fit/固定encoder快照，不能保证独立控制轨迹或val新类无损/旧旧风险/医学临床或统计显著效果。CPU通过不自动GPU-ready，更不证明方法有效。

## R2 工程修复谱系

首轮science0c707f9因目标Python静态with嵌套块上限，八子进程解析即退出；驻留2.190434727817774秒完整公开77bd9e29。未加载模型/图像、无数值检查或GPU。R2仅测试包装ExitStack及真实ViT层数断言，不改守卫数学、数据/预算/参数；新source新目录独立CPU检查，保留首次全部失败与费用。新增目标解释器compile gate，发生在公开源码冻结后、子进程派发前，不能用本机新Python编译替代。
