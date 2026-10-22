# 冻结方案：原生上层 token 重放，适应预训练表示而非追加小模块

## 假说和边界

有限风险实验已通过完整验收但主方法失败，且到达新类很弱；不能唯一归因某一超参。本轮检验：用稳定的下层空间 token 保留历史输入，同时开放原生上层表示学习，能否在公平固定表示重放控制之上改善初始/后来/尾类。它取消旧风险投影和教师目标，改变学习位置与历史信息格式，不放松旧方案门槛救分。

定点一手依据：[Latent Replay for Real-Time Continual Learning, arXiv1912.01100v2](https://arxiv.org/pdf/1912.01100)，已读第3/4节、5.1及相关讨论，非全部附录。中间激活记忆与可塑上层是成熟机制；本 ViT 适配不称忠实 MobileNet/AR1*/CWR* 复现、创新证明或 TMI 贡献。原文的批归一化老化与本 ViT LayerNorm 不同。

## 两臂强控制与公平口径

- FROZEN_UPPER：固定预训练最后两块与最终LayerNorm，只训练余弦head，普通balanced CE replay。
- PLASTIC_UPPER 主臂：同一预训练最后两块、LayerNorm与head，实际训练上层 attention/MLP，不额外加adapter/aux/teacher/constraint。

两臂相同总模型容量、token信息权限、记忆选择、当期数据、阶段、抽样和4500更新机会。可训练参数与反向算量不同是本次干预，不能声称等可训练容量/FLOPs。每臂上层总参数预计14,192,640→14,208,000；冻结臂只训练15,360→30,720头参数，主臂训练全部，实际资格复核。两臂都保存token，不让主臂多访问旧原图。不加随机初始化弱对照，不调优控制追分。

固定预训练ViT-B/16 patch embedding、位置嵌入和前十块，197×768 FP16输出token。主臂最后两块与最终norm FP32训练，部署CPU FP64。最终表示仍为norm后CLS与patch均值拼接1536维再L2归一，head行归一、scale25，无bias。全seen统一分类，无task-id路由/NCM/融合。下层和所有dropout/drop-path保持eval且p0，TF32关闭，显式attention、确定性算法、CUDA初始化前CUBLAS_WORKSPACE_CONFIG=:4096:8。

头仅由当前真实到达类的患者等权当前上层输出均值初始化，不提前读未来类。主臂上层跨阶段连续，两个臂每阶段重建AdamW：upper lr3e-5/head lr3e-4，wd1e-4，clip5，无调度/搜索。未根据val选择这些值。相同8seen类×8患者/图有放回=64行，一个确定性view，无输入dropout；microbatch8累计一次全逻辑batch梯度，基类2000、后来每阶段500，总4500/臂。与旧轮的双dropoutview/AdamW/SGD不是跨轮等协议因果比较。

## 数据权限和成本

相同20类10+2×5顺序、68058train（基类60129）、300val开发筛查；不test/Etest、不新下载、不未来图、不val训练/选择。每类≤32患者，每患者一个由metadata seed74002固定选择的token，不刷新。预计终点615患者，302592B/患者；约为旧1536维FP32 feature的49.25倍，是明确的新历史信息权限，不能混称原A/B方案。当前阶段所有token仅RAM，阶段末释放，不保存全train磁盘cache、不借其他实验feature或恢复静态IN/CLIP包。

预计每臂训练288000逻辑/上层forward行；主上层backward288000，冻结上层backward0；头前反向各288000。每臂初始化上层forward68058，val1350，最终训练probe12；额外资格/数学/完整网络接口另计。下层正式编码68058、val300一次，资格20图另计；资格额外重读2图执行完整encoder2+prefix2，上层接口2，属于重复计算，不能隐藏为仅20调用。FP16量化改变数值，独立验收以实际封存FP16 token为输入，不声称与无量化原图重放逐位一致。

## 首钟、硬资源和失败关闭

本独立实验新增实际存储<6,000,000,000B，5.4GB预停；首次远端实际准入至科学组/验收全部退出≤8h，所有准备、资格、故障和等待共享原钟，绝不重置。5s独立guard统计整个新根+中性入口max(st_size,st_blocks*512)，RSS32GiB、GPUreserved8GiB，要求host可用36GiB/GPU0free8GiB及实际NFS挂载/≥6GB余量与小写读probe。基类token约18.2GB RAM，不整表cat双份；保存stage只含有限memory/上层state与小probe，预计<3GB含临时副本。

截止前60s TERM，必要30s后KILL，完整组处理。科学不重试。任何资格/数值/预算失败，或完成后任一科学门失败，固定配方关闭；不扫层数/lr/步数/token精度/记忆/seed/门，不换主臂或事后refit救分。

## 强制bulk前真实GPU全流程

数学小型native-attention CE/有限差分独立检查；20张合法基类图正式reader→GPU前十层编码→正式GPU FP32/microbatch8/逻辑64，两臂各2次完整前反向/clip/AdamW。内部前8基类+后2基类模拟到达、记忆与head扩展；保存加载、真实两块FP64/独立NumPy显式attention/LayerNorm/exactGELU比较≤2e-6；验证冻结层不变、主上层实际改变；两张相同图完整encoder/拆分prefix接口检查。全部PASS才bulk，资格模型丢弃，未来图/val未读。CPU检查或GPU合成不代替。

## 封存、独立验收与固定九门

六stage×两臂共12模型全部val前seal，300val编码一次，各实际seen前缀，总2700模型query行（每臂1350）。分别保存FP64score，独立CPU NumPy从实际token和封存权重重算12score，绝对误差≤2e-6且预测完全同；独立memory患者复选、旧token逐项不变、患者交集0、前十层最终参数与原权重完全相同、冻结上层全stage与原参数同、主上层改变、stage/cost/seen顺序/真实probe重核。验收不新增编码或优化。最终资源/全组退出另行核实。

主绝对五门全部要求：final≥91/300、mean≥36.005291005291%、tail7≥25/105、initial≥44/150、later≥41/150。对唯一固定表示强控制四门全部要求：final+4、mean+1pp、tail4+2/60、later不降。保留旧单门阈值，因当前一控制共九门，不伪称十三门；完整独立验收也必须PASS。任何失败关闭。300val为重复开发筛查，不称临床test/SOTA；通过也不能证明唯一原因或新颖性。

完成后合并全部编码/初始化/训练/数学/资格/接口/评分/验收/失败成本，中文两臂比较，所有组退役，再有效7897代理push并确认远端SHA/匿名公开访问，无需人工审核。身份/影像/token/权重/逐图score/原始日志私有，不扩大清理。
