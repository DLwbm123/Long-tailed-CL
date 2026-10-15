# Snapshot guard 正式轮：失败终态与完整成本

本轮 INCOMPLETE / STOPPED_DEPENDENCIES。两条 ISIC 在 task3 epoch2 的 epoch_init 守卫未取得证书；各完成320/468更新。HK未派发，全四条训练屏障未通过，因此没有任何development val评价，test/reserved持续封存。没有重试、回退identity、放宽门或修改健康源。

|轨迹|状态|实测更新/计划|已存stage|失败GPU驻留秒|
|---|---|---:|---:|---:|
|ISIC_guard|INCOMPLETE|320/468|2|3122.357290|
|ISIC_LT_guard|INCOMPLETE|320/468|2|3156.261070|
|HK_guard|NOT_DISPATCHED_DEPENDENCY_STOP|null/176|0|null|
|HK_LT_guard|NOT_DISPATCHED_DEPENDENCY_STOP|null/176|0|null|

HK的null表示未生成STATUS/成本记录；协调器确认未派发，不假造测量。ISIC两stage checkpoint仅保留在NAS，未作部分评价。

## 14行主表（百分比；历史复用与本轮缺失分开）

|数据|方法|证据|BA|avgBA|tail|forget↓|old|new|Acc|avgAcc|F1|AIR差距|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ISIC | Core | 历史 | 55.838864 | 59.108538 | 51.372004 | 13.485029 | 50.777576 | 71.022727 | 55.254237 | 59.104771 | null | 3.047766 |
| ISIC | LT | 历史 | 57.128846 | 60.404841 | 54.684463 | 12.818650 | 52.535431 | 70.909091 | 56.610169 | 60.346638 | null | 1.757783 |
| ISIC | PMT | 历史 | 57.213909 | 59.852617 | 60.801937 | 7.198407 | 56.701879 | 58.750000 | 56.271186 | 59.759710 | 57.929007 | 1.672720 |
| ISIC | LT+PMT | 历史 | 57.897386 | 60.646373 | 62.087369 | 7.707116 | 57.196515 | 60.000000 | 56.949153 | 60.459633 | 58.514798 | 0.989243 |
| ISIC | PMT+guard | 未评价 | null | null | null | null | null | null | null | null | null | null |
| ISIC | LT+PMT+guard | 未评价 | null | null | null | null | null | null | null | null | null | null |
| ISIC | AIR | 历史 | 58.886629 | 61.682086 | 53.272738 | 10.264728 | 55.295809 | 69.659091 | 59.661017 | 62.398968 | 60.088227 | 0.000000 |
| HK | Core | 历史 | 62.456940 | 63.590989 | 47.166372 | 8.018505 | 59.642054 | 92.013238 | 83.922261 | 85.283917 | null | 1.485967 |
| HK | LT | 历史 | 62.470245 | 63.536719 | 47.166372 | 7.924568 | 59.656626 | 92.013238 | 83.863369 | 85.216954 | null | 1.472662 |
| HK | PMT | 历史 | 62.356909 | 63.605015 | 47.122510 | 8.038453 | 59.593951 | 91.367970 | 83.804476 | 85.307177 | 56.895103 | 1.585998 |
| HK | LT+PMT | 历史 | 62.525168 | 63.525978 | 47.171783 | 7.514359 | 59.778235 | 91.367970 | 83.922261 | 85.270423 | 57.252447 | 1.417739 |
| HK | PMT+guard | 未评价 | null | null | null | null | null | null | null | null | null | null |
| HK | LT+PMT+guard | 未评价 | null | null | null | null | null | null | null | null | null | null |
| HK | AIR | 历史 | 63.942907 | 64.440881 | 47.400288 | 6.609374 | 61.085973 | 93.940719 | 85.983510 | 86.279185 | 59.231669 | 0.000000 |

全部六指标贡献见CONTRIBUTION_DELTAS.csv：本轮对Core/LT、无约束PMT、组合与单独及交互均为null。历史PMT的ISIC旧类收益伴随新类损害、HK收益弱仍成立，不能用本轮求解证书代替医学效果。AIR存在数据/预算/预训练资源差异；这些都是反复使用的development val，非独立患者确认、临床或显著性证明。

## 真实训练中已取得的守卫证书

|轨迹|task/epoch/事件|约束|正dual|保留目标收益%|solver秒|最小slack|
|---|---|---:|---:|---:|---:|---:|
| ISIC_guard | 1/1/epoch_init | 9531 | 0 | null | 0.209760 | -8.677184052e-10 |
| ISIC_guard | 1/2/epoch_init | 9531 | 0 | null | 0.257418 | -8.656969422e-10 |
| ISIC_guard | 1/2/boundary_final | 9531 | 0 | null | 0.185473 | -8.552949306e-10 |
| ISIC_guard | 2/1/epoch_init | 11460 | 0 | null | 0.240677 | -3.816298196e-10 |
| ISIC_guard | 2/2/epoch_init | 11460 | 926 | 25.891093 | 269.727929 | -5.15399939e-09 |
| ISIC_guard | 2/2/boundary_final | 11460 | 955 | 28.393313 | 200.529419 | -2.580152917e-09 |
| ISIC_guard | 3/1/epoch_init | 66038 | 0 | null | 0.381343 | -7.82336705e-10 |
| ISIC_LT_guard | 1/1/epoch_init | 9531 | 0 | null | 0.174283 | -8.637427546e-10 |
| ISIC_LT_guard | 1/2/epoch_init | 9531 | 0 | null | 0.177354 | -8.61328907e-10 |
| ISIC_LT_guard | 1/2/boundary_final | 9531 | 0 | null | 0.161588 | -8.637187446e-10 |
| ISIC_LT_guard | 2/1/epoch_init | 11460 | 0 | null | 0.203289 | -3.828359067e-10 |
| ISIC_LT_guard | 2/2/epoch_init | 11460 | 841 | 26.866996 | 161.251276 | -6.279428917e-09 |
| ISIC_LT_guard | 2/2/boundary_final | 11460 | 951 | 28.597043 | 205.439418 | -2.946285751e-09 |
| ISIC_LT_guard | 3/1/epoch_init | 66038 | 0 | null | 0.410386 | -7.990708517e-10 |

14个成功事件按(job,task,epoch,event,steps)去除累计快照重复，完整标量/KKT/trace均保留。4个task2活跃约束事件保留25.89%–28.60%的同状态二次目标收益，不是分类收益。10个NATIVE_FEASIBLE事件正dual为0，收益接近浮点舍入且retained=null；即使identity布尔为false也不认定为有效纠错。失败task3 epoch2为9434 fit、66038条约束，失败audit未保存，不补数值或identity判断。

## 失败原因与诊断缺口

两项均为ValueError: Full-fit guard unresolved; no identity fallback。源在判定非CERTIFIED后先抛错，尚未写入events，因此未保存该次termination、KKT、迭代/trace和solver时间。只能确认未获得冻结数值证书，无法区分840秒软限、优化器停止或残差未过门；不能断言超时、OOM、约束不可行或只能identity。24小时总deadline未触发。具体匿名原因见FAILURES_ANON.json，私有日志/traceback不发布。

## 费用与缺失

两次失败GPU驻留全部计入：6278.618359938264秒（1.744060656 GPU小时）。父wall 3163.497510194778秒另列；CPU调度检查9.429333493113518秒另列。成功及失败CPU投影、等待、传输已包含GPU驻留，不相加。旧闭合393758.130153167993秒，本轮后400036.748513106257秒；四条旧预检不重复收费。

实际训练峰值显存、失败solver耗时/传输量、最终完整预算计数未落盘，均为null；不借用预检峰值、不把规划曝光当实测。成功事件资源只是局部小计，BUDGET_AUDIT完整保留包含关系、缺失及checkpoint字节。费用为资源时间，无单价不造金额。

每stage/31类的本轮n、recall、救回/破坏、错误换错误、间隔、旧新混淆以及same-state OFF/proposal与独立控制比较均未评价，见FINAL_CLASS_COMPARISON、STAGE_CLASS_COVERAGE和MISSING_EVALUATION。已公开历史逐类结果仍在../paired_transport1_20261015/results/；未读取旧NAS模型重评。

仅固定encoder/frame的epoch初始化与boundary最终头保护current-fit；一次optimizer更新后旧证书失效，不能承诺val新类无损。原生loss/proximal/参数不变，guard不新增提图；checkpoint无完整支持不能独立全点重验。科学源f4b272de092a2db492126654a8bcceef071fe8e1保持冻结。当前完整交付失败结果，停止本轮，不自动新增科学轮。
