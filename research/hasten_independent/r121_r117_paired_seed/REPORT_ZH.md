# R121 单新seed配对机制复验

**冻结联合门：未通过。**

|方法|macro %|head %|tail %|tiny %|micro %|
|---|---:|---:|---:|---:|---:|
|SFT|47.0895|81.0961|9.9913|0.0000|80.3298|
|SEED_SFT|47.8401|82.2367|10.3167|0.0000|80.0353|
|SEED_TARS_FULL_IS|47.2223|75.4230|16.4579|0.0000|78.6808|
|SEED_GRPO_FULL_IS|44.8745|78.3246|8.3835|0.0000|78.6219|
|SEED_CE_FULL_IS|47.8112|76.1737|16.8703|5.0000|77.6207|

新seed74006共同SFT和主RL/普通RL/CE配对；七项诊断联合门与六项方向复现分开报告。不是重跑旧九门，也不撤销R117原头类失败。

原224特征TRAIN6821/VAL1698只读复用，无新增副本。恰一新seed、不选seed，仍复用开发VAL；非独立患者、完整持续学习或成功方法确认。CE缺同KL/计算的限制保持。

本次七诊断门：{"macro_over_seedSFT_ge_1pp": false, "tail_over_seedSFT_ge_2pp": true, "head_over_seedSFT_ge_minus1pp": false, "macro_over_GRPO_positive": true, "tail_over_GRPO_positive": true, "macro_over_CE_positive": false, "tail_over_CE_positive": false}
六项方向复现：{"SFT_macro_positive": false, "SFT_tail_positive": true, "GRPO_macro_positive": true, "GRPO_tail_positive": true, "CE_macro_positive": false, "CE_tail_positive": false}

## 配对复验结论：完整正信号未复现

七项诊断门通过3/7；六项方向读出也只有3/6为正。主方法相对自己共同SFT macro−.6179pp/head−6.8137pp/tail+6.1412pp；相对普通RL macro+2.3477pp/tail+8.0744pp；相对CE macro−.5889pp/tail−.4124pp/head−.7507pp。因此保留“优于同查询普通RL”的方向，但“优于共同SFT和CE”的完整模式没有重复。不能只挑原seed74005或改门宣称稳定RL增量。

|seed|main−SFT macro/head/tail pp|main−GRPO macro/tail pp|main−CE macro/tail pp|
|---|---|---|---|
|74005 R117|+2.9268/−2.9260/+9.3117|+3.8536/+10.4530|+3.2094/+2.5532|
|74006 R121|−.6179/−6.8137/+6.1412|+2.3477/+8.0744|−.5889/−.4124|

两条独立训练随机流仍复用开发VAL，不能据两次训练宣称统计显著性、新患者泛化或完整持续学习；不采用两seed均值遮盖符号翻转。R121没有重跑PG_ONLY/naturalTARS，不能说重跑原R117九门；旧头类失败保持。新SFT、新TRAIN表、tape、响应种子都是冻结流程的新随机实例，差异不是只由一个初始化因素造成。

## 所有类别的收益和损害

统一逐类贡献表覆盖两seed下对SFT/GRPO/CE的全部23类，没有只选收益类。新seed主RL的A类31/61降至0/61、UC2从42/66降至26/66，tail中的duodenalbulb从13/21降至5/21；BD从16/41升至35/41、UC3从2/21升至10/21、impactedstool从1/33升至8/33。A+BD合并净正确−12，不能仅报告BD救回；这与旧seed的同对净+11不同。

所有RL终点tiny仍0。CE的tiny5%只来自UC2–3的1/4，另外还有非tiny尾类一个1/8，均不代表稳定稀缺类泛化。主RLtiny261次查询中251个全错组、10个混合组、11个成功响应/2088响应；普通RL为256个全错组、5个混合组、5个成功响应。增加可用响应在此没有形成VALtiny增量。此为重复、演化策略的响应数，不是独立图像或端点TRAIN准确率。旧R117未保存逐类全错/全对/混合，不能补造跨seed组成对照。

## 执行、失败保留与预算

正常退出0，854SFT及三臂各256更新全部实际非零；优化路径未更新logp差0，不是零步卡住。92类行/1622曲线/新seed初始化重放/无旧任务SFT/新TRAIN表数学和共享tape/七门与六方向/封存先于VAL/聚合转移均保存校验通过。

成功run：4丢弃native+854SFT+768post尝试，46,415响应=34native+6821TRAINfit+32768RL+6792VAL，12TRAIN编码/0新VAL编码。另保留失败native1的2TRAIN编码/0响应/0优化器，共14TRAIN编码；ALL_ATTEMPT_COSTS不删失败。worker wall5234.40s/CPU5233.15s；含准备/失败/重试至退出5588.73s，原八小时时钟未重置。GPU峰值allocated8,947,758,592B/reserved9,464,446,976B，组RSS峰值4,436,299,776B。原生缓存batch2差.0625、匹配历史batch8差0、native/manuallogit差0均保留，阈值没有放宽。

本次只读复用既有224特征，新增磁盘主要是小型adapter/报告，不复制2.2GB缓存；全研究旧根/新根/本机余量累计仍低于10,000,000,000B。工程准入与科学联合目标严格区分。

## 路线决定

遵守预声明：不追加seed挑结果，不扫原分辨率/SFT长度/lr/support/保护阈值，不将原seed正结果删除或泛化为所有RL无效。当前静态配方及一次配对复验收口，未启动新的GPU训练。下一步为基于原始TARS机制与完整正负证据的主线机制审查；只有能写出新的可否证问题、实际对照及资源/数据访问准入时才另冻协议执行，不以持续授权为由制造重复静态跑。具体审查入口见ROUTE_DECISION_ZH.md。持续研究授权没有撤销或暂停。
