# R120 原生448共同SFT与RL增量

**冻结联合门：未通过。**

|方法|macro %|head %|tail %|tiny %|micro %|
|---|---:|---:|---:|---:|---:|
|SFT|47.0895|81.0961|9.9913|0.0000|80.3298|
|NATIVE_SFT|52.5639|85.4701|16.6662|0.0000|84.8057|
|NATIVE_TARS_FULL_IS|51.4341|78.3158|22.1086|0.0000|81.8610|
|NATIVE_GRPO_FULL_IS|49.7115|81.5340|14.9960|0.0000|81.6843|
|NATIVE_CE_FULL_IS|51.5390|80.1824|20.2916|0.0000|82.2144|

新共同SFT贡献、相对该起点的RL增量、相对原224SFT的端到端变化分别报告。新TARS表和查询由新TRAINfit产生，历史比较不是单组件因果消融。

完整TRAIN特征仅RAM，无新增特征磁盘副本；已有1556TRAIN/1698VAL特征只读复用，计算费用单独计。单devseed/静态HK23非临床、持续学习或最终创新证明。

全部原门：{"macro_over_nativeSFT_ge_1pp": false, "tail_over_nativeSFT_ge_2pp": true, "head_over_nativeSFT_ge_minus1pp": false, "macro_over_GRPO_positive": true, "tail_over_GRPO_positive": true, "macro_over_CE_positive": false, "tail_over_CE_positive": true, "end_to_end_macro_over_original_ge_1pp": true, "end_to_end_tail_over_original_ge_2pp": true, "end_to_end_head_over_original_ge_minus1pp": false}

## 效果分解与失败结论

原生448共同SFT相对原224SFT：macro+5.4744pp/head+4.3740pp/tail+6.6749pp/micro+4.4759pp。这是重建共同SFT及相关条件的整体贡献，不能记为RL。

TARS-RL相对新的共同SFT：macro−1.1298pp/head−7.1544pp/tail+5.4425pp/micro−2.9446pp。相对普通RL：macro+1.7226pp/tail+7.1126pp；相对CE：macro−.1049pp/tail+1.8170pp/head−1.8666pp。相对原224SFT端到端macro+4.3446pp/tail+12.1174pp，但head−2.7804pp，不能用较弱旧起点掩盖新起点上的损失。

十门通过6项，失败为：相对新SFT的macro增量、相对新SFT的head保护、相对CE的macro正增量、相对原224SFT的端到端head保护。不存在联合成功。该结果支持原生448监督适配本身有价值，但不支持当前后训练RL在此条件下改善联合目标；不泛化为所有RL或高分辨率无效。停止分辨率/SFT长度/lr/保护阈值救分，不扩静态变体矩阵。

## 类别信号、位移与计算成本

共同SFT854及三臂各256次更新均有实际非零参数位移，不是零步卡住。主RL的A类20/61降至2/61、UC2从47/66降至15/66；同时UC3从4/21升至15/21、impactedstool从5/33升至10/33。收益和损害一并报告。

五个tiny类VAL在所有四终点仍全0。主RL的逐类查询次数36/42/55/44/61，成功响应数0/0/1/1/15；前两类全错组36/42，第三/四类各仅一个混合组，最后类12个混合组。这说明更大输入及该SFT仍未充分解决稀缺奖励机会，成功响应也没有形成可靠VAL迁移；不能把重复响应当独立图像。

同查询CE不含相同KL且前向更少，普通RL和主RL则共享查询与KL系数。整套配方比较不能孤立证明RL目标的必要性。缓存采样与teacherforcing的BF16差异保留在native/曲线，未更新同优化路径logp差全为0，不能宣称采样逐bit一致。

正常退出0。计数：4次丢弃native、854SFT、768post尝试，46,415响应（34native+6,821TRAINfit+32,768RL+6,792VAL）；TRAIN编码5,269，VAL新编码0。旧编码历史成本不重复计。worker wall5086.83s/CPU5081.30s；含首次准备至controller退出5892.55s，未重置12h原钟。GPU峰值allocated10,588,611,584B/reserved12,698,255,360B，组RSS峰值9,911,726,080B。TRAIN特征7,152,336,896B仅RAM，不是新增磁盘。

92类行、1622曲线、原生初始化/新TRAIN表数学、共享查询、实际位移、全部封存早于VAL读取、逐例转移的安全聚合和所有十门已由CPU-only保存验证通过；不重新生成或调参。

## 后继

依已记录路线，下一步优先对冻结R117224完整IS做一次新seed74006配对机制复验，包含独立新共同SFT/mainRL/普通RL/CE，先冻结完整协议再实现/资源与原生准入/启动。不撤销旧头类失败、不挑seed；用途是检验正负模式可重复性，不是确认成功方法。新训练seed仍复用开发VAL，不是独立患者或完整持续学习证据。
