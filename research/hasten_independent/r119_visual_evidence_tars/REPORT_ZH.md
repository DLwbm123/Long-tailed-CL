# R119 固定448视觉证据与RL增量

**冻结联合门：未通过。**

|方法|macro %|head %|tail %|tiny %|micro %|
|---|---:|---:|---:|---:|---:|
|SFT|47.0895|81.0961|9.9913|0.0000|80.3298|
|VISUAL_SFT|43.1622|77.4869|5.7170|0.0000|74.3816|
|VISUAL_TARS_FULL_IS|47.2370|76.6180|15.1851|0.0000|79.5642|
|VISUAL_GRPO_FULL_IS|45.1017|80.0472|6.9792|0.0000|79.5053|
|VISUAL_CE_FULL_IS|48.8520|76.0717|19.1578|5.0000|78.6808|

分辨率本身的读出变化、相同448下RL增量及相对原224起点的端到端变化分别报告。无新增SFT，CE计算更少且无KL。

单开发seed/静态HK23/重复开发VAL，非独立患者、完整持续学习或成熟方法创新证明。未按VAL扫分辨率或选checkpoint。

全部原门：{"macro_over_visualSFT_ge_1pp": true, "tail_over_visualSFT_ge_2pp": true, "head_over_visualSFT_ge_minus1pp": true, "macro_over_GRPO_positive": true, "tail_over_GRPO_positive": true, "macro_over_CE_positive": false, "tail_over_CE_positive": false, "end_to_end_macro_over_original_ge_1pp": false, "end_to_end_tail_over_original_ge_2pp": true, "end_to_end_head_over_original_ge_minus1pp": false}


## 分辨率本身、RL恢复与端到端效果

原224训练SFT改用448输入，macro43.1622/head77.4869/tail5.7170%，相对原224读出分别−3.9273/−3.6092/−4.2743pp。直接换输入已造成起点退化，不能把后训练相对此弱起点的大幅恢复全记为端到端提升。

主RL macro47.2370/head76.6180/tail15.1851%，相对同448起点macro+4.0749/tail+9.4681/head−.8690pp，局部三门通过；相对同分辨率普通GRPO的macro/tail高2.1354/8.2059pp。但相对CE，macro **−1.6150pp**、tail **−3.9727pp**；CE macro48.8520/tail19.1578/head76.0717%。原十门仅六项通过：RL相对CE的macro/tail、端到端macro>=原224+1pp、端到端head保护四门失败。

相对原224SFT，主RL macro仅+.1476pp、tail+5.1938pp、head−4.4782pp。相对重用R117端点，macro−2.7792/head−1.5521/tail−4.1178pp；不能宣称本轮提升或“更多像素解决了瓶颈”。CE在本轮macro/tail更优，但计算更少且无KL，正则与计算差异仍披露。

## tiny和视觉输入假设的边界

主RL五个tiny正确响应分别为0/424、0/320、1/464、0/368、7/496；对应混合组0、0、1、0、7。普通RL为0、0、2、0、5个正确响应。主RL所有tiny VAL仍0；CE tiny5%只来自UC2-3的1/4，其他tiny仍0。增加像素并未在这套固定起点/预算下产生可靠tiny信号或泛化。训练与验证分母小，不能据此否定所有更大分辨率训练。

本轮只检验了**224上训练的参数直接切到448并后训练**，没有训练原生448共同SFT。因此结论限定于该迁移条件，不能把输入适配问题、高分辨率本身、冻结视觉塔或RL技术类别混成一个因果判断。不会扫描新的分辨率、学习率、support长度或保护阈值。

## 完成验证和成本

全部三臂256/256真实非零参数更新；768条曲线、92类计数、十门、compact映射、类组计数、native原生/缓存parity及全部seal早于VALcache均通过saved-only检查，无新增模型调用/TEST。原生448末token logit parity0不等于缓存生成与teacher forcing在BF16下逐步概率相同，后者差异仍完整记录，未更新优化路径概率比保持1。

正常退出0，无守卫触发、无重试。39594生成响应=34native+32768RL+6792VAL；3丢弃native尝试、768科学更新、0SFT。TRAIN编码1560=1556新cache+2nativeencode+2原生parity重复视觉；VAL编码1698。新增BF16特征逻辑3412066304B，两文件均<4GiB，未下载本机或公开。worker墙钟3754.56秒/CPU3708.69秒；含准备的原钟4086.70秒。GPU峰值分配10,584,901,120B/保留12,320,768,000B；进程组RSS峰值6,121,578,496B。guard根峰值3,435,761,664B，加旧根2,399,010,816B约5.835GB，另留本机余量，严格低于累计10GB。保护全部历史/模型/失败证据，不清理重置成本。

后继若检验原生448适配，必须另立协议，先从无任务适配器的generic起点按固定SFT协议建立共同448起点，所有RL与CE共享并单独读出SFT贡献；不将重建SFT称为RL增量。此前infer-only结果不能作为扫描分辨率或挑SFT配置的许可，亦不能绕过强监督对照。当前仍单开发seed/静态HK23/反复开发VAL，非临床、完整CL或成熟创新结论。
