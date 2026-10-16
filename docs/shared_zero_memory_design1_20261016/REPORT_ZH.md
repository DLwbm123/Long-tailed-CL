# 共同零点记忆：完整合成分类序列

SYNTHETIC_SEQUENCE_CHECKS_COMPLETE
预声明合成分类信号：False。不是医学/ViT/Core复现，也不改变旧完整查询族失败。

|规模|阶段|方法|Acc|BA|旧BA|新BA|
|---|---:|---|---:|---:|---:|---:|
|small|1|oracle_wide|0.9852941176470589|0.9852941176470589|None|0.9852941176470589|
|small|1|oracle_local|0.9852941176470589|0.9852941176470589|None|0.9852941176470589|
|small|1|moment_local|0.9852941176470589|0.9852941176470589|None|0.9852941176470589|
|small|1|quadratic_local|0.9852941176470589|0.9852941176470589|None|0.9852941176470589|
|small|2|oracle_wide|0.9706498951781971|0.9707303211139529|0.9705882352941176|0.9710144927536232|
|small|2|oracle_local|0.9622641509433962|0.9694337880079568|0.9613970588235294|0.9855072463768116|
|small|2|moment_local|0.9685534591194969|0.9671248934356352|0.9724264705882353|0.9565217391304348|
|small|2|quadratic_local|0.9622641509433962|0.9694337880079568|0.9613970588235294|0.9855072463768116|
|small|3|oracle_wide|0.966796875|0.9610579557910122|0.9671248934356352|0.9428571428571428|
|small|3|oracle_local|0.953125|0.9564090853732796|0.9609263995453253|0.9428571428571428|
|small|3|moment_local|0.958984375|0.9529426988186579|0.9658283603296391|0.9142857142857143|
|small|3|quadratic_local|0.953125|0.9564090853732796|0.9609263995453253|0.9428571428571428|
|ISIC_scale|1|oracle_wide|1.0|1.0|None|1.0|
|ISIC_scale|1|oracle_local|1.0|1.0|None|1.0|
|ISIC_scale|1|moment_local|1.0|1.0|None|1.0|
|ISIC_scale|1|quadratic_local|1.0|1.0|None|1.0|
|ISIC_scale|2|oracle_wide|1.0|1.0|1.0|1.0|
|ISIC_scale|2|oracle_local|1.0|1.0|1.0|1.0|
|ISIC_scale|2|moment_local|1.0|1.0|1.0|1.0|
|ISIC_scale|2|quadratic_local|1.0|1.0|1.0|1.0|
|ISIC_scale|3|oracle_wide|1.0|1.0|1.0|1.0|
|ISIC_scale|3|oracle_local|1.0|1.0|1.0|1.0|
|ISIC_scale|3|moment_local|1.0|1.0|1.0|1.0|
|ISIC_scale|3|quadratic_local|1.0|1.0|1.0|1.0|
|HK_scale|1|oracle_wide|1.0|1.0|None|1.0|
|HK_scale|1|oracle_local|1.0|1.0|None|1.0|
|HK_scale|1|moment_local|1.0|1.0|None|1.0|
|HK_scale|1|quadratic_local|1.0|1.0|None|1.0|
|HK_scale|2|oracle_wide|1.0|1.0|1.0|1.0|
|HK_scale|2|oracle_local|1.0|1.0|1.0|1.0|
|HK_scale|2|moment_local|1.0|1.0|1.0|1.0|
|HK_scale|2|quadratic_local|1.0|1.0|1.0|1.0|
|HK_scale|3|oracle_wide|1.0|1.0|1.0|1.0|
|HK_scale|3|oracle_local|1.0|1.0|1.0|1.0|
|HK_scale|3|moment_local|1.0|1.0|1.0|1.0|
|HK_scale|3|quadratic_local|1.0|1.0|1.0|1.0|
|HK_scale|4|oracle_wide|1.0|1.0|1.0|1.0|
|HK_scale|4|oracle_local|1.0|1.0|1.0|1.0|
|HK_scale|4|moment_local|1.0|1.0|1.0|1.0|
|HK_scale|4|quadratic_local|1.0|1.0|1.0|1.0|
|HK_scale|5|oracle_wide|1.0|1.0|1.0|1.0|
|HK_scale|5|oracle_local|1.0|1.0|1.0|1.0|
|HK_scale|5|moment_local|1.0|1.0|1.0|1.0|
|HK_scale|5|quadratic_local|1.0|1.0|1.0|1.0|
|HK_scale|6|oracle_wide|1.0|1.0|1.0|1.0|
|HK_scale|6|oracle_local|1.0|1.0|1.0|1.0|
|HK_scale|6|moment_local|1.0|1.0|1.0|1.0|
|HK_scale|6|quadratic_local|1.0|1.0|1.0|1.0|

|规模|阶段|candidate−moment BA|candidate−local oracle BA|local−wide oracle BA|全指标无损|存在严格收益|
|---|---:|---:|---:|---:|---|---|
|small|1|0.0|0.0|0.0|True|False|
|small|2|0.0023088945723215293|0.0|-0.0012965331059960894|False|True|
|small|3|0.0034663865546217254|0.0|-0.004648870417732542|False|True|
|ISIC_scale|1|0.0|0.0|0.0|True|False|
|ISIC_scale|2|0.0|0.0|0.0|True|False|
|ISIC_scale|3|0.0|0.0|0.0|True|False|
|HK_scale|1|0.0|0.0|0.0|True|False|
|HK_scale|2|0.0|0.0|0.0|True|False|
|HK_scale|3|0.0|0.0|0.0|True|False|
|HK_scale|4|0.0|0.0|0.0|True|False|
|HK_scale|5|0.0|0.0|0.0|True|False|
|HK_scale|6|0.0|0.0|0.0|True|False|

CPU驻留319.54205643059686秒/user391.68629100000004/system190.917999/peak1836.4453125MiB。父wall319.6059019220993另列，内部拟合/头训练/评价/operation含驻留不相加。
前17轮4920.581218053587秒，累计5240.123274484184。失败[]，missing不补0。

|case|退出码|驻留秒|user|system|RSS MiB|timeout|
|---|---:|---:|---:|---:|---:|---|
|shared_zero_arithmetic|0|6.520243343897164|5.260286|1.12267|406.71484375|False|
|small|0|8.528026841115206|7.676443|1.222691|435.5703125|False|
|ISIC_scale|0|121.19684419268742|149.717856|76.667192|1032.53125|False|
|HK_scale|0|183.29694205289707|229.031706|111.905446|1836.4453125|False|

四臂同线性架构/48步/固定合成特征；候选和moment训练仅当前支持+旧memory，完整旧支持仅用于oracle对照及训练封存后的诊断。独立合成评价seed+10000固定，不访问医学val/test。
每stage不能被末stage覆盖；shared domain可行不等分类有效。无encoder更新或正式GPU预算证据。

详细全阶段旧新贡献、逐类正负、界宽与构造费用见 [ANALYSIS_ZH.md](ANALYSIS_ZH.md)。
