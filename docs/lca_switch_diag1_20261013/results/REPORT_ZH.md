# 最终 checkpoint 同状态模块开关诊断

OFF：保持 checkpoint 的 encoder、统计量、局部先验、竞争权重/RHS不变，仅令新增S为0后重解；ON：原保存float32解析头，double重建验收。差异统一为ON−OFF。OFF不是历史base/local_transport。
仅最终stage，因此不制造平均增量或遗忘指标。单seed反复development val；没有重训、调参、train/test/reserved读取。
本诊断不恢复去重前风险矩阵、不验证旧类漂移真值，也不能隔离训练过程中的全部历史作用。

|状态|模式|BA|Acc|tail|old|new|macroF1|
|---|---|---:|---:|---:|---:|---:|---:|
|ISIC_lca_supported|OFF|55.838864|55.254237|51.372004|50.777576|71.022727|56.612452|
|ISIC_lca_supported|ON|56.427833|55.932203|51.981760|51.184080|72.159091|57.134728|
|ISIC_lca_supported|ON−OFF pp|+0.588969|+0.677966|+0.609756|+0.406504|+1.136364|+0.522276|
|ISIC_local_transport_lca_supported|OFF|57.128846|56.610169|54.684463|52.535431|70.909091|58.029841|
|ISIC_local_transport_lca_supported|ON|57.128846|56.610169|54.684463|52.535431|70.909091|58.029841|
|ISIC_local_transport_lca_supported|ON−OFF pp|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|
|HK_lca_supported|OFF|62.456940|83.922261|47.166372|59.642054|92.013238|57.224157|
|HK_lca_supported|ON|62.456940|83.922261|47.166372|59.642054|92.013238|57.224157|
|HK_lca_supported|ON−OFF pp|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|
|HK_local_transport_lca_supported|OFF|62.470245|83.863369|47.166372|59.656626|92.013238|57.235882|
|HK_local_transport_lca_supported|ON|62.470245|83.863369|47.166372|59.656626|92.013238|57.235882|
|HK_local_transport_lca_supported|ON−OFF pp|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|

## 全部纠正与破坏

|状态|组|n|预测改变|纠正|破坏|错→另一错|净正确|
|---|---|---:|---:|---:|---:|---:|---:|
|ISIC_lca_supported|all|295|2|2|0|0|2|
|ISIC_lca_supported|old|211|1|1|0|0|1|
|ISIC_lca_supported|new|84|1|1|0|0|1|
|ISIC_local_transport_lca_supported|all|295|0|0|0|0|0|
|ISIC_local_transport_lca_supported|old|211|0|0|0|0|0|
|ISIC_local_transport_lca_supported|new|84|0|0|0|0|0|
|HK_lca_supported|all|1698|0|0|0|0|0|
|HK_lca_supported|old|1388|0|0|0|0|0|
|HK_lca_supported|new|310|0|0|0|0|0|
|HK_local_transport_lca_supported|all|1698|0|0|0|0|0|
|HK_local_transport_lca_supported|old|1388|0|0|0|0|0|
|HK_local_transport_lca_supported|new|310|0|0|0|0|0|

## 旧新类错误流向（ON−OFF为计数变化）

|状态|流向|OFF|ON|变化|
|---|---|---:|---:|---:|
|ISIC_lca_supported|old_correct|103|104|+1|
|ISIC_lca_supported|old_to_new|53|52|-1|
|ISIC_lca_supported|old_to_old_wrong|55|55|+0|
|ISIC_lca_supported|new_correct|60|61|+1|
|ISIC_lca_supported|new_to_old|15|14|-1|
|ISIC_lca_supported|new_to_new_wrong|9|9|+0|
|ISIC_local_transport_lca_supported|old_correct|107|107|+0|
|ISIC_local_transport_lca_supported|old_to_new|51|51|+0|
|ISIC_local_transport_lca_supported|old_to_old_wrong|53|53|+0|
|ISIC_local_transport_lca_supported|new_correct|60|60|+0|
|ISIC_local_transport_lca_supported|new_to_old|16|16|+0|
|ISIC_local_transport_lca_supported|new_to_new_wrong|8|8|+0|
|HK_lca_supported|old_correct|1140|1140|+0|
|HK_lca_supported|old_to_new|27|27|+0|
|HK_lca_supported|old_to_old_wrong|221|221|+0|
|HK_lca_supported|new_correct|285|285|+0|
|HK_lca_supported|new_to_old|25|25|+0|
|HK_lca_supported|new_to_new_wrong|0|0|+0|
|HK_local_transport_lca_supported|old_correct|1139|1139|+0|
|HK_local_transport_lca_supported|old_to_new|27|27|+0|
|HK_local_transport_lca_supported|old_to_old_wrong|222|222|+0|
|HK_local_transport_lca_supported|new_correct|285|285|+0|
|HK_local_transport_lca_supported|new_to_old|25|25|+0|
|HK_local_transport_lca_supported|new_to_new_wrong|0|0|+0|

## 实际作用

|状态|力/RHS|相对头变化|新类头变化范数|额外项新类直接力|分数RMS变化|相对分数RMS|
|---|---:|---:|---:|---:|---:|---:|
|ISIC_lca_supported|0.000282877522|0.00536289422|0.000170952418|0|0.00049933229|0.0021740912|
|ISIC_local_transport_lca_supported|0.000203560101|0.00323294607|0.000162582699|0|0.000312202713|0.00135868062|
|HK_lca_supported|3.11434286e-06|2.91733411e-05|9.96971732e-07|0|2.22505116e-06|1.43930279e-05|
|HK_local_transport_lca_supported|2.48451789e-06|2.54681574e-05|6.2985522e-07|0|1.66098538e-06|1.07054681e-05|

## 全部逐类正负

|状态|类|组|n|OFF recall|ON recall|差pp|纠正|破坏|改变|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
|ISIC_lca_supported|4|old|41|41.463415|43.902439|+2.439024|1|0|1|
|ISIC_lca_supported|7|old|24|45.833333|45.833333|+0.000000|0|0|0|
|ISIC_lca_supported|5|old|37|29.729730|29.729730|+0.000000|0|0|0|
|ISIC_lca_supported|1|old|37|51.351351|51.351351|+0.000000|0|0|0|
|ISIC_lca_supported|2|old|46|47.826087|47.826087|+0.000000|0|0|0|
|ISIC_lca_supported|6|old|26|88.461538|88.461538|+0.000000|0|0|0|
|ISIC_lca_supported|0|new|44|79.545455|81.818182|+2.272727|1|0|1|
|ISIC_lca_supported|3|new|40|62.500000|62.500000|+0.000000|0|0|0|
|ISIC_local_transport_lca_supported|4|old|41|43.902439|43.902439|+0.000000|0|0|0|
|ISIC_local_transport_lca_supported|7|old|24|45.833333|45.833333|+0.000000|0|0|0|
|ISIC_local_transport_lca_supported|5|old|37|40.540541|40.540541|+0.000000|0|0|0|
|ISIC_local_transport_lca_supported|1|old|37|48.648649|48.648649|+0.000000|0|0|0|
|ISIC_local_transport_lca_supported|2|old|46|47.826087|47.826087|+0.000000|0|0|0|
|ISIC_local_transport_lca_supported|6|old|26|88.461538|88.461538|+0.000000|0|0|0|
|ISIC_local_transport_lca_supported|0|new|44|81.818182|81.818182|+0.000000|0|0|0|
|ISIC_local_transport_lca_supported|3|new|40|60.000000|60.000000|+0.000000|0|0|0|
|HK_lca_supported|4|old|165|86.060606|86.060606|+0.000000|0|0|0|
|HK_lca_supported|5|old|1|0.000000|0.000000|+0.000000|0|0|0|
|HK_lca_supported|6|old|1|0.000000|0.000000|+0.000000|0|0|0|
|HK_lca_supported|22|old|21|71.428571|71.428571|+0.000000|0|0|0|
|HK_lca_supported|20|old|66|24.242424|24.242424|+0.000000|0|0|0|
|HK_lca_supported|1|old|113|94.690265|94.690265|+0.000000|0|0|0|
|HK_lca_supported|2|old|182|84.615385|84.615385|+0.000000|0|0|0|
|HK_lca_supported|15|old|123|99.186992|99.186992|+0.000000|0|0|0|
|HK_lca_supported|14|old|60|96.666667|96.666667|+0.000000|0|0|0|
|HK_lca_supported|12|old|41|80.487805|80.487805|+0.000000|0|0|0|
|HK_lca_supported|10|old|150|76.666667|76.666667|+0.000000|0|0|0|
|HK_lca_supported|11|old|61|27.868852|27.868852|+0.000000|0|0|0|
|HK_lca_supported|19|old|2|0.000000|0.000000|+0.000000|0|0|0|
|HK_lca_supported|16|old|8|37.500000|37.500000|+0.000000|0|0|0|
|HK_lca_supported|7|old|21|100.000000|100.000000|+0.000000|0|0|0|
|HK_lca_supported|0|old|7|71.428571|71.428571|+0.000000|0|0|0|
|HK_lca_supported|17|old|5|60.000000|60.000000|+0.000000|0|0|0|
|HK_lca_supported|9|old|164|98.780488|98.780488|+0.000000|0|0|0|
|HK_lca_supported|18|old|33|48.484848|48.484848|+0.000000|0|0|0|
|HK_lca_supported|13|old|160|94.375000|94.375000|+0.000000|0|0|0|
|HK_lca_supported|21|old|4|0.000000|0.000000|+0.000000|0|0|0|
|HK_lca_supported|8|new|153|98.039216|98.039216|+0.000000|0|0|0|
|HK_lca_supported|3|new|157|85.987261|85.987261|+0.000000|0|0|0|
|HK_local_transport_lca_supported|4|old|165|86.060606|86.060606|+0.000000|0|0|0|
|HK_local_transport_lca_supported|5|old|1|0.000000|0.000000|+0.000000|0|0|0|
|HK_local_transport_lca_supported|6|old|1|0.000000|0.000000|+0.000000|0|0|0|
|HK_local_transport_lca_supported|22|old|21|71.428571|71.428571|+0.000000|0|0|0|
|HK_local_transport_lca_supported|20|old|66|24.242424|24.242424|+0.000000|0|0|0|
|HK_local_transport_lca_supported|1|old|113|94.690265|94.690265|+0.000000|0|0|0|
|HK_local_transport_lca_supported|2|old|182|84.615385|84.615385|+0.000000|0|0|0|
|HK_local_transport_lca_supported|15|old|123|99.186992|99.186992|+0.000000|0|0|0|
|HK_local_transport_lca_supported|14|old|60|96.666667|96.666667|+0.000000|0|0|0|
|HK_local_transport_lca_supported|12|old|41|80.487805|80.487805|+0.000000|0|0|0|
|HK_local_transport_lca_supported|10|old|150|75.333333|75.333333|+0.000000|0|0|0|
|HK_local_transport_lca_supported|11|old|61|29.508197|29.508197|+0.000000|0|0|0|
|HK_local_transport_lca_supported|19|old|2|0.000000|0.000000|+0.000000|0|0|0|
|HK_local_transport_lca_supported|16|old|8|37.500000|37.500000|+0.000000|0|0|0|
|HK_local_transport_lca_supported|7|old|21|100.000000|100.000000|+0.000000|0|0|0|
|HK_local_transport_lca_supported|0|old|7|71.428571|71.428571|+0.000000|0|0|0|
|HK_local_transport_lca_supported|17|old|5|60.000000|60.000000|+0.000000|0|0|0|
|HK_local_transport_lca_supported|9|old|164|98.780488|98.780488|+0.000000|0|0|0|
|HK_local_transport_lca_supported|18|old|33|48.484848|48.484848|+0.000000|0|0|0|
|HK_local_transport_lca_supported|13|old|160|94.375000|94.375000|+0.000000|0|0|0|
|HK_local_transport_lca_supported|21|old|4|0.000000|0.000000|+0.000000|0|0|0|
|HK_local_transport_lca_supported|8|new|153|98.039216|98.039216|+0.000000|0|0|0|
|HK_local_transport_lca_supported|3|new|157|85.987261|85.987261|+0.000000|0|0|0|

## 费用与范围

本轮GPU进程驻留 220.528486670 秒；墙钟 110.49303603172302 秒。各次进程（包括失败）逐条计费，原训练和旧评价不重复收费。
历史闭合GPU 322803.622354565 秒；含本诊断累计 323024.150841236 秒。
CPU检查独立列于AUDIT；pipeline包含在驻留中，不重复相加。影像/模型/头/逐样本分数/特征/私有配置/raw日志不外传。
全部标量、目标函数变化、每类和old/new间隔变化见RESULTS.json。完成后不自动改模块或追加实验。

## 诊断结论（基于本轮冻结读数）

四个ON均逐类复现原公开最终结果，encoder所有state参数/buffer精确不变。新增项确实改变解析头和float32分数；全部ON目标函数低于OFF在ON目标下的值，solver和重建均通过。组合没有接线失效证据。
ISIC单独：295张中恰好2次预测改变，2纠正/0破坏/0错到另一错。类4旧类17→18/41，类0新类35→36/44；旧→新错误53→52，新→旧错误15→14。其余逐类不变。这里才有逐样本配对证据支持“只翻转两张”，此前净正确+2本身不足以证明。
ISIC组合、HK单独和HK组合：开关后逐样本预测完全相同；没有隐藏的正负翻转抵消。这一结论仅针对各自同状态ON/OFF，不声称与历史控制逐样本一致。
ISIC组合相对头变化0.323295%，分数相对RMS变化0.135868%；HK组合分别0.00254682%和0.00107055%。力/RHS也均非零；这些是实际机械作用，不能把非零当成有效贡献，也不能直接推断增大系数会改善。
组合的旧类错误：ISIC 104个中51个旧→新、53个旧→旧；HK 249个中27个旧→新、222个旧→旧。HK旧类内部混淆占89.16%，不支持把其全部问题归因于旧新偏移。旧类中心化M遗漏旧类公共模态，不等于完全不能间接改变旧新边界：ISIC单独的两次纠正均属于跨旧新组错误。
新类直接力为0，但解析竞争耦合使新类head仍发生非零变化。四状态均0新类正确→错误；然而真实类margin下降的新类数依次52/84、55/84、270/310、263/310，均值也均下降。变化很小，不能等同于新误分类、临床伤害或统计显著；同时不能以“显式力0/recall没降”宣称完全保护。
本诊断没有旧类漂移真值，也没有去重前U，故不能最终判定估计可靠性或把作用变弱归因于去重。没有更改模块、放宽强度、选checkpoint或追加实验。后续是否改模块由用户结合这些证据决定。

## 分类间隔正负（原始分数单位，非pp）

|状态|组|n|真实类margin均值变化|提高|不变|降低|旧新组margin均值变化|提高|不变|降低|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ISIC_lca_supported|all|295|+7.82125582e-05|163|0|132|+0.000135648099|176|0|119|
|ISIC_lca_supported|old|211|+0.000179827143|131|0|80|+0.000263126187|147|0|64|
|ISIC_lca_supported|new|84|-0.0001770336|32|0|52|-0.000184564718|29|0|55|
|ISIC_local_transport_lca_supported|all|295|+1.93924217e-05|148|0|147|+1.77635985e-05|149|0|146|
|ISIC_local_transport_lca_supported|old|211|+7.67361073e-05|119|0|92|+7.15587659e-05|118|0|93|
|ISIC_local_transport_lca_supported|new|84|-0.000124649455|29|0|55|-0.000117364739|31|0|53|
|HK_lca_supported|all|1698|-1.54751806e-06|534|28|1136|-1.58019443e-06|419|53|1226|
|HK_lca_supported|old|1388|-6.43734458e-07|494|28|866|-6.79682972e-07|382|53|953|
|HK_lca_supported|new|310|-5.59413625e-06|40|0|270|-5.61216185e-06|37|0|273|
|HK_local_transport_lca_supported|all|1698|-9.77702219e-07|599|38|1061|-1.02971148e-06|474|69|1155|
|HK_local_transport_lca_supported|old|1388|-3.82770894e-07|552|38|798|-4.41006736e-07|429|69|890|
|HK_local_transport_lca_supported|new|310|-3.64145925e-06|47|0|263|-3.66558952e-06|45|0|265|

真实类margin=正确类分数−其他类最大分数。旧新组margin=真值所属旧/新组的最大分数−另一组最大分数；它不代表组内分类正确。提高/降低为冻结float32读出之差的符号，不附加事后阈值。全部逐类间隔统计在MARGIN_CHANGES.csv，均保留正零负。

## 数值验收和逐进程费用

|状态|GPU|驻留秒|峰值torch分配bytes|重建相对误差|求解最大残差|
|---|---:|---:|---:|---:|---:|
|ISIC_lca_supported|0|34.045119799|1292172800|2.83340588e-16|7.11612924e-10|
|ISIC_local_transport_lca_supported|1|34.149616908|1292172800|1.63672617e-16|7.54154049e-10|
|HK_lca_supported|0|76.216014768|1292669440|1.46621796e-17|3.14667554e-10|
|HK_local_transport_lca_supported|1|76.117735196|1292014080|1.05783766e-17|3.19942799e-10|

CPU一次成功驻留 6.554404404 秒，其中pipeline 0.424991816 秒；客户端wall 6.841729375 秒另列，不相加。4个GPU子进程全部成功，没有本轮失败驻留。GPU驻留不是GPU利用时长，未提供计价费率，不换算货币费用。

公开边界：n<5的逐类连续margin统计保留在NAS私有汇总；公开JSON写null、CSV留空，以免单例均值/极值成为逐样本分数。所有类的分母、recall、纠正/破坏与margin正零负计数仍完整保留。此导出处理不改变推理、分类结果或费用。
