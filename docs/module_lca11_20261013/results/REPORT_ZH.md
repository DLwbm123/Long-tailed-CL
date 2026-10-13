# 新版LCA-inspired主四格：真实预检失败交付

本轮 INCOMPLETE，正式训练0更新、评价0阶段、完整真实预检0/4。两条ISIC预检在真实第一batch反传后、optimizer.step之前，因CUDA live state与CPU快照直接比较而失败。HK两条未派发，后续4条正式训练和18个stage评价均未运行。test/reserved封存，未开展development val分类评价。不能判断AIR差距缩小、新类保护或模块效用。

部署科学源：`97c0379685d8d98e90a73957302d49296d8b0755`。启动快照是历史RUNNING状态，本终止报告优先。已核对协调器及worker全部退出；不重启、不改失败部署源、不自动重跑。

## 原因、回执与修复

预检将参数保存为CPU快照，核对时遗漏 live tensor.detach().cpu()，torch.equal拒绝跨设备。此前CPU fixture将cuda映射为CPU identity，未覆盖设备边界。这是实现核对错误，不是训练发散或科学负结果。

旧入口还在完成模型state检查前写了generic PASS。两份文件缺少optimizer_updates/parameters_unchanged/lca_supported等终检字段，且worker/PROGRAM均INCOMPLETE；两份都按不完整回执保留，不计PASS。新collector要求终止COMPLETE与完整字段联合满足。

本地代码修复：比较前转CPU，全部检查后仅写一次PASS。CPU模拟回归检验成功路径、参数变化拒绝路径、设备归一化及不完整PASS拒绝。0GPU/0医学影像/0正式更新。修复未部署、未做真实GPU验证，不证明GPU-ready；按原协议等待新的明确授权后另冻结修复轮。

## 部分预检证据（不是PASS或效用）

|数据集/方法|结果|卡|batch|梯度norm|最大solve残差|GPU峰值byte|驻留秒|
|---|---|---:|---:|---:|---:|---:|---:|
|ISIC_lca_supported|INCOMPLETE|0|64|0.0166496988|7.9018090383e-10|12750242816|158.086669832|
|ISIC_local_transport_lca_supported|INCOMPLETE|1|64|0.0168296341|7.9084463260e-10|12947620864|158.253428904|
|HK/模块单独|未派发|—|—|—|—|—|—|
|HK/组合|未派发|—|—|—|—|—|—|

首task无旧类，新增风险先验预期为0；非零梯度和小残差只是部分反传/解析证据。未完成state未变化验证，不能称完成零更新准入。

## 主表：已公开控制复用与全部缺失

单位%；missing保留缺失，绝不填0。旧指标没有提供F1，留缺失。

|数据集|方法|finalBA|avgBA|tail|forget↓|old|new|finalAcc|avgAcc|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|ISIC|base|55.838864|59.108538|51.372004|13.485029|50.777576|71.022727|55.254237|59.104771|
|ISIC|local_transport|57.128846|60.404841|54.684463|12.818650|52.535431|70.909091|56.610169|60.346638|
|ISIC|lca_supported|missing|missing|missing|missing|missing|missing|missing|missing|
|ISIC|local_transport_lca_supported|missing|missing|missing|missing|missing|missing|missing|missing|
|ISIC|AIR|58.886629|61.682086|53.272738|10.264728|55.295809|69.659091|59.661017|62.398968|
|HK|base|62.456940|63.590989|47.166372|8.018505|59.642054|92.013238|83.922261|85.283917|
|HK|local_transport|62.470245|63.536719|47.166372|7.924568|59.656626|92.013238|83.863369|85.216954|
|HK|lca_supported|missing|missing|missing|missing|missing|missing|missing|missing|
|HK|local_transport_lca_supported|missing|missing|missing|missing|missing|missing|missing|missing|
|HK|AIR|63.942907|64.440881|47.400288|6.609374|61.085973|93.940719|85.983510|86.279185|

## 全部贡献与AIR差距

14组六指标增量全部missing：AB−base、AP−local_transport、AP−AB、AP−base、AB−AIR、AP−AIR、交互，每数据集7组。CONTRIBUTIONS.json显式null；未观察新正/零/负，不能以数学检查代替结果。

ISIC base既有AIR gap=3.047766pp；lca_supported新gap/缩小量未提供。

ISIC local_transport既有AIR gap=1.757783pp；local_transport_lca_supported新gap/缩小量未提供。

HK base既有AIR gap=1.485967pp；lca_supported新gap/缩小量未提供。

HK local_transport既有AIR gap=1.472662pp；local_transport_lca_supported新gap/缩小量未提供。


## ISIC已公开各stage和全部类（无新评价）

|方法|stage|n|Acc|BA|
|---|---:|---:|---:|---:|
|base|1|139|61.870504|60.852011|
|base|2|211|60.189573|60.634739|
|base|3|295|55.254237|55.838864|
|local_transport|1|139|61.870504|60.852011|
|local_transport|2|211|62.559242|63.233668|
|local_transport|3|295|56.610169|57.128846|
|AIR|1|139|64.028777|62.513047|
|AIR|2|211|63.507109|63.646581|
|AIR|3|295|59.661017|58.886629|

|类|train n|val n|base recall|local_transport recall|AIR recall|AB|AP|
|---|---:|---:|---:|---:|---:|---|---|
|4|458|41|41.463415|43.902439|48.780488|missing|missing|
|7|27|24|45.833333|45.833333|33.333333|missing|missing|
|5|256|37|29.729730|40.540541|54.054054|missing|missing|
|1|3263|37|51.351351|48.648649|62.162162|missing|missing|
|2|2835|46|47.826087|47.826087|56.521739|missing|missing|
|6|44|26|88.461538|88.461538|76.923077|missing|missing|
|0|10529|44|79.545455|81.818182|81.818182|missing|missing|
|3|1306|40|62.500000|60.000000|57.500000|missing|missing|

## HK已公开各stage和全部类（无新评价）

|方法|stage|n|Acc|BA|
|---|---:|---:|---:|---:|
|base|1|986|89.350913|64.946401|
|base|2|1015|85.615764|65.892203|
|base|3|1027|84.615385|64.413407|
|base|4|1224|83.905229|62.951878|
|base|5|1388|84.293948|60.885105|
|base|6|1698|83.922261|62.456940|
|local_transport|1|986|89.249493|64.818196|
|local_transport|2|1015|85.615764|65.892203|
|local_transport|3|1027|84.518014|64.267475|
|local_transport|4|1224|83.905229|62.996902|
|local_transport|5|1388|84.149856|60.775295|
|local_transport|6|1698|83.863369|62.470245|
|AIR|1|986|89.350913|65.257665|
|AIR|2|1015|85.911330|66.430064|
|AIR|3|1027|85.491723|65.071726|
|AIR|4|1224|85.130719|63.987918|
|AIR|5|1388|85.806916|61.955005|
|AIR|6|1698|85.983510|63.942907|

|类|train n|val n|base recall|local_transport recall|AIR recall|AB|AP|
|---|---:|---:|---:|---:|---:|---|---|
|4|626|165|86.060606|86.060606|86.060606|missing|missing|
|5|3|1|0.000000|0.000000|0.000000|missing|missing|
|6|6|1|0.000000|0.000000|0.000000|missing|missing|
|22|85|21|71.428571|71.428571|76.190476|missing|missing|
|20|288|66|24.242424|24.242424|37.878788|missing|missing|
|1|403|113|94.690265|94.690265|97.345133|missing|missing|
|2|736|182|84.615385|84.615385|90.659341|missing|missing|
|15|488|123|99.186992|99.186992|99.186992|missing|missing|
|14|252|60|96.666667|96.666667|98.333333|missing|missing|
|12|167|41|80.487805|80.487805|82.926829|missing|missing|
|10|595|150|76.666667|75.333333|77.333333|missing|missing|
|11|261|61|27.868852|29.508197|31.147541|missing|missing|
|19|6|2|0.000000|0.000000|0.000000|missing|missing|
|16|34|8|37.500000|37.500000|37.500000|missing|missing|
|7|83|21|100.000000|100.000000|100.000000|missing|missing|
|0|25|7|71.428571|71.428571|71.428571|missing|missing|
|17|23|5|60.000000|60.000000|60.000000|missing|missing|
|9|635|164|98.780488|98.780488|99.390244|missing|missing|
|18|127|33|48.484848|48.484848|42.424242|missing|missing|
|13|662|160|94.375000|94.375000|95.000000|missing|missing|
|21|18|4|0.000000|0.000000|0.000000|missing|missing|
|8|654|153|98.039216|98.039216|99.346405|missing|missing|
|3|644|157|85.987261|85.987261|88.535032|missing|missing|

## 费用与未完成范围

2次失败GPU0/1驻留分别158.086669832/158.253428904秒，总316.340098736秒（0.087872 GPU小时）。所有失败全计。旧闭合历史315757.8385525835秒，本轮加一次后316074.178651319秒；复用控制/AIR无新增费用，不相加历史cumulative。

全轮墙钟165.177006960秒，资源等待1.563338041秒已在墙钟内；两卡驻留和wall不相加。生产CPU全部3次驻留70.033424476秒（含导入失败、嵌套pipeline）；报告fixture pipeline0.006219084秒独立，未测process residence。此前设计CPU21.392109550秒历史不重计。新增修复mock检查此次记录驻留0.038970459秒，另一次未计时开发检查不虚造数值。

缺失：HK2预检、4正式轨迹1288目标更新（实际0）、18checkpoint/18stage评价、全部31类的新recall/新类保护及14组贡献。未重训旧控制、未读取旧checkpoint补评；missing不补0。当前无法判定新版模块能否缩小AIR差距。单seed/order、重复development val、极少分母；不作test/临床/等资源结论。
