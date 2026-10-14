# 动态锚定：主四格、静态贡献对照及AIR差距

主方法固定local_transport。控制指标及AIR复用；动态/静态每臂2epoch、原fit/meta与参数不变。静态与动态使用同强度定义，绝对lambda取各自起点G0；训练轨迹分化后不保证数值lambda相等。

|数据|方法|finalBA|avgBA|tail|forget↓|old|new|Acc|avgAcc|macroF1|AIR−方法 BA|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ISIC|base|55.838864|59.108538|51.372004|13.485029|50.777576|71.022727|55.254237|59.104771|未提供|+3.047766|
|ISIC|local_transport|57.128846|60.404841|54.684463|12.818650|52.535431|70.909091|56.610169|60.346638|未提供|+1.757783|
|ISIC|dynamic_anchor|54.858131|59.013108|48.867060|14.792673|49.469932|71.022727|54.237288|58.923766|55.568194|+4.028499|
|ISIC|local_transport_dynamic_anchor|55.233837|59.623022|51.437924|14.894877|50.008753|70.909091|54.576271|59.510694|56.170900|+3.652792|
|ISIC|static_anchor|55.480060|59.220418|49.542736|14.342222|49.920382|72.159091|54.915254|59.149755|56.177064|+3.406570|
|ISIC|local_transport_static_anchor|55.851215|59.978965|52.047680|14.938824|50.415257|72.159091|55.254237|59.894661|56.713187|+3.035414|
|ISIC|AIR|58.886629|61.682086|53.272738|10.264728|55.295809|69.659091|59.661017|62.398968|60.088227|+0.000000|
|HK|base|62.456940|63.590989|47.166372|8.018505|59.642054|92.013238|83.922261|85.283917|未提供|+1.485967|
|HK|local_transport|62.470245|63.536719|47.166372|7.924568|59.656626|92.013238|83.863369|85.216954|未提供|+1.472662|
|HK|dynamic_anchor|62.483290|63.628097|47.166372|8.061795|59.670914|92.013238|83.981154|85.324252|57.251014|+1.459617|
|HK|local_transport_dynamic_anchor|62.496595|63.575390|47.166372|7.967858|59.685486|92.013238|83.922261|85.259901|57.262739|+1.446312|
|HK|static_anchor|62.483290|63.628097|47.166372|8.061795|59.670914|92.013238|83.981154|85.324252|57.251014|+1.459617|
|HK|local_transport_static_anchor|62.496595|63.575390|47.166372|7.967858|59.685486|92.013238|83.922261|85.259901|57.262739|+1.446312|
|HK|AIR|63.942907|64.440881|47.400288|6.609374|61.085973|93.940719|85.983510|86.279185|59.231669|+0.000000|

## 全部六指标差异（pp；forget负值有利）

|数据|比较|BA|avgBA|tail|forget|old|new|
|---|---|---:|---:|---:|---:|---:|---:|
|ISIC|dynamic_anchor−base|-0.980733|-0.095429|-2.504944|+1.307644|-1.307644|+0.000000|
|ISIC|local_transport_dynamic_anchor−local_transport|-1.895009|-0.781820|-3.246539|+2.076228|-2.526678|+0.000000|
|ISIC|static_anchor−base|-0.358804|+0.111880|-1.829268|+0.857193|-0.857193|+1.136364|
|ISIC|local_transport_static_anchor−local_transport|-1.277631|-0.425877|-2.636783|+2.120174|-2.120174|+1.250000|
|ISIC|dynamic_anchor−static_anchor|-0.621929|-0.207310|-0.675676|+0.450450|-0.450450|-1.136364|
|ISIC|local_transport_dynamic_anchor−local_transport_static_anchor|-0.617378|-0.355943|-0.609756|-0.043946|-0.406504|-1.250000|
|ISIC|local_transport_dynamic_anchor−dynamic_anchor|+0.375707|+0.609913|+2.570864|+0.102205|+0.538821|-0.113636|
|ISIC|local_transport_static_anchor−static_anchor|+0.371156|+0.758547|+2.504944|+0.596602|+0.494875|+0.000000|
|ISIC|local_transport_dynamic_anchor−base|-0.605026|+0.514484|+0.065920|+1.409849|-0.768823|-0.113636|
|ISIC|dynamic_anchor−AIR|-4.028499|-2.668977|-4.405678|+4.527944|-5.825877|+1.363636|
|ISIC|local_transport_dynamic_anchor−AIR|-3.652792|-2.059064|-1.834814|+4.630149|-5.287056|+1.250000|
|ISIC|static_anchor−AIR|-3.406570|-2.461668|-3.730002|+4.077494|-5.375427|+2.500000|
|ISIC|local_transport_static_anchor−AIR|-3.035414|-1.703121|-1.225058|+4.674096|-4.880552|+2.500000|
|ISIC|dynamic_interaction|-0.914276|-0.686390|-0.741595|+0.768584|-1.219035|+0.000000|
|ISIC|static_interaction|-0.918827|-0.537757|-0.807515|+1.262981|-1.262981|+0.113636|
|HK|dynamic_anchor−base|+0.026350|+0.037108|+0.000000|+0.043290|+0.028860|+0.000000|
|HK|local_transport_dynamic_anchor−local_transport|+0.026350|+0.038671|+0.000000|+0.043290|+0.028860|+0.000000|
|HK|static_anchor−base|+0.026350|+0.037108|+0.000000|+0.043290|+0.028860|+0.000000|
|HK|local_transport_static_anchor−local_transport|+0.026350|+0.038671|+0.000000|+0.043290|+0.028860|+0.000000|
|HK|dynamic_anchor−static_anchor|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|
|HK|local_transport_dynamic_anchor−local_transport_static_anchor|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|
|HK|local_transport_dynamic_anchor−dynamic_anchor|+0.013305|-0.052706|+0.000000|-0.093937|+0.014572|+0.000000|
|HK|local_transport_static_anchor−static_anchor|+0.013305|-0.052706|+0.000000|-0.093937|+0.014572|+0.000000|
|HK|local_transport_dynamic_anchor−base|+0.039655|-0.015599|+0.000000|-0.050647|+0.043432|+0.000000|
|HK|dynamic_anchor−AIR|-1.459617|-0.812785|-0.233916|+1.452421|-1.415058|-1.927480|
|HK|local_transport_dynamic_anchor−AIR|-1.446312|-0.865491|-0.233916|+1.358484|-1.400486|-1.927480|
|HK|static_anchor−AIR|-1.459617|-0.812785|-0.233916|+1.452421|-1.415058|-1.927480|
|HK|local_transport_static_anchor−AIR|-1.446312|-0.865491|-0.233916|+1.358484|-1.400486|-1.927480|
|HK|dynamic_interaction|-0.000000|+0.001564|+0.000000|+0.000000|-0.000000|+0.000000|
|HK|static_interaction|-0.000000|+0.001564|+0.000000|+0.000000|-0.000000|+0.000000|

## ISIC最终每类及新类全部救回/破坏

|方法|类|角色|n|救回|破坏|净|control间隔|ON间隔|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|dynamic_anchor|4|old|41|0|3|-3|-0.031888|-0.038472|
|dynamic_anchor|7|old|24|0|0|+0|-0.020734|-0.022561|
|dynamic_anchor|5|old|37|0|1|-1|-0.055622|-0.063906|
|dynamic_anchor|1|old|37|0|0|+0|0.061863|0.058656|
|dynamic_anchor|2|old|46|1|0|+1|-0.014515|-0.013567|
|dynamic_anchor|6|old|26|0|0|+0|0.343985|0.344662|
|dynamic_anchor|0|new|44|0|0|+0|0.194911|0.192890|
|dynamic_anchor|3|new|40|0|0|+0|0.067322|0.077735|
|local_transport_dynamic_anchor|4|old|41|0|2|-2|-0.031125|-0.039041|
|local_transport_dynamic_anchor|7|old|24|0|0|+0|-0.016334|-0.018143|
|local_transport_dynamic_anchor|5|old|37|0|3|-3|-0.046794|-0.056535|
|local_transport_dynamic_anchor|1|old|37|0|0|+0|0.065907|0.063229|
|local_transport_dynamic_anchor|2|old|46|0|1|-1|-0.015146|-0.016661|
|local_transport_dynamic_anchor|6|old|26|0|0|+0|0.344558|0.345673|
|local_transport_dynamic_anchor|0|new|44|0|0|+0|0.193741|0.193201|
|local_transport_dynamic_anchor|3|new|40|1|1|+0|0.061358|0.070670|
|static_anchor|4|old|41|0|3|-3|-0.031888|-0.035735|
|static_anchor|7|old|24|0|0|+0|-0.020734|-0.022068|
|static_anchor|5|old|37|0|0|+0|-0.055622|-0.060695|
|static_anchor|1|old|37|0|0|+0|0.061863|0.060289|
|static_anchor|2|old|46|1|0|+1|-0.014515|-0.013170|
|static_anchor|6|old|26|0|0|+0|0.343985|0.345840|
|static_anchor|0|new|44|1|0|+1|0.194911|0.192191|
|static_anchor|3|new|40|0|0|+0|0.067322|0.075161|
|local_transport_static_anchor|4|old|41|0|1|-1|-0.031125|-0.038102|
|local_transport_static_anchor|7|old|24|0|0|+0|-0.016334|-0.018663|
|local_transport_static_anchor|5|old|37|0|3|-3|-0.046794|-0.058257|
|local_transport_static_anchor|1|old|37|0|0|+0|0.065907|0.062266|
|local_transport_static_anchor|2|old|46|0|1|-1|-0.015146|-0.016665|
|local_transport_static_anchor|6|old|26|0|0|+0|0.344558|0.342741|
|local_transport_static_anchor|0|new|44|0|0|+0|0.193741|0.191151|
|local_transport_static_anchor|3|new|40|2|1|+1|0.061358|0.071494|

## HK最终每类及新类全部救回/破坏

|方法|类|角色|n|救回|破坏|净|control间隔|ON间隔|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|dynamic_anchor|4|old|165|1|0|+1|0.257193|0.257307|
|dynamic_anchor|5|old|1|0|0|+0|-0.125314|-0.125607|
|dynamic_anchor|6|old|1|0|0|+0|-0.296228|-0.296621|
|dynamic_anchor|22|old|21|0|0|+0|0.119644|0.119490|
|dynamic_anchor|20|old|66|0|0|+0|-0.075457|-0.075385|
|dynamic_anchor|1|old|113|0|0|+0|0.396895|0.397157|
|dynamic_anchor|2|old|182|0|0|+0|0.366494|0.366357|
|dynamic_anchor|15|old|123|0|0|+0|0.601285|0.601141|
|dynamic_anchor|14|old|60|0|0|+0|0.293796|0.294134|
|dynamic_anchor|12|old|41|0|0|+0|0.182051|0.182449|
|dynamic_anchor|10|old|150|0|0|+0|0.135220|0.135274|
|dynamic_anchor|11|old|61|0|0|+0|-0.071590|-0.071785|
|dynamic_anchor|19|old|2|0|0|+0|-0.207984|-0.207707|
|dynamic_anchor|16|old|8|0|0|+0|-0.045717|-0.045546|
|dynamic_anchor|7|old|21|0|0|+0|0.525443|0.525524|
|dynamic_anchor|0|old|7|0|0|+0|0.049416|0.049401|
|dynamic_anchor|17|old|5|0|0|+0|0.049747|0.049365|
|dynamic_anchor|9|old|164|0|0|+0|0.497060|0.497001|
|dynamic_anchor|18|old|33|0|0|+0|-0.023011|-0.022830|
|dynamic_anchor|13|old|160|0|0|+0|0.378279|0.378262|
|dynamic_anchor|21|old|4|0|0|+0|-0.120116|-0.120148|
|dynamic_anchor|8|new|153|0|0|+0|0.378249|0.378279|
|dynamic_anchor|3|new|157|0|0|+0|0.205537|0.205396|
|local_transport_dynamic_anchor|4|old|165|1|0|+1|0.258670|0.258753|
|local_transport_dynamic_anchor|5|old|1|0|0|+0|-0.125585|-0.125789|
|local_transport_dynamic_anchor|6|old|1|0|0|+0|-0.289932|-0.289997|
|local_transport_dynamic_anchor|22|old|21|0|0|+0|0.117496|0.117354|
|local_transport_dynamic_anchor|20|old|66|0|0|+0|-0.076853|-0.076741|
|local_transport_dynamic_anchor|1|old|113|0|0|+0|0.398534|0.399118|
|local_transport_dynamic_anchor|2|old|182|0|0|+0|0.368454|0.368242|
|local_transport_dynamic_anchor|15|old|123|0|0|+0|0.602632|0.602985|
|local_transport_dynamic_anchor|14|old|60|0|0|+0|0.296245|0.296043|
|local_transport_dynamic_anchor|12|old|41|0|0|+0|0.181410|0.181203|
|local_transport_dynamic_anchor|10|old|150|0|0|+0|0.134928|0.134852|
|local_transport_dynamic_anchor|11|old|61|0|0|+0|-0.070342|-0.070294|
|local_transport_dynamic_anchor|19|old|2|0|0|+0|-0.211752|-0.211579|
|local_transport_dynamic_anchor|16|old|8|0|0|+0|-0.048733|-0.048757|
|local_transport_dynamic_anchor|7|old|21|0|0|+0|0.530117|0.530018|
|local_transport_dynamic_anchor|0|old|7|0|0|+0|0.045532|0.045480|
|local_transport_dynamic_anchor|17|old|5|0|0|+0|0.050460|0.049963|
|local_transport_dynamic_anchor|9|old|164|0|0|+0|0.497911|0.497993|
|local_transport_dynamic_anchor|18|old|33|0|0|+0|-0.022363|-0.022168|
|local_transport_dynamic_anchor|13|old|160|0|0|+0|0.381440|0.381356|
|local_transport_dynamic_anchor|21|old|4|0|0|+0|-0.120963|-0.121136|
|local_transport_dynamic_anchor|8|new|153|0|0|+0|0.379384|0.379349|
|local_transport_dynamic_anchor|3|new|157|0|0|+0|0.205708|0.205490|
|static_anchor|4|old|165|1|0|+1|0.257193|0.257088|
|static_anchor|5|old|1|0|0|+0|-0.125314|-0.125622|
|static_anchor|6|old|1|0|0|+0|-0.296228|-0.296535|
|static_anchor|22|old|21|0|0|+0|0.119644|0.119430|
|static_anchor|20|old|66|0|0|+0|-0.075457|-0.075339|
|static_anchor|1|old|113|0|0|+0|0.396895|0.397097|
|static_anchor|2|old|182|0|0|+0|0.366494|0.366309|
|static_anchor|15|old|123|0|0|+0|0.601285|0.600886|
|static_anchor|14|old|60|0|0|+0|0.293796|0.294109|
|static_anchor|12|old|41|0|0|+0|0.182051|0.182510|
|static_anchor|10|old|150|0|0|+0|0.135220|0.135103|
|static_anchor|11|old|61|0|0|+0|-0.071590|-0.071712|
|static_anchor|19|old|2|0|0|+0|-0.207984|-0.207716|
|static_anchor|16|old|8|0|0|+0|-0.045717|-0.045546|
|static_anchor|7|old|21|0|0|+0|0.525443|0.525530|
|static_anchor|0|old|7|0|0|+0|0.049416|0.049376|
|static_anchor|17|old|5|0|0|+0|0.049747|0.049385|
|static_anchor|9|old|164|0|0|+0|0.497060|0.497033|
|static_anchor|18|old|33|0|0|+0|-0.023011|-0.022827|
|static_anchor|13|old|160|0|0|+0|0.378279|0.378282|
|static_anchor|21|old|4|0|0|+0|-0.120116|-0.120163|
|static_anchor|8|new|153|0|0|+0|0.378249|0.378260|
|static_anchor|3|new|157|0|0|+0|0.205537|0.205357|
|local_transport_static_anchor|4|old|165|1|0|+1|0.258670|0.258530|
|local_transport_static_anchor|5|old|1|0|0|+0|-0.125585|-0.125729|
|local_transport_static_anchor|6|old|1|0|0|+0|-0.289932|-0.290015|
|local_transport_static_anchor|22|old|21|0|0|+0|0.117496|0.117375|
|local_transport_static_anchor|20|old|66|0|0|+0|-0.076853|-0.076748|
|local_transport_static_anchor|1|old|113|0|0|+0|0.398534|0.398850|
|local_transport_static_anchor|2|old|182|0|0|+0|0.368454|0.368279|
|local_transport_static_anchor|15|old|123|0|0|+0|0.602632|0.602939|
|local_transport_static_anchor|14|old|60|0|0|+0|0.296245|0.296061|
|local_transport_static_anchor|12|old|41|0|0|+0|0.181410|0.181210|
|local_transport_static_anchor|10|old|150|0|0|+0|0.134928|0.134891|
|local_transport_static_anchor|11|old|61|0|0|+0|-0.070342|-0.070318|
|local_transport_static_anchor|19|old|2|0|0|+0|-0.211752|-0.211548|
|local_transport_static_anchor|16|old|8|0|0|+0|-0.048733|-0.048731|
|local_transport_static_anchor|7|old|21|0|0|+0|0.530117|0.529982|
|local_transport_static_anchor|0|old|7|0|0|+0|0.045532|0.045443|
|local_transport_static_anchor|17|old|5|0|0|+0|0.050460|0.050056|
|local_transport_static_anchor|9|old|164|0|0|+0|0.497911|0.497974|
|local_transport_static_anchor|18|old|33|0|0|+0|-0.022363|-0.022174|
|local_transport_static_anchor|13|old|160|0|0|+0|0.381440|0.381385|
|local_transport_static_anchor|21|old|4|0|0|+0|-0.120963|-0.121135|
|local_transport_static_anchor|8|new|153|0|0|+0|0.379384|0.379381|
|local_transport_static_anchor|3|new|157|0|0|+0|0.205708|0.205479|

## 费用和限制

本轮GPU进程驻留25440.383818秒；24条成功operation逐次保留，旧控制训练不收费，配对控制新推理包含在eval费用。CPU及失败见AUDIT。
全stage逐类n/recall在STAGE_CLASS_RECALLS.csv，全stage救回/破坏与间隔在PAIRED_CORRECTIONS.csv，全部混淆矩阵与频数组在RESULTS。
旧类方向仍为统计运输估计；支持/年龄不是校准概率。锚定可能与FD/proximal重复或伤害新类，不把激活当收益，不以平均保护门称逐类无损。
单seed/order反复development val，非患者独立/临床/test证据；AIR解析批数、训练曝光和资源不同，不作等预算SOTA结论。所有正零负保留，不追加seed、强度、预算臂或失败重跑。

## 完整贡献结论

8个各自控制开发门均未全部通过。ISIC动态相对静态额外下降；HK动态和静态所有阶段分类recall相同，只有旧类4最终救回1例，不支持额外动态贡献。

详见[完整机制分析与下一步设计边界](ANALYSIS_ZH.md)及[全部阶段、31类及固定分母附表](ALL_STAGE_CLASS_ZH.md)。首次本地报告导入缺torch发生在远程收集前，已修复报告层，部署科学源未变，0新增GPU成本；工程记录保留于AUDIT。
