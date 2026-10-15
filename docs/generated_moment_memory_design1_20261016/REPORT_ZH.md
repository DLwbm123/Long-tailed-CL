# 生成分布矩拟合 CPU 结果

NO_COMPLETE_FRESH_BATCH_SIGNAL
冻结门：{'arithmetic': True, 'small': True, 'ISIC_scale': False, 'HK_scale': True}

|规模|方法|批次|平均误差|最大误差|
|---|---|---:|---:|---:|
|small|random|None|0.46949275434481286|4.070225454885384|
|small|moment|None|0.1194397546281767|0.9211246768856798|
|small|normalized_moment|None|0.11763816603764503|0.8606048552431345|
|small|posthoc_gaussian|0|0.07363098610946255|0.7559726988831166|
|small|generated_moment|0|0.07319190314926255|0.7536553925502316|
|small|posthoc_gaussian|1|0.08331841816700214|0.7593491523100655|
|small|generated_moment|1|0.09381691499386442|0.7927498922753673|
|small|posthoc_gaussian|2|0.06551714786057598|0.47179842940476746|
|small|generated_moment|2|0.08368790449377035|0.6833153527847231|
|small|posthoc_gaussian|3|0.09116700437632143|0.8211607396582377|
|small|generated_moment|3|0.09194671600063602|0.801307816237923|
|small|posthoc_gaussian|4|0.0751669366021236|0.44045530523334087|
|small|generated_moment|4|0.07983150553798342|0.4364070273050906|
|small|posthoc_gaussian|5|0.06969015617320551|0.48193347519196506|
|small|generated_moment|5|0.07869502568747622|0.6692841425237193|
|small|posthoc_gaussian|6|0.07124561156619878|0.5068421465801678|
|small|generated_moment|6|0.08731302548257562|0.6006251363696113|
|small|posthoc_gaussian|7|0.07532616316561486|0.5708071641282362|
|small|generated_moment|7|0.08299626838182193|0.546534085168727|
|ISIC_scale|random|None|0.7426432062066558|9.30599857946361|
|ISIC_scale|moment|None|0.38132055384804664|6.305717785750893|
|ISIC_scale|normalized_moment|None|0.3800882723377137|6.2905064793167025|
|ISIC_scale|posthoc_gaussian|0|0.16327410426072453|5.970537104379332|
|ISIC_scale|generated_moment|0|0.1634385682016222|5.968313577405038|
|ISIC_scale|posthoc_gaussian|1|0.1581396278130798|5.99037262222895|
|ISIC_scale|generated_moment|1|0.15804731357513416|5.988352214809842|
|ISIC_scale|posthoc_gaussian|2|0.1690114198272145|5.958603579846834|
|ISIC_scale|generated_moment|2|0.16844043124909414|5.956506568753609|
|ISIC_scale|posthoc_gaussian|3|0.155487507636205|5.968397831998237|
|ISIC_scale|generated_moment|3|0.15543336888633344|5.9659344200056985|
|ISIC_scale|posthoc_gaussian|4|0.15703728109462148|5.989044484935349|
|ISIC_scale|generated_moment|4|0.15632892196846968|5.9870291979467805|
|ISIC_scale|posthoc_gaussian|5|0.15077870352224482|5.9743579488437835|
|ISIC_scale|generated_moment|5|0.15029932687673353|5.972232282088585|
|ISIC_scale|posthoc_gaussian|6|0.16854693244544794|5.948562378751205|
|ISIC_scale|generated_moment|6|0.17017313881551313|5.946190338609899|
|ISIC_scale|posthoc_gaussian|7|0.15464692655216358|5.925230835801097|
|ISIC_scale|generated_moment|7|0.15579992328322367|5.922967301280121|
|HK_scale|random|None|0.5901057534828332|9.224681345566598|
|HK_scale|moment|None|0.2901069827701376|4.212113611536965|
|HK_scale|normalized_moment|None|0.2893387740087247|4.195976891300087|
|HK_scale|posthoc_gaussian|0|0.12346903946948008|2.779458526540156|
|HK_scale|generated_moment|0|0.12295693935674039|2.764650871092913|
|HK_scale|posthoc_gaussian|1|0.11782080769162541|2.755247083819951|
|HK_scale|generated_moment|1|0.11760841029870905|2.741951708578083|
|HK_scale|posthoc_gaussian|2|0.12498191998275884|3.0693736024474836|
|HK_scale|generated_moment|2|0.12472788600249958|3.0528153698301255|
|HK_scale|posthoc_gaussian|3|0.11979257125065962|2.8339959083520476|
|HK_scale|generated_moment|3|0.11823880423036279|2.826862048914125|
|HK_scale|posthoc_gaussian|4|0.12915723199255252|2.9374448307120877|
|HK_scale|generated_moment|4|0.12814281143559084|2.921235723176251|
|HK_scale|posthoc_gaussian|5|0.11985841791593167|2.475709752915206|
|HK_scale|generated_moment|5|0.11979090397089091|2.4628744544078427|
|HK_scale|posthoc_gaussian|6|0.1279843373353057|2.702050036484812|
|HK_scale|generated_moment|6|0.1281913806844896|2.686552321266067|
|HK_scale|posthoc_gaussian|7|0.11344990035926965|2.7416990137529726|
|HK_scale|generated_moment|7|0.11443651762806527|2.734143308567958|

CPU独立驻留518.763515999075秒，user433.66657000000004，system480.78160199999996，peak RSS3290.6796875MiB；父wall518.7988575110212另列。内部构造/eval/operation包含于驻留，不相加。

先前十轮3323.4341566190124秒，累计3842.1976726180874秒。失败[]；缺失null不补0。

|case|退出码|驻留秒|user|system|RSS MiB|timeout|
|---|---:|---:|---:|---:|---:|---|
|generated_moment_arithmetic|0|8.272671629674733|5.303287|1.1263969999999999|408.80859375|False|
|small|0|7.7685393169522285|7.927352|0.8202079999999999|435.20703125|False|
|ISIC_scale|0|159.9921401613392|136.874131|150.261166|1676.13671875|False|
|HK_scale|0|342.73016489110887|283.5618|328.573831|3290.6796875|False|

0GPU/医学images/features/checkpoints/val/test/正式更新。每次只使用16点，8批仅审计，不平均替代主门。
新查询先固定再独立采样；不能承诺观察当次样本后再自适应选择的所有查询。生成分布估计无偏不等真实旧分布无偏。
归一化可改变均值，原二阶残差不保证各向同性或PSD。拟合解析生成矩仍不保证目标优化收敛或高阶判别查询正确；两个拟合臂各48步，case总96步全计。尚无完整序列/分类训练/生产预算证据。

# 生成矩拟合结果：仍无完整信号

主候选small8/8、ISIC0/8、HK8/8，整体门false；posthoc同Gaussian解码对照也是8/8、0/8、8/8。4CPU正常，0工程失败/超时。解析矩cubature误差<=1.11e-16，联合中心/权重FD1.37e-11，query FD3.01e-10通过，仅证明实现的数学性质。

ISIC主候选8批max5.922967–5.988352，虽全部较posthoc小约.002，但门限5.675146远未满足；mean5批好3批坏。HKmax8批小幅好，mean6批好2批坏。smallmean仅1批好7批坏，max4好4坏。所有96贡献及57比较/665类误差保留，不能只报HK最大误差改善。

必须用相同生成矩目标比较：small posthoc .002859001318→candidate .001280428902改善；ISIC .005283290477→.005288944133略坏；HK .008077880799→.008088961390略坏。各臂初末训练目标明显下降，不代表候选优于同目标控制，也没有全局/局部最优证书。不能从此说容量不足或生成矩目标已解决问题。高维有效节点最小约15.947/15.713，非单点权重塌缩。mean/Q/trace和全部class见ANALYTIC_GEOMETRY。

8批平均仅诊断：ISIC mean .146225/max5.963441，与posthoc .146506/5.965638近似；HK .114462/2.767579，mean略差posthoc .114424而max略好2.781206；small .039286/.300255好于posthoc .042982/.383602。不得用平均改判逐批主门。共同stress不同，不跨轮max直接对比。

停止继续微调归一化、采样或目标系数。下一个机制检验拟重新分配相同存储：原16中心向量+16权重，共16d+16数；用1均值向量+15主协方差方向+15特征值+1剩余方差，同为16d+16。剩余方向以等方方差保留trace，生成16新点再归一化，主方向由全部current synthetic支持一次分解，不按query/val选。保留原moment48和只均值+等方协方差对照，检验方向结构，非扩容量或调强度。原始Gaussian矩可核对，单位归一化依然改变矩、非高斯高阶结构仍可能丢失。此为新假说，不保证通过；低样本统计可能识别原特征，合成通过不自动授权医学部署存储。

资源：4独立CPU驻留518.763515999075秒，父wall518.7988575110212分列；user433.666570/system480.781602，peak3290.6796875MiB。两臂各48更新，配对case96，全部实测计费，内部构造/评价/operation不相加。11轮43CPU累计3842.1976726180874秒；GPU新增0、历史400036.74851310626不变，client/collector未测null。

未访问医学images/features/checkpoint/val/test，没有完整序列、分类训练和生产预算证据；该实现主门未过，不进入医学GPU。
