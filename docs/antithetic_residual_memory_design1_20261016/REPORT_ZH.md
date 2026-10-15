# 相反噪声配对 CPU 结果

NO_COMPLETE_FRESH_BATCH_SIGNAL
冻结门：{'arithmetic': True, 'small': True, 'ISIC_scale': False, 'HK_scale': True}

|规模|方法|批次|平均误差|最大误差|
|---|---|---:|---:|---:|
|small|random|None|0.4502553584520718|4.199816732111838|
|small|moment|None|0.11237643819645948|1.004328256082256|
|small|normalized_moment|None|0.11067845833569516|0.9332474030705082|
|small|independent_residual|0|0.13347347368650445|0.924381928211726|
|small|antithetic_residual|0|0.08104265940199011|0.5741210821656342|
|small|independent_residual|1|0.13034761892114288|0.7813505365777447|
|small|antithetic_residual|1|0.06928216601101773|0.35573990230685504|
|small|independent_residual|2|0.12266695923080506|0.6769704148517461|
|small|antithetic_residual|2|0.07065471565196323|0.38356087165094443|
|small|independent_residual|3|0.13908752797671745|0.7251618419857335|
|small|antithetic_residual|3|0.06770609252753647|0.35849662287376605|
|small|independent_residual|4|0.13615348983487766|0.7478504436129456|
|small|antithetic_residual|4|0.07562239599316799|0.5888157866688997|
|small|independent_residual|5|0.12664600199303028|0.8746820315445873|
|small|antithetic_residual|5|0.07584001213511683|0.5006927333227909|
|small|independent_residual|6|0.14939991418887957|0.9952375183695743|
|small|antithetic_residual|6|0.08186176372921274|0.44650548802775303|
|small|independent_residual|7|0.13362424876628615|0.5293629912974271|
|small|antithetic_residual|7|0.06499426769780511|0.5565861457091487|
|ISIC_scale|random|None|0.7982730418326701|9.387177700234119|
|ISIC_scale|moment|None|0.3891898718974787|6.149714664596658|
|ISIC_scale|normalized_moment|None|0.3875725563870572|6.135455239268062|
|ISIC_scale|independent_residual|0|0.13949295820628896|5.865181449412852|
|ISIC_scale|antithetic_residual|0|0.14181015817209147|5.531118590234952|
|ISIC_scale|independent_residual|1|0.1458686495707421|5.189823099042481|
|ISIC_scale|antithetic_residual|1|0.13882054080425135|5.530756240892428|
|ISIC_scale|independent_residual|2|0.2071941500596283|5.937230980421933|
|ISIC_scale|antithetic_residual|2|0.1461628170284471|5.5482533889683925|
|ISIC_scale|independent_residual|3|0.17033422964920175|5.911769660211659|
|ISIC_scale|antithetic_residual|3|0.1412247358766332|5.560951995871038|
|ISIC_scale|independent_residual|4|0.16778179275707303|5.364966480701645|
|ISIC_scale|antithetic_residual|4|0.142671158406835|5.566251302351098|
|ISIC_scale|independent_residual|5|0.16064291251954252|5.92249882061768|
|ISIC_scale|antithetic_residual|5|0.13775886292034767|5.527962469137352|
|ISIC_scale|independent_residual|6|0.17167989492607608|5.486270167546019|
|ISIC_scale|antithetic_residual|6|0.15984851809790437|5.584598035235775|
|ISIC_scale|independent_residual|7|0.17557341338690294|5.539374237920082|
|ISIC_scale|antithetic_residual|7|0.14131871033314905|5.534634940593852|
|HK_scale|random|None|0.6591563993286383|9.582540202712716|
|HK_scale|moment|None|0.36466918497064676|4.163684921982225|
|HK_scale|normalized_moment|None|0.3637010567044953|4.146263016262982|
|HK_scale|independent_residual|0|0.17893130438521446|3.0679942335084314|
|HK_scale|antithetic_residual|0|0.1549758265708062|2.9266019481231855|
|HK_scale|independent_residual|1|0.18436604026848202|2.9224799809493605|
|HK_scale|antithetic_residual|1|0.15621620395407182|2.9470958371363776|
|HK_scale|independent_residual|2|0.16375423403822573|3.1273901473269174|
|HK_scale|antithetic_residual|2|0.15373171457788287|2.9332658625202863|
|HK_scale|independent_residual|3|0.20716389008336267|3.0509652893974195|
|HK_scale|antithetic_residual|3|0.15428279026634886|2.7994677380004664|
|HK_scale|independent_residual|4|0.1737478588959632|2.883268892558293|
|HK_scale|antithetic_residual|4|0.15411256506317322|2.957429933132699|
|HK_scale|independent_residual|5|0.20060711584115534|2.8943992153850373|
|HK_scale|antithetic_residual|5|0.15581561170809421|2.7550078811862186|
|HK_scale|independent_residual|6|0.17761652583249135|2.5892867130297845|
|HK_scale|antithetic_residual|6|0.15933942496719186|2.9193122948128574|
|HK_scale|independent_residual|7|0.19621807288502233|2.8485624146799533|
|HK_scale|antithetic_residual|7|0.16218713609335256|2.892588475977407|

CPU独立驻留224.36202564975247秒，user211.102288，system175.379569，peak RSS3192.5546875MiB；父wall224.41521131899208另列。内部构造/eval/operation包含于驻留，不相加。

先前八轮2876.677951775957秒，累计3101.0399774257094秒。失败[]；缺失null不补0。

|case|退出码|驻留秒|user|system|RSS MiB|timeout|
|---|---:|---:|---:|---:|---:|---|
|antithetic_arithmetic|0|6.775995247066021|5.663867|1.075773|406.70703125|False|
|small|0|7.522858678828925|7.155359|0.936399|434.74609375|False|
|ISIC_scale|0|81.87118098465726|70.381382|72.78984|1430.59765625|False|
|HK_scale|0|128.19199073920026|127.90168|100.577557|3192.5546875|False|

0GPU/医学images/features/checkpoints/val/test/正式更新。每次只使用16点，8批仅审计，不平均替代主门。
新查询先固定再独立采样；不能承诺观察当次样本后再自适应选择的所有查询。生成分布估计无偏不等真实旧分布无偏。
归一化可改变均值，原二阶残差不保证各向同性或PSD。相反噪声边缘分布不变，不能修复分布偏差或保证降低方差。尚无完整序列/分类训练/生产预算证据。

# 相反噪声配对：结果解释

整体冻结门未通过：small8/8，ISIC4/8（0、1、5、7），HK8/8。四独立CPU正常、算术PASS、0工程失败/超时。57方法批次/665类误差，所有96贡献、560批次类geometry、35类残差均保留。零差与负结果不删除。

配对降低了本次8批估计的平均预测方差：small68.298%、ISIC75.324%、HK88.287%左右，精确值见FAILURES_ANON；有限8批不是总体方差保证。枚举同边缘期望损失误差0、梯度2.22e-16、FD6.89e-11通过；相反线性响应反例方差.5→1保留，不宣称普适降方差。

但降低方差仍没有解决ISIC最大误差：配对八批5.527962–5.584598，原moment6.149715的门限5.534743，4批超门。八批平均5.548066也未过，平均只是诊断。最大误差集中类1只是匿名描述，不按该类选点或阈值。方差缩小使误差更稳定地停在门限附近，不等于证明无限采样极限偏差，但已足以否定“配对即可完整过门”。共同stress不同，不能把旧轮5.538539与新轮6.149715当基线训练性能变化。

配对对独立采样并非逐批全优：ISIC批0平均误差更坏；ISIC批1、4、6最大误差更坏。HK批1、4、6、7最大误差更坏；small批7最大误差更坏。HK八批平均预测的平均误差.150373略差于独立.149216。所有正负配对差见CONTRIBUTIONS。原moment/random基线未降低。

生成器仍存在结构性偏差风险：n+sqrt(rho)xi再归一化可改变节点均值，原unit反例偏移.053502；各向同性trace补偿不等于真实方向残差。这里相同边缘分布的配对不能修复这部分。下一步只前瞻检验一个分布改变：节点切空间对称噪声，z=n+sqrt(1-||n||²)u，u为与n正交的单位对称噪声。它在合法域中精确保单位长度并在期望上保留每节点n；不增加节点、记忆更新或可调强度。保留相同配对方式与原配对各向同性控制，单独检查退化投影拒绝、零/单位中心、期望/梯度、有限批次误差。保留n不等于真实类均值精确，切向协方差也不等于真实分布；不能承诺过门或医学改善。

成本：本轮CPU独立驻留224.36202564975247秒，父wall224.41521131899208另列；user211.102288/system175.379569，peak3192.5546875MiB。配对构造small .014608/ISIC .049441/HK .136270秒包含于驻留；缓存512/1024/2944bytes另列。九轮35次CPU累计3101.0399774257094秒；新增GPU0、历史GPU400036.74851310626不变。内部operation/构造/评价不叠加。client/collector未测null。

没有医学数据/val/test/分类器训练/完整序列/生产预算验证；不是GPU-ready。相反噪声本轮科学门失败，不原实现进医学GPU。
