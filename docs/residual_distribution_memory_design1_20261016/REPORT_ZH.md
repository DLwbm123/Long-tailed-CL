# 随机残差分布记忆 CPU 结果

NO_COMPLETE_FRESH_BATCH_SIGNAL
冻结门：{'arithmetic': True, 'small': False, 'ISIC_scale': False, 'HK_scale': True}

|规模|方法|批次|平均误差|最大误差|
|---|---|---:|---:|---:|
|small|random|None|0.45350251371561145|4.075733304953106|
|small|moment|None|0.11284992097326098|0.988301295426318|
|small|normalized_moment|None|0.11134644051126531|0.9177445704006066|
|small|residual_distribution|0|0.13547086963154445|0.9467661281011819|
|small|residual_distribution|1|0.13005730152404527|0.8163059076683501|
|small|residual_distribution|2|0.1252895532894389|0.6631759352521196|
|small|residual_distribution|3|0.13635704695610662|0.7568577587950491|
|small|residual_distribution|4|0.13735063971749376|0.7226150941505336|
|small|residual_distribution|5|0.12560137403495716|0.8575019901912553|
|small|residual_distribution|6|0.14815030198547477|1.0402739194336723|
|small|residual_distribution|7|0.1353333007672328|0.5138823572727762|
|ISIC_scale|random|None|0.8595924895045198|9.785622872249956|
|ISIC_scale|moment|None|0.3845185019123322|5.538539311773269|
|ISIC_scale|normalized_moment|None|0.3832221212315557|5.525830853232792|
|ISIC_scale|residual_distribution|0|0.12720967949740994|5.27287303791209|
|ISIC_scale|residual_distribution|1|0.12992759743289534|4.42687708274115|
|ISIC_scale|residual_distribution|2|0.1920265868923533|5.362550380607232|
|ISIC_scale|residual_distribution|3|0.1594499875665287|5.3001838547853595|
|ISIC_scale|residual_distribution|4|0.1552986509249097|4.705360660268939|
|ISIC_scale|residual_distribution|5|0.14459651859496228|5.359097858647155|
|ISIC_scale|residual_distribution|6|0.15832872583127908|4.826805074052064|
|ISIC_scale|residual_distribution|7|0.16222447447630114|4.885988137415945|
|HK_scale|random|None|0.6787070194381141|9.600021340635934|
|HK_scale|moment|None|0.3469451937816254|4.238027309039489|
|HK_scale|normalized_moment|None|0.3459742698319176|4.220374375144622|
|HK_scale|residual_distribution|0|0.1704643890436353|2.4981520691783627|
|HK_scale|residual_distribution|1|0.1741195787346487|2.4015193807236432|
|HK_scale|residual_distribution|2|0.1537369000560565|2.663407451718995|
|HK_scale|residual_distribution|3|0.19502749843057768|2.524653503174258|
|HK_scale|residual_distribution|4|0.1575744500754725|2.872538806766687|
|HK_scale|residual_distribution|5|0.18557191098245215|2.3853451813995656|
|HK_scale|residual_distribution|6|0.1623169395717527|2.1894652854635943|
|HK_scale|residual_distribution|7|0.1785325052220445|2.5171022814038855|

CPU独立驻留214.61660983832553秒，user193.463152，system171.886767，peak RSS3272.46875MiB；父wall214.6673744339496另列。内部构造/eval/operation包含于驻留，不相加。

先前七轮2662.0613419376314秒，累计2876.677951775957秒。失败[]；缺失null不补0。

|case|退出码|驻留秒|user|system|RSS MiB|timeout|
|---|---:|---:|---:|---:|---:|---|
|distribution_arithmetic|0|7.271133493166417|5.584816|1.142256|406.0625|False|
|small|0|7.777618984691799|7.18611|1.10187|435.01953125|False|
|ISIC_scale|0|88.38305753422901|79.271321|76.392653|1606.84375|False|
|HK_scale|0|111.1847998262383|101.420905|93.249988|3272.46875|False|

0GPU/医学images/features/checkpoints/val/test/正式更新。每次只使用16点，8批仅审计，不平均替代主门。
新查询先固定再独立采样；不能承诺观察当次样本后再自适应选择的所有查询。生成分布估计无偏不等真实旧分布无偏。
归一化可改变均值，原二阶残差不保证各向同性或PSD。尚无完整序列/分类训练/生产预算证据。
