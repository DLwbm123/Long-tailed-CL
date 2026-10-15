# 固定子空间半径：完整分析

冻结主门（算术+两个高维每批）通过；small单列而非事后排除。只是有限合成查询信号，非医学贡献或GPU-ready。

## small
通过批次{'gaussian': [0, 1, 3, 5, 6], 'block_spherical': [0, 3, 4, 5, 6]}；最大门限0.8063005046418036、mean门限0.09467740318664856。
有限8批mean prediction variance相对Gaussian +2.744653%，非总体定理。
原baseline构造0.2802151390351355秒；共用谱构造0.007268228102475405秒；评价0.5033040381968021秒，均在驻留内。
baseline/Gaussian/candidate memorybytes=16896；fresh16点=16384，谱构造仅计一次。

相对Gaussian mean_delta: 改善6，相等0，变坏2。
相对Gaussian max_delta: 改善4，相等0，变坏4。
8批平均只诊断：gaussian_eight_batch_average_DIAGNOSTIC_ONLY mean 0.035550818449056276, max 0.26082881292364。
8批平均只诊断：block_spherical_eight_batch_average_DIAGNOSTIC_ONLY mean 0.03509404999382925, max 0.2559191143224777。
## ISIC_scale
通过批次{'gaussian': [0, 1, 2, 3, 4, 5, 6, 7], 'block_spherical': [0, 1, 2, 3, 4, 5, 6, 7]}；最大门限1.9367260992370607、mean门限0.2625020357757948。
有限8批mean prediction variance相对Gaussian -0.475236%，非总体定理。
原baseline构造42.39215867780149秒；共用谱构造11.784598890691996秒；评价36.34469094034284秒，均在驻留内。
baseline/Gaussian/candidate memorybytes=1573888；fresh16点=1572864，谱构造仅计一次。

相对Gaussian mean_delta: 改善5，相等0，变坏3。
相对Gaussian max_delta: 改善5，相等0，变坏3。
8批平均只诊断：gaussian_eight_batch_average_DIAGNOSTIC_ONLY mean 0.047914278374824114, max 1.1196630473034825。
8批平均只诊断：block_spherical_eight_batch_average_DIAGNOSTIC_ONLY mean 0.047387756092128815, max 1.1118069881302732。
## HK_scale
通过批次{'gaussian': [0, 1, 2, 3, 4, 5, 6, 7], 'block_spherical': [0, 1, 2, 3, 4, 5, 6, 7]}；最大门限2.870842280395719、mean门限0.2745957371726634。
有限8批mean prediction variance相对Gaussian -12.228036%，非总体定理。
原baseline构造98.25712823821232秒；共用谱构造23.560285876039416秒；评价10.239654592704028秒，均在驻留内。
baseline/Gaussian/candidate memorybytes=4524928；fresh16点=4521984，谱构造仅计一次。

相对Gaussian mean_delta: 改善8，相等0，变坏0。
相对Gaussian max_delta: 改善3，相等0，变坏5。
8批平均只诊断：gaussian_eight_batch_average_DIAGNOSTIC_ONLY mean 0.071238873098518, max 0.9852062637111789。
8批平均只诊断：block_spherical_eight_batch_average_DIAGNOSTIC_ONLY mean 0.061142366747739804, max 0.9275623726673885。

## 正负解释
两个高维每批均优于原moment/random门，主协议通过。Gaussian同轮也两高维8/8，不得把前轮HK4/8到本轮8/8的变化归因固定半径；共同stress方法集合改变，不是跨轮相同查询。
HK candidate平均误差8批全优于Gaussian，但最大误差仅3好5坏；ISIC mean5好3坏/max5好3坏；small mean6好2坏/max4好4坏。small candidate批1/2/7 mean坏于moment，只有5/8；原主门明确只要求两个高维，不将small表述全部通过。
raw矩cubature误差6.94e-18、FD9.23e-12、rank1两点与零协方差/退化拒绝通过，仅实现性质；不保证一般非高斯类分布/归一化矩/所有未来查询。只有有限8批方差变化，不能作总体定理。
低样本统计可包含原支持信息，当前全合成，不自动授权医学原始特征回放。没有完整序列、encoder或classifier训练、重压缩及生产预算证据。后续先冻结完整序列CPU，保持原16点/15方向/48baseline步/门，不进入GPU。
4CPU独立驻留260.43532313080505秒；13轮51CPU累计4342.275483778678秒，新GPU0；旧GPU400036.74851310626不变。父wall260.48516097897664另列；内部operation不相加，client/collector未测null。0工程失败/超时，全部正负保留。
