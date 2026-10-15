# 谱分布记忆：完整分析

整体冻结门未过；ISIC8/8通过，HK4/8通过，小规模6/8通过。均为合成近似误差，不是医学准确率。

## small
逐批通过[0, 1, 3, 4, 5, 6]，最大门限0.7864375935841696；平均门限0.09619073335623314。
有限8批mean prediction variance相对isotropic变化-5.664086%，不是总体方差定理。
原基线构造0.36461789114400744秒；谱分解0.08134183334186673秒；评价0.3764898730441928秒，均已包含驻留。
candidate/baseline bytes均16896；isotropic 1056；fresh16点16384。不是零工作内存/计算等价。

batch0最坏匿名类1，合成count=136，max=0.5940876235585724。
batch1最坏匿名类3，合成count=35，max=0.6794266176477792。
batch2最坏匿名类1，合成count=136，max=0.7149000899079493。
batch3最坏匿名类1，合成count=136，max=0.458315285452066。
batch4最坏匿名类1，合成count=136，max=0.6093971127574014。
batch5最坏匿名类3，合成count=35，max=0.5576418912724159。
batch6最坏匿名类2，合成count=69，max=0.623251578964247。
batch7最坏匿名类2，合成count=69，max=0.5592599900949806。
只诊断八批平均isotropic_eight_batch_average_DIAGNOSTIC_ONLY: mean=0.0420740963357561, max=0.2816694594637269；不代替主门。
只诊断八批平均spectral_eight_batch_average_DIAGNOSTIC_ONLY: mean=0.035634294549284705, max=0.2379935031633984；不代替主门。
## ISIC_scale
逐批通过[0, 1, 2, 3, 4, 5, 6, 7]，最大门限1.8243368303310903；平均门限0.2288546680757737。
有限8批mean prediction variance相对isotropic变化89.888669%，不是总体方差定理。
原基线构造36.86958563607186秒；谱分解9.593222757801414秒；评价28.888075528666377秒，均已包含驻留。
candidate/baseline bytes均1573888；isotropic 98368；fresh16点1572864。不是零工作内存/计算等价。

batch0最坏匿名类4，合成count=1429，max=0.5528509103773821。
batch1最坏匿名类5，合成count=23，max=0.7594158179389972。
batch2最坏匿名类1，合成count=14，max=1.1847665741076305。
batch3最坏匿名类1，合成count=14，max=0.7797854375080489。
batch4最坏匿名类1，合成count=14，max=0.890177415015192。
batch5最坏匿名类2，合成count=130，max=0.8169941740193043。
batch6最坏匿名类2，合成count=130，max=0.74551249783985。
batch7最坏匿名类5，合成count=23，max=1.0566284086513513。
只诊断八批平均isotropic_eight_batch_average_DIAGNOSTIC_ONLY: mean=0.09588401232692433, max=2.5482284241064423；不代替主门。
只诊断八批平均spectral_eight_batch_average_DIAGNOSTIC_ONLY: mean=0.041144675752363385, max=0.632682281362372；不代替主门。
## HK_scale
逐批通过[1, 5, 6, 7]，最大门限2.771119398126121；平均门限0.2883539830173883。
有限8批mean prediction variance相对isotropic变化172.696835%，不是总体方差定理。
原基线构造98.7075543240644秒；谱分解21.384557956364006秒；评价8.937312616035342秒，均已包含驻留。
candidate/baseline bytes均4524928；isotropic 282808；fresh16点4521984。不是零工作内存/计算等价。

batch0最坏匿名类20，合成count=4，max=3.3615404159083253。
batch1最坏匿名类20，合成count=4，max=2.5760591890357523。
batch2最坏匿名类20，合成count=4，max=2.9005905399010956。
batch3最坏匿名类12，合成count=2，max=3.013430857113457。
batch4最坏匿名类20，合成count=4，max=3.586819831893454。
batch5最坏匿名类20，合成count=4，max=2.0286235716507193。
batch6最坏匿名类20，合成count=4，max=2.5878464734334994。
batch7最坏匿名类12，合成count=2，max=2.019111998727081。
只诊断八批平均isotropic_eight_batch_average_DIAGNOSTIC_ONLY: mean=0.28377574842070474, max=5.8487094793092425；不代替主门。
只诊断八批平均spectral_eight_batch_average_DIAGNOSTIC_ONLY: mean=0.10373931344178744, max=2.650300512940956；不代替主门。

## 机制与负结果
ISIC spectral每批mean/max均优于isotropic且全过原门。HK spectral也每批mean/max优于isotropic，但batch0/2/3/4未过最大门，其中0/4甚至比原moment最大误差更坏。小规模batch2/7平均误差超过moment。
原始协方差Fro误差降低不代表归一化后的高阶损失或采样方差降低；高维mean prediction variance均上升，不能称降方差方法。HK低样本类已有近乎精确raw协方差，但Gaussian形状及归一化仍可能造成误差。只凭这些结果不能归因某一机制；无完整loss积分/因果证书。
候选与原moment每类16d+16存储精确相等，isotropic更小，方向结构贡献不能误写成同容量isotropic控制。eigh临时协方差/工作区不部署，资源由RSS记录。HK单支持类确定性近零误差是退化fixture，不是医学稀少类收益；统计可能暴露低样本原支持，本轮只合成。
0工程失败/超时，缺失client/collector成本null。4子进程驻留239.64248802978545秒；十二轮47次合计4081.840160647873秒。新GPU0，旧GPU400036.74851310626秒不变。父wall另列，不叠加内部operation。
未检查序列/encoder或classifier训练/医学数据/生产预算，不GPU-ready，不改变主门。下一轮只能机制检验，不能加节点、强度或采样批挑结果。
