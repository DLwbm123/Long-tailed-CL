# GAIN：ISIC/HK 冻结验证协议

用户直接授权在ISIC/HK复用base/local_transport原18stage checkpoint验证GAIN，无重训/调参/旧图回放/生成旧features，test/reserved封存。设计源aff748ee9ea4f926819db51bb46749a0881ef537；独立生产science source在启动前公开冻结，不把CPU设计源或后续报告commit当部署源。主控制local_transport，主模块GAIN，四head和全部参数不因val结果改选。

## 固定数据、来源和机制

原科学源base 0cfe6b648f5de1b8e8783265e7a0b97b64939952 / local_transport 0f0d8a0717c6da09109cc9205cccf44c490d3e84及其独立delivery，详细order/task/count/18stage参考均在PROTOCOL_LOCK。ISIC18718train/295val、HK6821train/1698val，仅已有val读入评价，原训练图不读。原权重/双adapter编码器/固定eval变换Resize224/Normalize(.5)严格复用，不下载模型或数据。

GAIN按联合原/候选赢家估计救回与破坏，每类1/K；先验不补支持量、不改Q/R/pairs或均值，实际uniform质量=count/n验收。n/(n+16)只影响供体/不确定范围，不降低稀有类宏权重。旧contrast方向由原native曲率求解，原double最优解重构与原float32头逐元素/相对误差核对；.001R/.5competition保留。能量radius1、固定scale0/.125/.25/.5/1、净门>1e−12和其他设计参数原封不变。三个贡献头为GAIN_NO_BORROW、GAIN_RESCUE_ONLY、NEUTRAL_MATCHED；没有ETF、RL、optimizer或追加预算臂。

实现共享原几何预条件器；GAIN与RESCUE_ONLY共用同一ray、同五候选模型区间，仅选择准入目标不同。CPU小维逐head与冻结helper一致。NO_BORROW改变整个借用提案及情景路径，不能称单独风险门消融。NEUTRAL匹配主GAIN实际接受delta的守卫前native能量，主GAIN拒绝时也是0，不据此称方向有效。

float32量化位模式与原设计一致。高维压力检查发现固定绝对1e−8的旧零和校验可能误拒正常量化；仅改为IEEE float32舍入包络1e−8+eps32*sum(abs(delta_old))，拒绝超出包络的公共项。未重新居中、截断/重标度delta，也未改收益门或参数。公开实际旧列和残差及包络；理想旧contrast保持公共项，实际保存float32可能有舍入残差。不可修复的旧均值下界在评价中用min(原旧max,RAW旧mean)核对，保留数值差异。

## 高维CPU、生产准备与真实GPU屏障

启动前完整1536×23合成CPU检查须包含有实质非零提案且PCG确实迭代的原系统，而不只跑guard接口或数值零force。同测两控制、四head、dense小维oracle/共享计算一致、侧车roundtrip、新列0、stage-one identity、非法scope/错误source/budget/分母拒绝，以及22行/40贡献完整报告fixture。全部成功与失败CPU驻留独立记录，pipeline包含其中，不相加。CPU合成不是医学效果或真实GPU通过。

唯一生产程序固定四轨迹ISIC_base→ISIC_local_transport→HK_base→HK_local_transport、两个slots。首先CPU准备4轨迹所有18stage、72侧车；逐stage核对旧资产来源、版本、seed/order/task/count/旧budget、原OFF重构、统计支持和原文件size/mtime未变。无GPU/val读入。四个PREPARED精确齐全封存后才进入GPU预检。

真实零更新预检四条各固定最终stage首64个已见val影像；CPU/GPU四头读出finite/一致、冻结编码器全部state精确、0optimizer/policy、testfalse，不输出分类指标。4/4 PASS封存后才统一EVALUATE。worker在val.csv打开前逐次核对phase/四PREPARED/source/侧车版本，以及评价时四预检PASS，不允许提前读取val分类指标。

只用GPU0/1，launch及每GPU子进程>=32768MiB准入；保护其他任务，不借2/3/杀进程。实际NAS nfs/free/小读写probe，每卡GPU准入失败停止、不自动retry。中性父子入口和私有env/TMPDIR，ps完整命令和NVML启动核实。

## 固定曝光、成本和停止

CPU四准备，GPU4预检+4共享评价=8次，所有出现的process residence逐次记录；预检256图、18stage统一val15966图，总16222。四头不重复图像/编码器费用。原控制checkpoint只读复用；AIR只用本地公开结果，重评/费用0。源checkpoint、delta/模型银行、样本/逐样本分数/features/indices/原始log/privateconfig永留NAS私有。

全程序deadline14400秒从真实launch开始，包括准备/等待/预检/评价，不重置。失败保留私有明确原因、全部成功失败驻留，停止依赖，不能自动retry/放宽门/重冻修复科学轮。全部identity也完整交付，没有自动下一实验。没有创建/恢复monitor；按实验后台规则启动一次确认后即可返回，后续询问检查实时状态，不持续轮询。

## 完整主四格、贡献和评价

主四格A0=base/AL=local_transport/B=base+GAIN/P=local_transport+GAIN。原控制+4head为20行主表，加AIR2参考行，全部六指标finalBA/avgBA/tail/forget/old/new及final/avgAcc、已有macroF1。40固定对比：16 head−OFF、12 GAIN−各贡献头、2 P−B、2交互、8主四格−AIR；raw完整16行和guarded−raw贡献另列，所有正零负，不挑最好。

1260主readout stage-class n/recall行；另1008 raw stage-class行；31类各stage/终局救回破坏/错转错、旧旧旧新新旧新新全流、margin/normalized margin/作用量/守卫移除或救回、全部72 identity状态。每类预测救回/破坏区间、macro净界与实际对应事件、可修复原旧赢家质量、不可修复原新赢家质量/均值floor、原支持与供体权重/有效数、五候选区间/步长及原能量都保留。n<5连续margin摘要抑制，离散n/计数/recall不抹除。矩误差范围未医学校准；实际落入区间不称置信覆盖证明。

forget与历史peak/终局变化一并报告，peak下降不称保持改善。开发门相对各自控制BA>=+1pp、tail>=−.5pp、forget<=+1pp、old>=−1pp、新类recall不下降；两个数据集/两主模块格均核对，不凭样本净纠正数或某小分母类称宏收益通过。单seed/order、反复development val，非独立患者/临床/封存test验证或统计显著。

COMPLETE且active空、全部worker实际退出后collect一次，严格核对4COMPLETE、72侧车、4真零、8成功GPU01、18stage/分母/OFF精确、finite/residual<=1e−8、全部费用。INCOMPLETE保留PARTIAL_RESULTS/缺失/失败原因和费用，不补0、不称完整比较。真实结束完整源/协议/正负报告/结果/贡献/AUDIT/cost显式代理push feat/prototype-analytic，remote SHA和精确报告匿名HTTP200后NAS独立public_delivery/DELIVERY/HANDOFF。尚未结束不得称完成交付。
