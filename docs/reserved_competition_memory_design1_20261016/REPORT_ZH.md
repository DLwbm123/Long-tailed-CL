# 稳定接口、预留类别竞争与判别记忆：合成 CPU 结果

状态：SYNTHETIC_FACTORIAL_COMPLETE；冻结联合信号门：False。这是新同架构 2×2 合成验证，不是医学 Core/LT 或 GPU-ready 结果。

|规模|stage|方法|Acc %|BA %|旧 BA %|新 BA %|
|---|---:|---|---:|---:|---:|---:|
|small|1|fixed_moment|82.598039|82.169118|null|82.169118|
|small|1|fixed_decision|82.598039|82.169118|null|82.169118|
|small|1|assigned_moment|85.049020|84.926471|null|84.926471|
|small|1|assigned_decision|85.049020|84.926471|null|84.926471|
|small|2|fixed_moment|72.746331|73.289642|72.977941|73.913043|
|small|2|fixed_decision|68.763103|70.197499|65.441176|79.710145|
|small|2|assigned_moment|70.230608|71.199204|69.117647|75.362319|
|small|2|assigned_decision|62.054507|64.704106|59.375000|75.362319|
|small|3|fixed_moment|65.234375|65.085556|65.828360|62.857143|
|small|3|fixed_decision|63.476562|62.078804|62.771739|60.000000|
|small|3|assigned_moment|66.015625|65.035661|67.666596|57.142857|
|small|3|assigned_decision|56.250000|57.567250|60.565857|48.571429|
|ISIC_scale|1|fixed_moment|82.920792|82.677303|null|82.677303|
|ISIC_scale|1|fixed_decision|82.920792|82.677303|null|82.677303|
|ISIC_scale|1|assigned_moment|82.722772|82.085173|null|82.085173|
|ISIC_scale|1|assigned_decision|82.722772|82.085173|null|82.085173|
|ISIC_scale|2|fixed_moment|64.055300|64.361433|64.234667|64.614963|
|ISIC_scale|2|fixed_decision|55.357143|53.083400|44.568660|70.112879|
|ISIC_scale|2|assigned_moment|64.919355|64.904754|65.014660|64.684942|
|ISIC_scale|2|assigned_decision|59.158986|55.563885|50.113406|66.464843|
|ISIC_scale|3|fixed_moment|55.511978|54.437934|54.835523|53.245164|
|ISIC_scale|3|fixed_decision|50.763197|43.318145|39.910970|53.539669|
|ISIC_scale|3|assigned_moment|55.469578|55.053073|55.488249|53.747542|
|ISIC_scale|3|assigned_decision|48.547806|46.150929|44.474069|51.181511|
|HK_scale|1|fixed_moment|47.403685|38.293194|null|38.293194|
|HK_scale|1|fixed_decision|47.403685|38.293194|null|38.293194|
|HK_scale|1|assigned_moment|51.423786|41.374712|null|41.374712|
|HK_scale|1|assigned_decision|51.423786|41.374712|null|41.374712|
|HK_scale|2|fixed_moment|49.025974|43.723354|43.645290|44.230769|
|HK_scale|2|fixed_decision|39.935065|32.803266|32.918956|32.051282|
|HK_scale|2|assigned_moment|49.350649|43.576445|43.475780|44.230769|
|HK_scale|2|assigned_decision|46.590909|37.883000|37.498136|40.384615|
|HK_scale|3|fixed_moment|45.600000|37.013195|41.948288|0.000000|
|HK_scale|3|fixed_decision|33.280000|30.984915|32.116237|22.500000|
|HK_scale|3|assigned_moment|48.000000|42.825673|46.869096|12.500000|
|HK_scale|3|assigned_decision|42.080000|40.547346|41.286993|35.000000|
|HK_scale|4|fixed_moment|40.405405|37.706238|41.043195|9.342105|
|HK_scale|4|fixed_decision|32.702703|28.089211|27.879893|29.868421|
|HK_scale|4|assigned_moment|42.972973|40.287107|43.200141|15.526316|
|HK_scale|4|assigned_decision|39.729730|39.018869|39.863195|31.842105|
|HK_scale|5|fixed_moment|37.247924|37.147643|41.057922|0.000000|
|HK_scale|5|fixed_decision|29.062871|23.782727|25.860866|4.040404|
|HK_scale|5|assigned_moment|39.145907|40.847837|45.147609|0.000000|
|HK_scale|5|assigned_decision|35.943060|35.494148|38.273436|9.090909|
|HK_scale|6|fixed_moment|30.443160|34.124414|37.374358|0.000000|
|HK_scale|6|fixed_decision|25.529865|22.272674|23.219184|12.334315|
|HK_scale|6|assigned_moment|31.695568|37.892274|41.402878|1.030928|
|HK_scale|6|assigned_decision|31.599229|32.019605|33.306796|18.504103|

全部类别见 CLASS_RECALL；各组救回/破坏/错换错和错误流向见 PAIRED_PREDICTIONS；分配、记忆简单效应及交互见 CONTRIBUTIONS；同状态记忆误差见 MEMORY_DIAGNOSTICS。没有遗漏的失败补零。

独立子进程 CPU 驻留 731.059669525 秒；加此前一次累计 6593.252146675 秒。父 wall 731.1072428212501 秒另列；峰值 RSS 1695.3984375 MiB。内部训练、压缩、分配、评价含在驻留内不再相加。新增 GPU 0，历史 GPU 不变。

失败条目：[]。具体匿名失败说明见 FAILURES_ANON（若有）。

固定输入空间与旧 slot/记忆不变不代表适配器后特征或分类不变。有限 sliced-W1 和两快照梯度匹配没有任意未来查询保证。固定接口与教师保持为共同条件，本轮不单独证明其贡献。无生产 checkpoint/真实 encoder/医学数据验证。所有系数与数据事前冻结，不据结果改门或追加预算。
