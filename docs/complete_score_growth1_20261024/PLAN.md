# 完整旧类别分数冻结与新类原生上层增长：冻结协议

协议状态 PROTOCOL_FROZEN_IMPLEMENTATION_PENDING。不是RUNNING；实现和所有实际资格完成前禁止科学bulk。独立结构依据和与旧路线差别见DESIGN_DRAFT_ZH，旧MIL/upper/expansion不恢复，旧失败不追改。

## 假说与对照

固定输入坐标与旧完整类别分数函数可以保留既有医学判别，同时每批新类获得可学习的完整原生上层；匹配FEATURE_FROZEN保留旧head可塑性，隔离完整函数冻结。两增长臂同总容量/新expert初始化/实际患者draw/优化配置；NATIVE_RIDGE同输入完整患者类矩强锚点。第一阶段两增长臂同状态、同数学训练且应相同；后续旧head可塑性是唯一结构干预。旧功能固定不保证allseen预测固定，新类竞争和弱基阶段保留为失败可能。

## 数据与训练

沿用原NIH train68058、val300、固定20类顺序和10+2×5到达，患者train/val隔离；不访问test/Etest/future数据或以val选参数。原始预训练ViT-B/16前10块永远冻结；image embedding FP16量化后lower10在FP32显式attention，lower输出存FP16空间token197×768。新expert每次从同一原始最后2块+norm复制，FP32运行，25*cos([CLS,patchmean],head)输出仅属于本次到达类。所有seen输出拼接统一CE，无taskid路由/融合/128压缩。输入精度接口独立记录，不冒称与历史75/300锚点同一数值路径。

每类head初始化为当前类患者等质量原始upper描述均值的单位方向；只有到达后初始化。本stage current完整图可用；旧只保留每类32患者、每患者1图的lower token；32按seed74003固定患者优先键和图像键排序。两臂完全相同实际draw：每步8个seen类不放回，每类8患者有放回，再在患者图内有放回，共64行，非64独立患者。base3000更新，后续每stage500，5500/臂、共11000科学更新。AdamW lr3e-4/wd1e-4/eps1e-8/betas(.9,.999)，每stage重建同规则优化器、micro8累加sum CE/64、全可训练参数统一clip5。无增强/dropout/TF32/搜索/额外seed。

旧expert upper/norm两臂均冻结，主臂旧head也冻结，控制旧head可变。旧upper特征由本轮当前合法token/有限记忆按本臂已封存参数生成，stage内仅RAM缓存，控制由缓存特征训练旧head，主臂由缓存特征生成固定旧score；不借历史cache、不持久保存全train feature。真实生成/读取/训练费用逐项记账。Native对每到达类用全部当期患者等权、类等权1537维带常数特征全矩ridge1e-4，保存每类均值/累计矩用于独立重建；格式与token记忆不同，不称等内存。

## 验收与门

三臂×六stage共18模型在val读取前全部seal；一次300val image encoding，每stage仅seen query，每类15。独立NumPy核对全部18组4050score行、prediction完全一致、原生lower10+upper2全链query与40训练proof、实际draw/患者记忆/到达、旧函数冻结、同原始expert初始化、成本和期限。每stage保存最后一次更新前64行probe，独立NumPy复算CE；不冒称重放完整11000步轨迹。

主门13项：绝对final≥91/300、mean≥36.005291005291%、tail7≥25/105、initial10≥44/150、later10≥41/150；对Native与FEATURE_FROZEN各final≥+4/300、mean≥+1pp、tail4≥+2/60、later10不降。不是旧17门减少后的同实验：新方法三臂，旧四臂17门与旧失败永久保留。机制四门为FULL对FEATURE的四个相对门。完整中文表报告所有stage和tail/initial/later分母，无选择性报告。

真实资格前先独立小维目标/导数和两阶段冻结/可塑/保存加载检查；实际GPU使用20合法基类图分8+2类，两臂各两次实际micro8更新，检查旧upper/主旧head精确不变、控制旧head确有更新、新upper确有更新、输出顺序和cache/直接路径一致，FP64/NumPy目标误差≤2e-6，FP32分数及probe≤5e-5。同保存梯度/同初态独立递推AdamW 逐坐标参数误差≤3e-7+1e-6×|独立预期参数|，报告绝对误差和最大归一化误差（FP32权重尺度舍入界），落盘再assert。规范实现是micro8，不以full64轨迹等价作本轮准入，不追改MIL联合失败或再跑旧比较。完整GPU峰<8GiB且hostfree40GiB、实际NFS写读探针通过后，以1.5×5500×真实两臂配对step+10800s严格小于原剩余钟减60s才bulk；测最差两阶段配对step，非ETA。

## 硬预算与停止

从第一次本方法实际数学/资格运行起12h共钟，包括后续准备/传输/等待/正式/验收/重试，无远端另起钟，不自动重试失败资格。新根+中性入口硬4e9B、3.6e9预停；预计<3.1GB含12个expert upper各仅保存一次约681MB、原始upper57MB、矩/封存/head、query/proof、有限memory和临时写副本。RSS32GiB、GPU8GiB；原累计30GB/remote27GB预停和local100MB余量更严不变，所有旧失败根和入口累计。不删证据。独立于NFS的整组GNU timeout截止前60s TERM/30s KILL及资源guard；父子argv中性。任何准入/资源/权限/数值失败关闭，不放宽或换参考救分；科学一次bulk，末独立验收、全组退出、中文完整成本报告及有效7897proxy push/SHA/匿名确认后交付。当前未运行，不声称峰值或可行预算已通过。
