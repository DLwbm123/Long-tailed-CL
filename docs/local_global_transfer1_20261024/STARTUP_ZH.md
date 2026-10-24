# 原生局部与全局迁移：真实资格通过及科学启动

状态：RUNNING_QUALIFIED_BASE_ONLY_LOCAL_GLOBAL_TRANSFER。原部署源65b2203c4f2167d693e81d308fa7363c15739608。所有真实数值、完整链与预算门均PASS后，既有worker已开始正式Native基类编码；启动快照至少4008/60129图（数值journal已到4160），正式优化0、val0，无性能结果。资格文件中的science_started=false仅指资格模型自身，不覆盖之后的正式运行状态。

方法在实际base图上学习同图双view局部栅格对应与全局强度不变性；原ViT最后4块及norm在base FP64训练，前8块固定，明确没有LoRA。base结束整个map永久冻结，后来类别仅更新等质量患者类全矩ridge联合系统。推理full1536原生CLS+patchmean；投影器与监督临时head丢弃，无旧cache、分数融合、任务ID路由、未来类初始化或非线性旧矩运输。局部/全局自监督是成熟机制的本项目医学适配，不声称忠实VICRegL复现、新颖性、医学效果或临床保证。

|臂|base表示训练|后来优化|推理判别|
|---|---|---|---|
|NATIVE_RIDGE|无|0|同格式全seen患者类ridge|
|BASE_SUPERVISED|配对view监督CE|0|丢弃CE头后同ridge|
|GLOBAL_REDUNDANCY|同图全局冗余约束|0|丢弃projector后同ridge|
|主LOCAL_GLOBAL_REDUNDANCY|.5全局+.5对应栅格局部约束|0|丢弃projector后同ridge|

三个学习臂相同原生起点、实际draw、两view增强及2000次base更新机会；每步8基类各4患者有放回=32逻辑图/64views，不称32独立患者。sampler使用base类别元数据，故不是完全无标签预训练。目标头及活跃参数/FLOPs不同披露。AdamW3e-5/wd1e-4/eps1e-8/clip5/micro8；后继类0梯度更新。双view同图16px整patch位移及弱亮度变化，reflect边界从局部匹配排除，未宣称增强一定保留病理信息。局部方差/协方差沿draw逐坐标计算，不能用位置差异避免collapse。

真实资格20合法base图：完整原生reader/embedding/前8块/后4块及量化接口，与独立NumPy比较。patch FP64误差7.716050021144838e-15；lower8误差2.8776980798284058e-12、原生upper4误差3.5388358909926865e-16；像素增强接口差0。FP16rounding最大.007373809814453125是实际量化量，不是FP64比较失败。

三个臂各2实际丢弃更新，共6；确定性同canonical micro8双pass重放输出差0。新Torch目标对独立小维解析目标误差7.105427357601002e-15、导数4.440892098500626e-16。真实upper两个指定参数/臂的coupled replay有限差分最大2.9879988278557965e-10（只核指定分量，不声称遍历所有参数导数）；完整same-gradient/same-initial-state独立AdamW最大绝对误差8.881784197001252e-16。全model每臂实际变化约6.01e-5，原native权重固定，保存加载PASS。训练后完整upper4 NumPy误差最大4.85722573273506e-16；8+2到达式全矩读出最大误差7.19979631469414e-14。前述是完整链、目标、指定导数与优化算子的组合证据，不是旧full64/micro8轨迹等价，也不恢复任何历史缺失误差。

精确batch耦合：先收集64view原生token，完整coupled目标求导一次，再按相同micro重放上层并传播output梯度。不能替换为micro目标平均。所有重算/探针和丢弃更新保留。资格实际canonical20编码、额外patch embedding20行、增强128view编码；三臂各128首pass/128重放前向/128反向行。额外梯度探针384前向/192反向，有限差分768前向，目标GPU/NumPy各192行，训练后upper GPU/NumPy各60行，以及3臂各2到达式系统另计。小维纯标准库121调用/1210行及81几何检查、Torch首次目标比较与更早未保存准备细分保留，不补0；全部旧科学与资格失败成本继续累计。

实测最慢三臂canonical pairedstep6.69571527838707秒，冻预算1.5*2000*pairedstep+14400=34487.14583516121秒，小于当时原剩余52710.54011011124秒减60。只排除单列的资格独立算子参考开销（两次paired全资格89.447267/85.574371秒，其中独立算子82.751551/78.988023秒）；这些开销仍保留在原总钟内。PASS是准入，不是ETA。

实际NFS4小写读、hostfree40GiB/GPUfree8GiB均PASS。实文件max(size,blocks512)比率加10%系数2.2071379559365654，物理预测5345470225.70693B<5.4GB预停、6GB硬限，预测不等最终峰。资格GPUreserved峰5213519872B<8GiB，allocator7GiB留context；nvidia-smi进程0MiB采样不能作无GPU或完整峰证据。启动RSS采样峰4219092992B<32GiB，新根入口采样峰1513240806B，累计窗口末采样15539260090B<remote27GB预停；原30GB/本地100MB余量窗口不重置、不删旧证据腾额。

原研究first1792820251.646123、deadline1792877851.646123，16h包含全部阅读、实现、传输、等待、资格、科学及独立验收；不另起远端16h或延钟。独立于NFS的整组timeout截止前60秒TERM/30秒KILL，5秒资源guard；任何失败/预停/截止按冻结停止，不重试、救分或换主臂。

完成前不下性能结论。四map和24系统先seal，val300仅编码一次；独立完整8+4层query300+训练proof40、4map1200唯一query特征、24读出5400score/全预测一致、实际训练batch的post-update endpoint、在线全患者矩复核及原17门/对GLOBAL机制4门均需验收。验收不新增原图编码或完整优化轨迹重放。全组退出、完整中文四臂表、全部失败/未知成本以及代理推送/远端SHA/匿名验证之后才算最终交付。

只公开源码与小聚合。身份、图像、tokens、feature、权重、矩、head、逐图score及原始日志私有。300val已反复用于开发，不是临床test，不用于参数选择或SOTA声称；不访问test/Etest或新数据，不恢复已关闭路线。
