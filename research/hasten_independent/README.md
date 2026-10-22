# 当前运行：r06 增量adapter零初始化

r05均衡采样主门失败（平均31.2197%、最终24.6667%），保留负结果并回到r04单位化参照。训练侧发现随机新adapter在优化前引起约35–37%旧图像表示变化；r06仅测试增量adapter零初始化，3seed配对。参见[r05结果](r05_balanced/RESULTS_ZH.md)和[r06协议](r06_zero_init/PROTOCOL.md)。

# 当前运行：r05 类别均衡优化采样

r04三种子单位化平均32.3549%、最终26.3333%，通过预注册配对开发门且距离饱和解除，但仍低旧路线。正在以固定训练步数检验均衡采样改善尾类。参见[r04结果](r04_unit_scale/RESULTS_ZH.md)和[r05协议](r05_balanced/PROTOCOL.md)。仍遵守12小时/累计新增30GB窗口。

# 当前运行：r04 输入尺度修正

r03 BiomedCLIP两臂平均28.8446%、最终22%，双曲分支发生距离饱和。已启动r04三种子raw/unit/unit-no-proj配对，固定单位化图文输入，其余HASTEN设置不变。参见[r03结果](r03_biomed_nih/RESULTS_ZH.md)和[r04协议](r04_unit_scale/PROTOCOL.md)。仍在用户12小时、累计新增30GB窗口内。

# 当前运行：r03 BiomedCLIP NIH

用户已明确授权立即开始。同NIH划分/顺序/训练参数，两臂只更换BiomedCLIP骨干系统。协议见[r03_biomed_nih/PROTOCOL.md](r03_biomed_nih/PROTOCOL.md)。独立8小时/8GB，唯一hasten监测已更新；保留r02负结果。

# 当前状态：r02 NIH完成

两臂12阶段、21340更新完成并通过预测计数核对。完整版平均19.8448%/最终13.6667%；关闭投影19.5988%/14.3333%，明显落后旧Spatial及CLS＋patch。详见[r02_nih/RESULTS_ZH.md](r02_nih/RESULTS_ZH.md)。保持HASTEN独立路线自主分析与后继，唯一监测继续。


# HASTEN独立研究路线

2026-10-22用户授权：先做必要修复和验收，再跑短增量实验及四臂消融；结果出来后自行分析继续。与原方法独立，原方法不因本次授权恢复或改变。

固定官方源码提交：`063f295d08991d4da5b1f46a61bfcd279a94dbc6`。

阶段：r01已完成36阶段、4560次更新，工程验收通过；三个预注册完整效果/保护门均未通过。独立路线保持自主跟进，原方法仍暂停。[本轮结果](r01/results/ANALYSIS_ZH.md) · [数值比较](NUMERICAL_COMPARISON_20261022_ZH.md) · [协议](r01/PROTOCOL.md)。此前简化初始任务结果不能代替完整增量机制检验。新实验单独记录，使用已暴露支持数据时明确标为探索性开发证据。每个实验新增存储严格小于50GB、原始墙钟不超过24小时；重试不重置时钟。只保留当前会话一个30分钟监测。

计划四臂：H-full、H-no-tree（保留图文锚定）、H-shuffle、H-no-proj。正式配置、适配差异、验收和结果将在此目录发布；私有数据与模型不入库。

- r06零初始化负结果：[报告](r06_zero_init/RESULTS_ZH.md)。r07父节点文本适配器一致性：[预注册](r07_parent_text/PROTOCOL.md)。

- r07父文本路由负结果：[报告](r07_parent_text/RESULTS_ZH.md)。r08全部已见类分类目标：[预注册](r08_class_ce/PROTOCOL.md)。
