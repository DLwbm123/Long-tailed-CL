# HASTEN独立路线 r02：旧NIH协议数值比较

2026-10-22运行前冻结。用户要求在旧Long-tailed CL的NIH划分和增量协议上运行HASTEN，与Spatial、CLS＋patch比较；独立路线，有结果自行分析继续。r01结果保留，不重跑HK，不修改旧对话项目或其进程。

## 数据与可比边界

只读复用Spatial的原始manifest和author_labels：68058 train、300 val（每类15），患者无交集；不打开test。原始类别顺序[13,19,6,14,17,1,9,4,15,8,18,2,0,12,16,3,7,11,5,10]，10+2+2+2+2+2六阶段。核对每行原始标签、单标签和患者ID来源。训练侧<250图的7类为尾类。标签重映射只为连续增量ID，报告保留原始映射。

HASTEN仍用r01固定OpenAI CLIP ViT-B/32 FP32冻结编码器、原生验证预处理及未单位化512维特征。旧CLS＋patch/Spatial使用augreg21k→1k ViT-B/16、Resize224和.5归一化、等类/等患者权重解析ridge。此次是同数据划分及增量协议的路线性能比较，不能归因为层级单一机制、等骨干或等计算公平比较。缓存训练没有随机增强，不称官方逐位复现。

主参照只读复用已封存结果：CLS＋patch(df_matched_joint)六阶段BA为44、37.7778、33.8095、33.3333、27.7778、27.3333%，平均34.0053%，最终82/300；Spatial signed-p3为42、36.1111、31.9048、32.5、28.8889、28%，平均33.2341%，最终84/300。不额外调参或训练参照。

## 固定模型、树和预算

沿用r01官方提交063f295d08991d4da5b1f46a61bfcd279a94dbc6及同一最小兼容＋投影修正补丁（单样本、回放池、SVD V方向、实际AdamW增量右投影、FP64双曲几何）。无新的模块。主H-full与诊断H-no-proj各一条六阶段轨迹，seed74002；后者只关闭投影，原因是r01投影负收益需要跨数据检查，而非选出一个更漂亮主结果。主结果仍固定H-full，不事后改名。

10epochs、AdamW lr.001、wd.001、batch64、cosine至0、sample_num0；曲率1、hierarchy.5、entail.1、reg.1、margin.2、kcone.1、stat.5、投影能量阈值.95均不搜索。每阶段仅训练当前类，旧类只用官方均值/协方差高斯回放。

树为预先固定的解剖部位分组森林：肺实质（atelectasis/consolidation/edema/emphysema/fibrosis/infiltration/mass/nodule/pneumonia）、胸膜（effusion/pleural thickening/pneumothorax）、心血管（cardiomegaly/tortuous aorta/aortic calcification）、肺外（hernia/pneumoperitoneum/pneumomediastinum/subcutaneous emphysema）、一般胸片观察（no finding）。此为人工指定语义分组，不是临床确证的唯一层级，不构造严重度或因果包含链。每阶段只打开已到达叶类和所属父节点。叶类和父节点均固定单模板a chest X-ray showing {}.，不搜索措辞。

## 隔离与验收

每阶段按类到达编码训练图；复用同编码器缓存给第二臂。所有12个阶段模型全部拟合封存后才编码300val，禁止训练期验证/早停/选参。空验证loader的训练日志accuracy=0是禁用哨兵，不是真实准确率。恢复一致性只用当阶段32个训练特征，检查预测完全相同；正式记录finite、旧adapter/父节点不变、逐类回放次数、投影残差。任一失败属于工程失败，保留原时钟和全部日志，不吞掉继续。

输出官方固定融合主读出、纯双曲和原型分支，各阶段/平均/最终BA、逐类计数、旧新类、尾类以及checkpoint恢复回执。验证集已多轮开发，单seed结果只是探索性数值比较，不做独立确认或统计显著性宣传。阶段平均与最终BA同时优于两参照才记为数值双指标领先（无需人为门槛宣称成功）；两指标冲突如实报告。对投影按配对轨迹差异解释，不修改本轮配置。结果后再预注册与HASTEN直接相关的后继。

## 资源与交付

my-gpu独立r02_nih目录；原图、权重、上游代码与环境只读共享，新增预计<2GB，硬限8,000,000,000字节预留256MiB。首次实际执行至退出8小时硬截止，修复/重试共享原始时钟；CPU72000秒、RSS12GiB、GPU reserved<8GiB。独立controller每2秒检查进程/RSS/时钟，每10秒检查新增存储；截止前30秒TERM、5秒后KILL，只处理本进程组。挂载已确认NFS、可用383TB，写读探针后启动。沿用唯一hasten每30分钟心跳，更新为NIH工作；不触碰旧对话的独立监测。完成后经代理公开自有源代码、协议、聚合和中文比较报告；不公开图像、身份、逐样本张量、模型或原始日志。
