# TissueMNIST-LT IF100 固定配对协议

用户选择TissueMNIST并采用原static_pc与固定local_transport一对比较。新增数据仅官方TissueMNIST224；保留原ViT/adapter/全局统计/解析竞争头，RL暂停，医学/Path/Organ/Tissue test均封存，不追加NIH或CIFAR训练。本轮只有2新真实训练轨迹、524 adapter更新、0策略更新，不根据val挑候选/强度/seed/类或阶段。

Tissue为肾皮质细胞灰度8分类，原32×32×7切片最大投影成32×32后放大224，不增加原生细节。原数据来自3个参考组织标本，本实验只有官方样本索引，不重建患者/标本隔离，不声称临床泛化。官方train165466/val23640；只train无放回下采样，val保留原分布与全部分母，不作为meta。test数组不打开。官方224与原缓存128的train/val标签精确核对，不改数据版本或逐类截断供给。

IF100、nmax5000；所有原train类供给都≥5000，采样可行。frequency seed74003得到rank=[4,2,7,5,6,3,0,1]，按rank取floor(5000*100**(-rank/7))，数量5000/2589/1341/694/359/186/96/50，总10315。各类seed74002+c*2003随机排列前缀、无放回；train/split74002。order seed74004得到[7,3,4,0,6,2,1,5]，固定任务2+2+2+2。fit/meta沿原80/20，meta任务末进入原拟合；每臂更新40/128/74/20，总262，两臂524。冻结头[4,2,7]、中[5,6]、尾[3,0,1]，3/2/3类macro组。

两臂相同预训练起点、样本、划分、顺序及2epoch/batch64/workers4/lr.0003/AdamW WD.01/clip5/FD10/ridge.001/proximal.01和原调度。候选四区域CLS条件pool、去图像区域均值并归一化，每类4中心各1/2入原P；旧中心类均衡区域共同位移，global common_shift。推理仍原全局解析头，无新增局部CE、读出、学习head或旧逐样本回放。模块激活和loss下降不等于分类收益。

2真实首batch backward、0optimizer预检全部PASS才TRAIN；两条TRAINED且262/0policy才统一官方val评价四阶段。新配对不要求医学historical_metrics；原ISIC/HK历史guard仍强制。检查loss/clip梯度有限、解析残差≤1e−8、所有阶段配对类分母与官方val一致、8epoch诊断/arm及局部先验4*K列/prior_delta>0。不虚构历史/逐样本等价或读出改变预测数。

仅GPU0/1（每卡准入≥32768MiB），2/3禁止，不干预其他任务。所有大文件NAS，真实nfs挂载/空间/小写读探针通过后启动。中性主子进程命令，私有路径经环境/stdin传递。单轮24h包含下载/CPU准备/等待/预检/训练/评价，不重置。8准确HTTP Range、不自动重试失败分段；使用实际收齐字节数，不以预分配大小声称下载完成。实体PNG在独立root，加载器路径保护不放宽。

正常2preflight+2train+2evaluate=6成功GPU驻留记录，同job/operation=train的每次费用都保留，不去重；失败另外全计。CPU准备及CPU检查、协调器墙钟分别列，不加GPU。历史161646.8880625614秒单列，各round成本一次累加而不求和cumulative。首次本地CPU检查客户端脚本构造出现SyntaxError，在任何远端检查/GPU执行前修正；原脚本保留私有回执，客户端执行耗时未单独测量，不能伪造CPU/GPU费用。

相对同配置原base门BA≥+1pp/tail≥−.5/forget≤+1/old,new≥−1。全六指标及增量、Acc、四stage、8类recall/n、3/2/3频率组全部正负公开。新类仍降但≥−1只通过保护门，不叫无损；单seed/order重复val开发证据，不代替ISIC/HK双目标或官方test benchmark。

本轮完成/失败先GitHub显式代理提交推送源/协议/完整匿名结果/报告/审计/成本，核对账号/分支/远端SHA与精确commit匿名HTTP，再NAS public_delivery及独立DELIVERY/HANDOFF。不外传影像/权重/逐样本分数特征/样本索引/统计中心原型矩阵/私有配置或原始日志。失败保留输出、停止依赖，不原轮自动正式重跑/放宽检查；修复另立清楚尝试。未结束不加后续科学轮。
