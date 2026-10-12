# DAT 与 Hao：ISIC/HK 非官方论文重实现冻结协议

用户已明确批准补完剩余 DAT/Hao 比较，允许 Hao 按原方法保存旧训练图像回放，并允许根据论文重实现、缺失参数前瞻冻结。AIR/完整APART及诊断均已交付，不重复训练、评价、收费，也不继续调参。本轮只有 DAT_paper/Hao_paper × ISIC/HK 四轨迹，seed/order/现有 train/val 均严格复用；仅GPU0/1，test/reserved封存，RL暂停。以 PROTOCOL_LOCK.json 为机器规格。

## 复现身份与证据

核对本机 Zotero 的论文全文、DAT官方WACV正文/补充材料、Hao论文和代理GitHub仓库搜索后，未找到可核验的两篇作者完整实现。搜索没有找到不等于证明不存在。DAT论文只承诺未来开放代码；两篇的缺失设置不能由第三方配置冒充作者实际日程。找到的 Hokimastah/CALB-DML 与 CALB-CBDM 是独立改编及新方法仓库，未用其代码，也不外传用户PDF。

- DAT：Gu 等，WACV2025，[官方论文](https://openaccess.thecvf.com/content/WACV2025/papers/Gu_Dynamic_Adapter_Tuning_for_Long-Tailed_Class-Incremental_Learning_WACV_2025_paper.pdf)、[官方补充](https://openaccess.thecvf.com/content/WACV2025/supplemental/Gu_Dynamic_Adapter_Tuning_WACV_2025_supplemental.pdf)。保留 hyper-adapter/cache、当前train的聚类复用判据、平衡/不平衡双支路、有效样本数加权蒸馏、key匹配和跨任务蒸馏。未替换为本项目解析头。
- Hao 等，ACM MM2024，[论文](https://arxiv.org/abs/2407.13768)。保留ImageNet预训练ResNet18、可学习余弦分类器、CIL-balanced CE、distribution margin、旧模型logit L1蒸馏、herding旧图像回放。没有加入本项目局部先验、运输或漂移模块。

公开名称始终带 `_paper`，只称非官方论文重实现/医学迁移，不称作者完整代码、原论文精确复现或等资源性能胜负。

## 事先补全的选择

DAT固定40epoch/task、Adam .03、batch24、constant lr/WD0/no clip、tau .5、beta .9999、T2、所有loss系数1。40epoch与tau .5取论文DomainNet方案作新域迁移选择；不是medical val选择。未披露的generator具体实现为共享down/up权重加可学习layer identity通道调制，bottleneck64，12层FFN平行残差，down Kaiming/up0/层调制1。保留同缓存IN21k→IN1k ViT以复用预训练控制；与DAT论文IN1k-only有差别。每cache保存所分配任务类的并集分类头，复用时复制旧行并扩展新行；balanced head用同并集。预测按每张图冻结CLS与task keys最近邻路由到对应cache，只输出其已分配类别，其他已见列−1e9，无batch多数投票或GT task。复用entry替换后，旧keys保留并指向替换entry。balanced teacher概率detach；所有类的有效样本权重按原train计数，不额外归一化。匹配旧entry时，用其冻结旧logits在当前图上的MSE保持旧输出，不回放旧图/逐样本特征。

平衡支路每批各当前新类数量严格相同，因此实际balanced batch为floor(24/Knew)*Knew：HK首task为13，其余24。每个不平衡batch对应一个平衡batch、合并loss做一个optimizer step；单task不平衡样本每epoch遍历一次。balanced支路当前图可重复抽样，单列曝光差异。当前特征只临时用于kmeans（自行实现kmeans++/Lloyd、10次restart/至多100次迭代/中心变化平方和≤1e-8、固定seed、Hungarian标签匹配），不持久化、不读取未来任务图或旧图。strict >tau复用，等号创建。

Hao正文没有给出完整epoch/lr/scheduler，固定100epoch/task、SGD .1/momentum.9/WD.0005、各task重置milestones60/80×.1/no clip；batch32延用医学皮肤设置。alpha .5、lambda_d .4、lambda_k .75、margin .4、余弦scale初始化10为事先完成的选择，不调val。CBC按当前+实际旧bank计数加log prior及old log alpha，等价Eq7；margin按Eq9作batch内old exemplars/new labels的逐项和，不把其改成均值；Gaussian按全局原train频率缩放，再用cosine；KD按Eq11 old raw logits绝对差逐类和、逐样本平均。firsttask没有old margin/KD。所有ResNet参数可学；预训练使用torchvision ResNet18 IMAGENET1K_V1，明显不同于ViT控制。

herding只存每类min(20,原train n)张不同旧训练图的私有引用，不复制图像、不重复伪造独立样本。HK原类5/6/19/21分别只有3/6/6/18张；CBC使用实际计数，不虚构每类20。新bank在task边界从该任务图上用当前model确定性eval变换选出，旧bank原封保留。最终bank ISIC160/HK413张。公开只给支持聚合，不发布sample IDs。

## 数据、预算与屏障

现有ISIC18718 train/295 val，4+2+2任务；HK6821 train/1698 val，13+2+2+2+2+2任务。仅复用原order和label map、同medical变换（train Resize256/RandomCrop224/flip .5/Normalize .5，eval Resize224/Normalize .5），不新增划分、seed、参数或预算臂。Hao输入归一化也沿用共享medical设置，属于迁移选择，不等于论文原预处理。

|数据集|方法|各task更新|总更新|
|---|---|---|---:|
|ISIC|DAT_paper|6680 / 4800 / 19760|31240|
|HK|DAT_paper|6560 / 200 / 80 / 1280 / 1160 / 2200|11480|
|ISIC|Hao_paper|12600 / 9300 / 37400|59300|
|HK|Hao_paper|12300 / 1100 / 1000 / 3400 / 3200 / 5300|26300|

四条合计128320次optimizer更新，显著大于原方法2epoch/fit-meta及APART10epoch，不能称等更新量。DAT额外balanced曝光和Hao旧图回放单列。固定队列 ISIC_DAT→HK_DAT→ISIC_Hao→HK_Hao，两卡slots先两DAT；空卡按队列承接Hao。24h从实际launch起，包含等待、预检、训练与评价，不重置；超时/失败完整保留，不自动重跑或删失败。

生产CPU先核对真实12层ViT/ResNet18合成反传、冻结core、公式、旧teacher冻结、路由、银行/计数、checkpoint、封存权限与完整八方法报告。CPU不打开医学图、不作效果声明。4个真实第一batch backward/0optimizer预检均PASS、参数未改变/有限非零梯度才正式训练；Hao BN buffer只在被丢弃shadow中更新，不能作正式统计。初始DAT down梯度可为0（up0），检查up/key/head非零有限与CPU下一步down可反传，不伪造初始全部梯度非零。

四轨迹精确TRAINED并核对更新/全部task checkpoint后统一development val；checkpoint method/version/dataset/order/seed/source/task/budget必须精确一致。训练全程零val读取，无val选epoch、key、seed或参数。真实launch前NAS实际nfs/free/小写probe及GPU01≥32768MiB，不借GPU2/3、不干预他人进程；中性父子命令与NVML核实。只做一次启动状态检查，无默认持续轮询。

## 交付与范围

仅匿名metrics/STATUS/每epoch标量、MODEL_COST参数/真实checkpoint字节/GPU峰值、4预检和PROGRAM.costs；图像/weights/逐样本分数/IDs/特征/key/bank索引/矩阵/privateconfig/rawlog不公开。正常4preflight+4train+4eval=12GPU费用逐次驻留一次，失败额外全部计；CPU、数据/权重准备与parallel wall另列。复用控制更新/费用0，不加各轮cumulative。

完整八方法×两数据集六指标、Acc/可获得macroF1、24个新方法相对全部六旧控制的六增量、全部stage/31类n/recall、所有正负及频数组/小分母。旧F1缺失不补0。单seed/order反复val仅开发证据，非独立患者/临床或封存test基准。失败保留缺失范围，missing不填0/单臂不当全比较。

实际结束后代理推送项目GitHub同branch的源码/协议/全部正负/报告/audit/cost，核对remote SHA与精确commit报告匿名HTTP200，再独立NAS public_delivery/DELIVERY/HANDOFF。AIR/APART等旧轮只复用本地公开结果，禁止旧模型再评价、旧轮重复交付或重复计费。
