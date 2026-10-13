# 最终 checkpoint 同状态模块开关诊断（冻结协议）

用户授权四个最终 checkpoint：ISIC/HK × 模块单独/与 local_transport 组合。训练源固定 2fdfaa546309b7590a17d039a137c34a7047e807，父轮交付 b7325e5f2e197fa1674ee57f37b764f811986abb。诊断源码的精确提交另存启动回执；不修改历史训练源。

## 固定干预

OFF 仅令新增二次项 S=0，固定 encoder、bank、local memory、R、保存的竞争权重、RHS、M 和原解析求解容差。ON 使用保存的 float32 head；重建原 float64 解须相对误差≤1e−8，保存的float32必须与double cast精确一致。禁止重算竞争权重来替代原保存值。读最终ISIC stage3、HK stage6；不选择checkpoint、不恢复训练。

ON 必须逐类精确复现已公开最终 recall、固定分母及Acc，否则该诊断失败，不将漂移的复算当新科学结论。所有差异统一ON−OFF。OFF是本checkpoint上的局部反事实，不是历史base/local_transport；不能据此解释完整训练路径或宣称去掉模块重训会怎样。

## 数据和计算预算

仅直接打开既有val.csv和对应295/1698张val影像；禁止调用同时读取train.csv的通用manifests入口。原order/seed74002、eval bicubic Resize224→Normalize(.5)、batch64、workers0、shuffleFalse。每个checkpoint一次encoder前向，共3986影像曝光；两种解析头共用同一份内存特征，共7972头读出曝光。0 optimizer/0 policy/0 train/test/reserved读取；不下载、不复制影像，不保存新模型。train频数仅复用已公开数字。

只GPU0/1，启动前每卡空闲≥32768MiB；GPU2/3禁止、其他进程受保护。两个slots按ISIC单独/组合、HK单独/组合固定顺序执行；4个GPU子进程驻留逐次计费，失败也保留。实际launch开始2小时上限含推理/求解，失败停止后续依赖，不自动retry。CPU合成检查驻留独立列，pipeline含在驻留内不重复相加；旧训练与评价不重复收费。

## 预注册输出

实际作用：S W_OFF M/RHS、ON/OFF力范数、总/旧/新头变化、原生目标和新增项能量、求解残差、保存头重建误差、原float32读出分数变化（原始及去公共分量）、真实类margin和旧新组margin变化的全体/旧/新/逐类聚合。直接新类力为0不等于新类头或分类不变。

每状态ON/OFF完整BA/Acc/tail/old/new/macroF1；62个状态×类别配对计数（31类×两种来源），n、正确数、预测变化、纠正、破坏、错到另一错，全部正零负。旧新混淆：旧→新、旧→旧错、新→旧、新→新错，和旧/新正确数，计数必须守恒。每类分母精确匹配冻结val支持。

不利用结果挑强度、投影、参数、刷新频率、seed或再跑方案。只读最终stage，不制造平均增量/遗忘反事实。原始去重前风险和拟合映射未保存在checkpoint，本轮不重建；不读旧类训练图，因此不能验证旧类真实漂移。

## 验收与交付

CPU先验证小矩阵直接解/PCG、S=0退化、坏checkpoint拒绝、双向翻转计数、混淆守恒、非法scope/分母拒绝和完整四状态报告。真实诊断逐次验证metadata、参数及buffer不变、ON公开结果一致、finite/solve≤1e−8。只公开匿名汇总，不发布影像、模型、头、bank、矩阵、逐样本分数/ID/特征、私有路径或raw日志。

结束交付源码、协议、全部结果/正负变化、报告、审计和实际费用；GitHub显式代理验证精确SHA与报告匿名HTTP200，再独立NAS public_delivery/DELIVERY。缺失不填0，失败保留具体去标识原因和全部费用。之后不自动改模块或追加实验，test继续封存。
