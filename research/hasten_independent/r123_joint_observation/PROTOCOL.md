# R123 冻结联合观察读出诊断

## 运行前冻结

本协议及SCIENTIFIC_FREEZE在实现/native/科学测量之前保存。原始首次准备时钟见freeze，独立4小时，失败、重试、所有子任务共钟不重置。它是服务于保留历史的RL主线构建的必要仪器诊断，非RL训练、方法成功、长期价值或创新证明。当前已知MedVR与LFPC有全图/局部联合观察先例，复用不是新颖性。R122原十门失败及所有旧结果不变。

## 数据、仪器、输入

只读R122封存的全部175唯一TRAIN组件与原两组框、非法回退，原A88/B87和类别定义不变。不重提议、不筛框、不看VAL/TEST、不换seed、prompt或ROI。识别器仍为按时间选定的R121共同preRL SEED_SFT，已拟合全部TRAIN；A/B非crossfit或患者独立。所有权重冻结，无优化器。

5个读出：NATIVE_ROOT单全图；DUPLICATE=[全图,全图]；FIXED_CENTER=[全图,原中心.25-.75]；GLOBAL_BOX=[全图,R122GLOBAL提议框]；DETAIL_BOX=[全图,R122CENTER_DETAIL提议框]。所有图224平方/64visualtokens，四联合读出同128视觉tokens且逐组件相同实际输入token数，原单图category-only问题文本不变，greedy max24。全部走原生HF接口，不声称手动多图cache parity。NATIVE_ROOT更便宜，仅作原图损害控制；四个联合读出顺序按selectionrank旋转，避免同一臂总先执行。crop floor/ceil保持R122原定义，原非法框继续全图回退；必须验证相应联合读出与DUPLICATE完全一致。

## 费用、native与不可变

每science组件5响应/9视觉编码行，合875/1575。native使用原TRAIN0,1及其R122已存native框，各跑完整5读出矩阵（10响应/18编码），单图hidden及同形末tokenhead原生/手动核对另4编码；正常合885响应/1597TRAIN编码/0新proposal/0VAL/0optimizer。所有重编码/无效回退/失败/native费用计入；实际输入/生成tokens/wall逐读出保存。旧R122提议与R121SFT成本单列为历史依赖，不能称总体0成本。

native必须验证图像grid、真实64/128视觉token数、四联合输入长度一致、单图native/manual hidden与同形head差<=.05、原task/reference不可变、原框/拆分一致及新全部矩阵时间1.5倍+600秒余量可在原时钟内完成。raw全序列headBF16差另报不据此更改阈值。无新模型/feature/cache下载或大磁盘副本。每2秒独立guard包含所有旧根与本机200MB余量、新root/cache/tmp，9GB预停、真实/data探针/free>=1GB、CPU4/RSS48GB/allocator56GiB/filecap256MiB/core0。仅pro5000物理GPU1，中性argv。预计新增<20MB，累计严格<10,000,000,000B。

## 十三项冻结诊断门

DETAIL_BOX对DUPLICATE：macro>=+1pp/tail>=+2/head>=−1（3门）；对NATIVE_ROOT同三门，防止改变参考隐藏联合接口损害（3门）；对FIXED_CENTER和GLOBAL_BOX分别macro>0/tail>0（4门）；A、B各nonTiny18对DUPLICATE>0（2门）；四联合实际视觉行和输入tokens相等，且max24上限相同（1门）。micro、tiny、实际生成tokens/time及所有逐类分母完整报告，不追加事后主门。

## 决策与停止

全部175完成后统一保存，不按中间结果修改科学参数；退出0后CPU-only saved-only核对175/88-87/345classrows/13门/原框及回退/费用/模型不可变，不重新生成。失败保留全部成本/源码/原钟并按原协议根因修复，禁止调prompt/ROI/seed或读VAL救分。负结果关闭该冻结仪器的联合证据救援，不继续接口/顺序/阈值变体。通过仅表示联合证据有用，仍不证明非myopic/RL必要性；后续真正序列RL需另冻完整状态、奖励、同反馈同预算fixed/myopic/监督或精确控制，不能把一次单步读出改名RL。真continual另需generic无未来类/各臂独立连续状态/有界旧证据。

只公开源码/协议/安全聚合，不下载或公开组件索引、逐例预测/框、图像、特征、权重、私有路径或凭据。原生运行和后处理source SHA分列；source-only GitHub全程显式7897代理/no-force、remoteSHA/匿名200验证。
