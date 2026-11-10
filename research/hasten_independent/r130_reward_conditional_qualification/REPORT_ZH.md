# R130：旧类条件KL未通过机制资格，关闭该固定配方

五臂各32次实际非零更新及保存完整性通过，预先冻结的6个性能条件仅1个通过；合并完整性为2/7条件满足，整体资格失败。它是单次任务转换资格，不能解释成原完整九门通过/失败次数，也不能宣布所有RL无效。

| arm | macro % | head % | tail % | tiny % | old5 % | new5 % | micro % |
|---|---:|---:|---:|---:|---:|---:|---:|
| PREFIX_CURRENT_MENU | 33.6674 | 51.3504 | 7.1429 | 0.0000 | 45.9148 | 21.4200 | 52.5934 |
| RAPO | 31.6836 | 48.0441 | 7.1429 | 0.0000 | 40.1522 | 23.2150 | 49.1701 |
| ACC_CTAN | 29.4422 | 46.6894 | 3.5714 | 0.0000 | 29.9345 | 28.9499 | 47.0954 |
| ACC_FULL_KL | 34.3520 | 52.4915 | 7.1429 | 0.0000 | 47.2752 | 21.4288 | 53.7344 |
| ACC_OLD_COND | 30.6710 | 46.3565 | 7.1429 | 0.0000 | 34.4055 | 26.9365 | 46.9917 |
| CE_OLD_COND | 30.8768 | 49.0804 | 3.5714 | 0.0000 | 30.3280 | 31.4256 | 49.2739 |

各读出同964组件，旧624/新340；60条完整class正确整数/分母见CLASS_RESULTS.csv。组内query不是独立患者。

## 冻结条件全部保留

| 性能条件 | 实际差值或关系 | 通过 |
|---|---|---|
| D−C macro≥1pp | -3.6810pp | 否 |
| D−C tail≥2pp | +0.0000pp | 否 |
| D oldmacro≥C | -12.8697pp | 否 |
| D newmacro≥B−1pp | -2.0134pp | 否 |
| D macro>匹配CE | -0.2058pp | 否 |
| D tail>匹配CE | +3.5714pp | 是 |
全5臂32非零步、own Adamstep32/sigma、同当前tape、无旧/未来TRAIN实际访问、固定prefix、全五臂seal后VAL与CPU重算通过；完整性通过不替代效能条件。

## 实际机制证据与边界

去掉保留PG的B相对A：oldmacro -10.2177pp、newmacro +5.7349pp。当前条件下保留与获得有取舍，不能因单探针acc/ret梯度余弦−.984864就称retention全有害；A/B也不单独识别全部CTAN效应。初native恒定retention下A/B浮点梯度已有微差，不声称唯因果隔离或逐位同一。
完整候选KL的C有较高旧类但新类较弱；D解除old/new质量约束后相对C新macro +5.5077pp，却损旧macro -12.8697pp，未实现更好的联合保留/学习。D不胜匹配CE宏均值，不支持RL独有增量。不能挑C最高macro或D优于B的部分指标换成主方法成功。
RAPO实际256query中91组全错，其中63组保留值不恒定；246个错误响应有正的实际PG标量优势，27个正确响应有负总优势。B/C/D保留项不进PG，错误正总优势仅诊断；E无PG。它们是标量系数/组计数，不是参数梯度份额、错误强化的已证实因果或VAL效果。每类/配对完整贡献保留，不据此改类/重采/调权。
全部tiny读出0；相同自然256query中hemorroids(3 TRAIN)未被抽到，ileum(6 TRAIN)仅1query且八答全错。两个tiny各1个VAL组件；这些结果不说明充分训练后临床无效，也不据此给本配方追加预算/强制覆盖/换seed救分。

## 诊断与完整核验

原固定8query梯度探针、4×624菜单2×2及其CPU审计此前已完成，不重复模型调用。当前完整核验独立解码10240科学响应（10237 EOS完成），mask/EOS/padding/文本一致；private原始token/prediction均未fetch/public。有限候选KL不是自由生成全概率。终点相对prefix实际L2均非零；中间每步L2是实际仪器值，未伪造逐类梯度。
finalize_saved完整childCPU2.960440秒/wall4.060934；另独立Fraction整数算术从60类行重算全部metrics/门与60配对行；全部saved coefficient分析和token/endpoint审计分别实际执行，0新模型/编码/生成/optimizer。梯度norm/cos没有重新前向重算，不混称独立梯度复验。

## 成本、失败与原钟

原准备1794310608.4689498→科学退出1794318935.124686，原含准备8326.655737秒，退出0，原23hdeadline1794393408.4689498不改；审计/交付仍计同钟。成功worker wholeCPU2690.518582秒，body2687.480181只是子集，不重复相加。首次工程断言失败wholeCPU73.672867秒、160生成/5native+1probe optimizer保留；所有native更新和两次probe反事实更新丢弃还原。首次probe未保存逐token，不能事后补造；成功probe/科学rollout均完整保存私有。
实际全部attempt18744生成=10240科学+128probe+2496menu+5784VAL+96native；167optimizer=160科学+5native+2probe；4TRAIN native编码/0VAL新编码；0新SFT，历史306/4868/624另记非免费。
生成prefill 3096528 inputcells，额外native语言parity588，active生成89291，padded decodecells128712，cached decoder forwardrows109968；TF每phase/kind行、inputcells和active tokens见ALL_ATTEMPT_COSTS/TF_COSTS，不假装完整FLOPs或每算子GPU秒。
CPU软件5.224675秒；既有probe/menu审计2.090272/2.011244秒、当前fullfinalizer/reward/token/local整数审计均全额保留，body不重复相加。预执行静态检查修了finalizer类名映射被prompt列表覆盖的局部变量，首次实际fullfinalizer执行即通过；scientific worker未改。准备shell/SSH/Git CPU未全面仪器，未知非0。

## 后继决定

关闭固定qO旧条件KL配方，不进入完整5task，不延长32步或扫描系数/温度/seed/菜单。不是整个研究暂停；当前没有活跃本任务GPU。详见ROUTE_DECISION_ZH.md。R125/R128原门与负证据保持，不以本短实验覆盖。
