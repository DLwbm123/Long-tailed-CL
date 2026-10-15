# 完整序列结果与机制边界

整体NO_COMPLETE_SEQUENCE_SIGNAL。ISIC3stage均8/8；HK stage3/4均0/8，其他4stage8/8；small8/8、1/8、2/8。所有12stage接口精确不等于近似误差过门。

## small stage1 /seen2
主候选8/8；moment mean/max 0.12589310747632912/1.05678259301038；candidate mean 0.08338939750365078–0.10936080299182949 / max 0.3870998266997263–0.7981081027397239。
old mean None–None；new mean 0.08338939750365078–0.10936080299182949。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差0.0。单方法memory8448bytes；三份控制共享谱的测试checkpoint28748bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 0.5687162913382053, 'spectral_seconds': 0.004709251690655947, 'baseline_updates': 48, 'candidate_updates': 0}，eval0.3967624241486192秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好8/零0/坏0。
对moment old_mean_abs_error_delta:好0/零0/坏0。
对moment new_mean_abs_error_delta:好8/零0/坏0。
对gaussian mean_abs_error_delta:好4/零0/坏4。
对gaussian max_abs_error_delta:好3/零0/坏5。
对gaussian old_mean_abs_error_delta:好0/零0/坏0。
对gaussian new_mean_abs_error_delta:好4/零0/坏4。
## small stage2 /seen3
主候选1/8；moment mean/max 0.0912311572247311/0.7639636480703258；candidate mean 0.07939870932140976–0.11019138826126255 / max 0.4794701639181467–0.99521503316502。
old mean 0.08449272481623031–0.1060355299135805；new mean 0.06921067833176862–0.12777636965987685。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差1.1102230246251565e-16。单方法memory12672bytes；三份控制共享谱的测试checkpoint41356bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 0.11024770187214017, 'spectral_seconds': 0.0015726001001894474, 'baseline_updates': 48, 'candidate_updates': 0}，eval0.4728491702117026秒，已包含驻留。
对moment mean_abs_error_delta:好2/零0/坏6。
对moment max_abs_error_delta:好5/零0/坏3。
对moment old_mean_abs_error_delta:好1/零0/坏7。
对moment new_mean_abs_error_delta:好3/零0/坏5。
对gaussian mean_abs_error_delta:好2/零0/坏6。
对gaussian max_abs_error_delta:好2/零0/坏6。
对gaussian old_mean_abs_error_delta:好3/零0/坏5。
对gaussian new_mean_abs_error_delta:好2/零0/坏6。
## small stage3 /seen4
主候选2/8；moment mean/max 0.10108582351631763/0.7789537900492478；candidate mean 0.086073613967509–0.11104314193576542 / max 0.5257918396751888–0.9153962054372147。
old mean 0.07842796393949848–0.11170456624405757；new mean 0.08007927411617294–0.13718729159547807。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差1.1102230246251565e-16。单方法memory16896bytes；三份控制共享谱的测试checkpoint54028bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 0.11277722520753741, 'spectral_seconds': 0.0015658647753298283, 'baseline_updates': 48, 'candidate_updates': 0}，eval0.5337946498766541秒，已包含驻留。
对moment mean_abs_error_delta:好3/零0/坏5。
对moment max_abs_error_delta:好5/零0/坏3。
对moment old_mean_abs_error_delta:好2/零0/坏6。
对moment new_mean_abs_error_delta:好8/零0/坏0。
对gaussian mean_abs_error_delta:好3/零0/坏5。
对gaussian max_abs_error_delta:好3/零0/坏5。
对gaussian old_mean_abs_error_delta:好4/零0/坏4。
对gaussian new_mean_abs_error_delta:好4/零0/坏4。
## ISIC_scale stage1 /seen4
主候选8/8；moment mean/max 0.5544065620656697/4.166670835531525；candidate mean 0.13101406447656497–0.1852780671831061 / max 1.072537822899588–1.9808472024934685。
old mean None–None；new mean 0.13101406447656497–0.1852780671831061。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差0.0。单方法memory786944bytes；三份控制共享谱的测试checkpoint2364296bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 21.181374623905867, 'spectral_seconds': 5.530853232834488, 'baseline_updates': 48, 'candidate_updates': 0}，eval6.652396377176046秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好8/零0/坏0。
对moment old_mean_abs_error_delta:好0/零0/坏0。
对moment new_mean_abs_error_delta:好8/零0/坏0。
对gaussian mean_abs_error_delta:好3/零0/坏5。
对gaussian max_abs_error_delta:好5/零0/坏3。
对gaussian old_mean_abs_error_delta:好0/零0/坏0。
对gaussian new_mean_abs_error_delta:好3/零0/坏5。
## ISIC_scale stage2 /seen6
主候选8/8；moment mean/max 0.42622542874096414/3.8019906772297527；candidate mean 0.1129792625702501–0.17703744406282895 / max 1.1521490776851708–1.6167080808219176。
old mean 0.09968446870503382–0.1535209660775146；new mean 0.13574592882971995–0.25634892266874687。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差6.245004513516506e-17。单方法memory1180416bytes；三份控制共享谱的测试checkpoint3544776bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 7.799864192958921, 'spectral_seconds': 2.7760799499228597, 'baseline_updates': 48, 'candidate_updates': 0}，eval13.823641593102366秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好8/零0/坏0。
对moment old_mean_abs_error_delta:好8/零0/坏0。
对moment new_mean_abs_error_delta:好8/零0/坏0。
对gaussian mean_abs_error_delta:好7/零0/坏1。
对gaussian max_abs_error_delta:好6/零0/坏2。
对gaussian old_mean_abs_error_delta:好3/零0/坏5。
对gaussian new_mean_abs_error_delta:好5/零0/坏3。
## ISIC_scale stage3 /seen8
主候选8/8；moment mean/max 0.4099048089081172/3.3483374153105334；candidate mean 0.07728475611818048–0.15003983807037205 / max 0.8015797549649086–1.4180455123210844。
old mean 0.0699559368947553–0.17295730557335964；new mean 0.04408301743841898–0.10911304772027479。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差9.71445146547012e-17。单方法memory1573888bytes；三份控制共享谱的测试checkpoint4725192bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 8.308862212114036, 'spectral_seconds': 3.6989768939092755, 'baseline_updates': 48, 'candidate_updates': 0}，eval37.49823981523514秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好8/零0/坏0。
对moment old_mean_abs_error_delta:好8/零0/坏0。
对moment new_mean_abs_error_delta:好8/零0/坏0。
对gaussian mean_abs_error_delta:好4/零0/坏4。
对gaussian max_abs_error_delta:好3/零0/坏5。
对gaussian old_mean_abs_error_delta:好5/零0/坏3。
对gaussian new_mean_abs_error_delta:好3/零0/坏5。
## HK_scale stage1 /seen13
主候选8/8；moment mean/max 0.32233244509788067/3.1915095386676775；candidate mean 0.12068279018369679–0.16125187609091357 / max 1.6788905207577944–2.1275741335499743。
old mean None–None；new mean 0.12068279018369679–0.16125187609091357。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差0.0。单方法memory2557568bytes；三份控制共享谱的测试checkpoint7676144bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 52.790370646398515, 'spectral_seconds': 12.68128107022494, 'baseline_updates': 48, 'candidate_updates': 0}，eval6.477951135952026秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好8/零0/坏0。
对moment old_mean_abs_error_delta:好0/零0/坏0。
对moment new_mean_abs_error_delta:好8/零0/坏0。
对gaussian mean_abs_error_delta:好8/零0/坏0。
对gaussian max_abs_error_delta:好5/零0/坏3。
对gaussian old_mean_abs_error_delta:好0/零0/坏0。
对gaussian new_mean_abs_error_delta:好8/零0/坏0。
## HK_scale stage2 /seen15
主候选8/8；moment mean/max 0.29268473698629527/3.144305894788356；candidate mean 0.12495381874635582–0.1547140844970509 / max 1.5882352111367704–2.038636293097051。
old mean 0.11713401818679693–0.15392066733560394；new mean 0.08095489554005743–0.17578252238348857。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差1.1102230246251565e-16。单方法memory2951040bytes；三份控制共享谱的测试checkpoint8856624bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 4.424288225360215, 'spectral_seconds': 1.8613325650803745, 'baseline_updates': 48, 'candidate_updates': 0}，eval7.515117694623768秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好8/零0/坏0。
对moment old_mean_abs_error_delta:好8/零0/坏0。
对moment new_mean_abs_error_delta:好3/零0/坏5。
对gaussian mean_abs_error_delta:好8/零0/坏0。
对gaussian max_abs_error_delta:好7/零0/坏1。
对gaussian old_mean_abs_error_delta:好8/零0/坏0。
对gaussian new_mean_abs_error_delta:好3/零0/坏5。
## HK_scale stage3 /seen17
主候选0/8；moment mean/max 0.289335711449918/3.2328027063138447；candidate mean 0.14488288663843352–0.17611297616304317 / max 3.2564277246524167–3.712401212558981。
old mean 0.10231187095897916–0.1503175382080253；new mean 0.36957876082567725–0.47412820713826864。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差1.1102230246251565e-16。单方法memory3344512bytes；三份控制共享谱的测试checkpoint10037040bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 4.611264296807349, 'spectral_seconds': 1.9978355262428522, 'baseline_updates': 48, 'candidate_updates': 0}，eval8.189773292280734秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好0/零0/坏8。
对moment old_mean_abs_error_delta:好8/零0/坏0。
对moment new_mean_abs_error_delta:好2/零0/坏6。
对gaussian mean_abs_error_delta:好8/零0/坏0。
对gaussian max_abs_error_delta:好5/零0/坏3。
对gaussian old_mean_abs_error_delta:好8/零0/坏0。
对gaussian new_mean_abs_error_delta:好5/零0/坏3。
## HK_scale stage4 /seen19
主候选0/8；moment mean/max 0.2550316875017892/2.4147366264582537；candidate mean 0.13547077440487032–0.16567537851533987 / max 2.589066412525754–3.2818091445217474。
old mean 0.14214215604911462–0.17251741840007953；new mean 0.0787640304287937–0.146544282549581。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差9.71445146547012e-17。单方法memory3737984bytes；三份控制共享谱的测试checkpoint11217392bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 6.951708382926881, 'spectral_seconds': 2.097534691914916, 'baseline_updates': 48, 'candidate_updates': 0}，eval9.361577346920967秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好0/零0/坏8。
对moment old_mean_abs_error_delta:好8/零0/坏0。
对moment new_mean_abs_error_delta:好8/零0/坏0。
对gaussian mean_abs_error_delta:好8/零0/坏0。
对gaussian max_abs_error_delta:好5/零0/坏3。
对gaussian old_mean_abs_error_delta:好8/零0/坏0。
对gaussian new_mean_abs_error_delta:好3/零0/坏5。
## HK_scale stage5 /seen21
主候选8/8；moment mean/max 0.2497417539087551/3.8230730008289253；candidate mean 0.10748401432032997–0.1433514744525321 / max 1.7172758155222958–2.216445812118221。
old mean 0.103825287867804–0.1339880359735903；new mean 0.061339692439620584–0.3257611207436985。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差1.0408340855860843e-16。单方法memory4131456bytes；三份控制共享谱的测试checkpoint12397808bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 5.274622330907732, 'spectral_seconds': 2.071388268843293, 'baseline_updates': 48, 'candidate_updates': 0}，eval10.239848072174937秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好8/零0/坏0。
对moment old_mean_abs_error_delta:好8/零0/坏0。
对moment new_mean_abs_error_delta:好8/零0/坏0。
对gaussian mean_abs_error_delta:好8/零0/坏0。
对gaussian max_abs_error_delta:好5/零0/坏3。
对gaussian old_mean_abs_error_delta:好8/零0/坏0。
对gaussian new_mean_abs_error_delta:好8/零0/坏0。
## HK_scale stage6 /seen23
主候选8/8；moment mean/max 0.2782229785005824/3.94927702136318；candidate mean 0.11087945821209173–0.13765840081532912 / max 1.723467900838207–2.4720650477044623。
old mean 0.11232072169466789–0.14116362947439937；new mean 0.09574619164504207–0.16086468130255427。
旧memory/draw/checkpoint精确；固定query旧loss差0.0，ETF误差9.71445146547012e-17。单方法memory4524928bytes；三份控制共享谱的测试checkpoint13578288bytes，不当单臂部署/ViT大小。
构造{'baseline_seconds': 7.504208150319755, 'spectral_seconds': 2.0646255100145936, 'baseline_updates': 48, 'candidate_updates': 0}，eval11.569399731699377秒，已包含驻留。
对moment mean_abs_error_delta:好8/零0/坏0。
对moment max_abs_error_delta:好8/零0/坏0。
对moment old_mean_abs_error_delta:好8/零0/坏0。
对moment new_mean_abs_error_delta:好8/零0/坏0。
对gaussian mean_abs_error_delta:好8/零0/坏0。
对gaussian max_abs_error_delta:好6/零0/坏2。
对gaussian old_mean_abs_error_delta:好8/零0/坏0。
对gaussian new_mean_abs_error_delta:好3/零0/坏5。

## 限制与下一步
HK stage3/4的candidate最坏类均匿名15，合成count5。Gaussian同期也0/8；旧memory张量、生成点、固定query均精确不变，不能归因存储被覆盖或旧支持重压缩。stress随stage改变，不能将后续误差变化当旧memory漂移。第3stage新类mean部分更坏，虽然宏观mean降低，不能掩盖新类近似误差。
表示只有均值/低秩协方差和尾方差。rawcov精确仍不足以保证非线性查询；本轮不能确认所有误差来自高阶信息。下一步做同一二阶矩不同支持分布的确定性反例审计，明确moment-only表示边界，不加节点/更新/采样预算追门。反例也不能证明该HK fixture的因果或医学失败。
没有classifier/encoder训练、损失性重压缩、医学数据/val/test/GPU。本轮只append，memory随seen增长，非固定全局预算。低样本统计可能含原支持信息，不能自动授权医学原始feature回放。
本轮4CPU310.0024135960266秒，14轮55次累计4652.2778973747045秒；父wall310.0570597620681另列，内部操作不相加，client/collector未测null，新GPU0，旧GPU400036.74851310626不变。0工程失败/超时。
