# 拟合实际生成分布的矩：前瞻CPU协议

切向保均值轮27c8e80整体失败，ISIC8批最大误差全部较同轮isotropic控制更坏。单位/均值恒等不保证未来损失。本轮检验一个不同机制：旧训练拟合中心Q，解码却加入条件噪声；拟合目标应与实际生成分布一致。该解释仅是假说，不已证前轮因果。

用Gaussian xi的正交投影归一化，得到中心n的均匀切向方向u_noise，z=n+sqrt(1-r²)u_noise。r=||n||>1e-12，r²<=1+1e-12，d>=2；零中心明确排除，正残差退化投影拒绝，不另造方向。E[z]=n，E[zzT]=nnT+(1-r²)/(d-1)*(I-nnT/r²)。公式来自Gaussian投影的旋转对称性，不能套用原Rademacher。unit中心原样。数学解析矩验证用正负正交切向cubature仅积分一二阶，不将其当非线性loss精确积分。

主候选generated_moment用解析生成矩拟合真实current synthetic均值和二阶矩，目标两项原系数1不增加超参数。原ball moment48控制完全保留；posthoc_gaussian使用旧目标拟合中心再同Gaussian解码，以隔离拟合目标；另random、normalized_moment，共5方法。两拟合臂相同原随机初始化、16中心/权重、48 Adam lr.02/投影/权重clamp、seed263/264/265。候选和baseline各48，整个case实际96记忆更新全计，不能叫case48或算子计算等价。无扩节点、查询或正式训练预算。

4独立CPU：解析矩/cubature/PSD trace/中心与权重联合梯度FD/采样query梯度FD/排列/unit中心/零与越域拒绝/可复现；small32d512K4、ISIC规模1536d9434K8、HK规模1536d1038K23，原全合成非高斯单位支持和公开类比例，非医学fit。只当前支持构造，未来部署只合成生成器参数；不回访旧医学支持。

查询48=4随机+2五方法共同stress各8，rank8/beta8/Fro2/vnorm2/stress12 Adam .01不改；查询只评价不更新memory。冻结后8新16点批，各自判门：两高维每批max<=.9各原moment/random且mean不增，加算术PASS才完整有限信号。共同stress变更不跨轮最大误差直接配对；不挑类/批/门。8批平均仅诊断。每个拟合臂的initial/final目标、解析mean/Q误差、trace、有效节点、逐批误差/方差/geometry全部正负公开；目标优化好仍可能损失差，不能混同。

每类存储16(d+1)float64；每方法pair cache128bytes/类另计，每次16点。两个拟合成本、解码、查询、内存workspace均纳入实测wall/user/system/RSS。case1800/总体5500秒仅CPU停止界；无工程重试或改阈值。0GPU/医学images/features/checkpoints/val/test。没有完整序列/分类训练/生产预算，不GPU-ready。完整结果先交付后分析，不能以矩变好或有限查询通过称医学贡献。
