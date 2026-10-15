# 保中心均值的切向分布记忆：前瞻CPU证伪

相反噪声轮ec547c5整体未过：small8/8、ISIC4/8、HK8/8。八批预测方差下降约68/75/88%，仍有ISIC误差集中在门限附近。此为有限MC证据，不已证无限采样偏差；归一化会改变节点期望均值的算术反例仍成立。本轮只改变生成分布，既定配对、16节点、48次矩Adam、lr.02及原seed全部不变，不调噪声强度。

对球内中心n，u=normalize(xi-(xi·n)n/||n||²)，z=n+sqrt(1-||n||²)u。n=0时u=normalize(xi)，||n||=1时原样返回。对正残差退化投影拒绝，不能另造方向或静默重抽；数值域和容差见锁。因u与n正交且u(-xi)=-u(xi)，合法域内||z||=1且E[z]=n。均值仅保留原中心，不保证原中心等于真实类均值；Rademacher切向方向不是旋转不变的均匀球面分布，也不恢复真实高阶或方向矩。

主候选tangent_paired，控制random/原moment48/normalized_moment/原isotropic_paired。共享一次原48步中心权重构造和冻结中心贪心配对，原16点PRNG配xi/-xi，候选无需rho；原各向同性控制保留其rho。四个独立CPU：枚举均值/单位norm/符号对称/梯度FD/边缘期望/零与单位中心/非法域拒绝/可复现；32d512K4、1536d9434K8、1536d1038K23。全合成单位支持和公开比例，不冒充医学fit。

仍48查询=4随机+2五方法共同stress各8，rank8/beta8/Fro2/vnorm2，每stress12 Adam .01，只评价不更新记忆。共同stress变更不跨轮直接比较。冻结查询后8新16点批次，每个批次两高维都须max<=.9各原moment/random且mean不增，加算术PASS才完整有限信号。绝不平均替代主门。逐批相对各向同性与归一化控制的全部正负、逐类均值误差/trace/方差/expected center mean MSE完整保留。

候选存储16(d+1)float64/类，不再增加rho，但使用衍生128byte/类pair cache；临时噪声、投影和解码工作区由实际RSS及逻辑批bytes报告，不能称免费。每case1800/总5500秒仅CPU资源门。所有失败费用保留、健康源不改、不重试。0GPU/医学image/features/checkpoint/val/test。未检查完整序列、分类训练或生产预算；过门也不能直接称医学贡献。

完整结果先GitHub/NAS交付再分析，失败不调门/增容量追门；未来医学GPU须完整生产协议和真零预检。
