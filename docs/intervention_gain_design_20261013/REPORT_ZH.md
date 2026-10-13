# 可干预净类均衡收益：设计与CPU验收报告

已完成独立数学helper、合成CPU检查与贡献协议冻结。没有生产入口、GPU实验、医学checkpoint/影像/val读取或性能结果，test/reserved继续封存。主控制固定local_transport。

## 设计结果

以每类等权的“OFF错→ON对”减“OFF对→ON错”为目标；联合建模原/候选全部类别赢家，不把重叠pairwise混淆质量直接当可修复收益。原曲率求解只提供旧contrast候选方向，原bank/Q/R/pairs和新列不变。五个固定训练统计候选按救回下界−破坏上界准入，全部非正返回identity，随后使用原old-max守卫。

借用只构造投影相似形状的方差先验与压力情景，不补观测支持量、不移动类别均值、不改原解析系统。目标稀有类仍权重1/K；n影响不确定范围而非宏目标重要性。支持量必须对应原zero-action均匀权重和组件质量=count/n。供体相似核没有经医学可靠性校准。

精确守卫范围：OFF赢家为新类则保留；OFF赢家为旧类则等于RAW赢家。可救回原旧赢家的旧类错误及一部分新类错误；不能修复原新赢家的旧类错误或新新混淆。旧contrast零和还使ON旧最大值不低于原旧平均分，新类分数低于该平均分的错误不在可修复集合。保护新类recall/几何margin不等于保护所有旧类或新类precision。

local_transport继续负责R和共同统计运输，新模块只在该控制自己的头和曲率下处理剩余的可干预收益，无第二次运输。原曲率能量radius固定1，不沿用旧0.1弱限制；最终是否非零由净收益模型门决定，不能靠扩大能量伪造作用。

## CPU证据及明确的负结果

- 两次合成CPU均PASS，GPU0、医学影像/checkpoint/val读取0、正式optimizer更新0。
- 15个原联合高斯转移独立区间oracle，另90个声明矩误差集合中的压力案例；救回/破坏界覆盖对应精确概率。加权组件支持量、负投影方差和不完整类别模型被拒绝。
- 独立正例ray获得精确macro救回1/3、破坏0，选定非零修正；这是人工构造统计世界，不是医学效果，也不是完整原解析系统非零成功。
- 反例只看救回会接受：救回.133333、破坏.333333、净−.200000；主净门正确返回identity。
- base与local_transport两个完整native小维合成系统的四头均返回identity；不能声称已证明端到端非零收益。保守概率界可能拒绝真实有益修正，是尚未解决的风险。
- SPD原曲率方向与独立dense二次目标解一致，最大相对残差7.124067321787803e−10；stage-one identity、原bank/R/pairs不变、float32/64守卫新类margin、checkpoint往返及错误head/source拒绝PASS。
- 1536×23只检查neutral能量与接口，未做完整高维native协方差/风险求解，也没有真实GPU零更新预检。

## CPU费用

两次独立process residence：34.0435039261356秒、9.291289448738098秒，合计43.3347933748737秒。pipeline分别3.438965118955821和2.3056604340672493秒（合计5.74462555302307秒）已包含在驻留，不能重复相加。没有失败CPU进程；第二次为新增入口验收和压力oracle后的最终检查。GPU费用0。以前已交付诊断/控制不再计费。

## 冻结贡献范围与未完成工作

将来须另行授权后才接生产。主四格base/local_transport/base+GAIN/local_transport+GAIN；三个固定贡献头GAIN_NO_BORROW、GAIN_RESCUE_ONLY、NEUTRAL_MATCHED。18原stage×4头=72侧车、主表20行+AIR2参考；40组全部六指标/Acc/已有F1差异，1260控制及模块stage-class行，以及实际救回/破坏/错转错、新类间隔/守卫可修复范围、AIR差距、全部正零负和费用。GAIN_NO_BORROW改变提案和情景整个路径，不能称风险门单因素消融；neutral只匹配守卫前native能量。

生产元数据、真实float32原控制回归、完整高维缓存/运行时间、共享GPU评价、医学模型校准均未完成。这是设计冻结与CPU验收交付，不是新医学实验结果。矩范围不是医学95%或分布无关置信区间；统计漂移及患者独立性假设仍需独立证据。未来反复development val不能称独立确认。

实现见tools/intervention_gain.py与tools/check_intervention_gain.py；完整数学见DESIGN_ZH.md，固定科学资产/参数/贡献与准入见PROTOCOL_LOCK.json和PROTOCOL_ZH.md，实际CPU标量见CPU_COMPATIBILITY_CHECK.json和AUDIT.json。
