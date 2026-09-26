# 验收表

## 来源与设定
- [ ] 新run来自b01ba12；旧分支/run只读；唯一新run_id。
- [ ] PAPER_CODE_PROJECT_DIFF.md 列出原文、代码、本轮适配差异。
- [ ] ISIC2+2+2+2；HK2×10+3；原3个order；6个正常Task1父状态。
- [ ] G与B均完成原生配套加载，B不是Generic CLIP加医学prompt。
- [ ] 新neural epochs=0/optimizer steps=0；test/reserved=0。
- [ ] 不继承旧的ignore-time或多GPU override。

## 数学
- [ ] m4096、两RP seed独立于parent；Ra/Ru实际文件hash。
- [ ] 分支与RF尺度、bias、ReLU、RF后无归一化与01完全相符。
- [ ] S=sum class second moments；Q=class mean；W=(S+KλI)^(-1)Q。
- [ ] 不等类样本数、最后3类、列顺序、新列补0、重复样本不改变等权目标的测试通过。
- [ ] 无伪逆/jitter静默替代；残差<=1e-8；primal/dual/CV一致。
- [ ] RF统计由当前真实特征产生，不由旧均值/二阶矩伪造。
- [ ] Macro-F1反例通过；zero-recall比较集合；空old/tail=null。

## CV与文本
- [ ] 只Task1组件CV，同预算9λ；所有新映射独立选择，之后冻结。
- [ ] 分组不足时全parent统一fallback，不能以val补足。
- [ ] 相同W应用CSE1和025；不挑alpha；原class-ID排序ties。
- [ ] 无未来类文本；现有两个医学模板不更换；text每模型配套。

## 数据与资产
- [ ] 当期train临时缓存只在当期消费，独立流访问分账。
- [ ] fit无法打开val/test/未来train；实际反向测试通过。
- [ ] 关键副本目标端rehash与1个父/1个bank恢复通过。
- [ ] 大图像副本核验范围单列，没有伪称全量逐字节完成。
- [ ] stage原子落盘、备份回执、恢复无重复累计。
- [ ] 最终保存各stage W、R与lambda锁、末bank、score/hash，并保护私有信息。

## 时间与完成
- [ ] 第一个remote命令就固定T0，重启不改；<=43200秒、单GPU。
- [ ] 按T+2h资源锁tier，估计时间有1.5安全系数，报告留1h。
- [ ] 8:30/10:30/11:00/12:00各截止遵守。
- [ ] 核心one-RP含LOCK 855/8721行；two-RP1170/11934；全扩展1440/14688。
- [ ] 未完成不填0、不当方法FAIL；只完整有效比较可裁决效用。
- [ ] RF1为主、RF2如实单列，不挑种子、不平均logits当免费ensemble。
- [ ] 2000组件bootstrap固定parent；多重开发与极小分母限制清楚。
- [ ] FINAL报告及read/forward/CV/solve/资源/备份/失败/缺失清单齐全。
- [ ] 默认commit-only；push仅有本轮明确授权才做；无私有数据泄露。
- [ ] NEXT_DECISION=STOP，停止本run监测，无第二轮自动任务。
