# CT5-F CURRENT_STATUS

COMPLETE_CT5F — 2026-09-19；NEXT_DECISION=STOP。

完整45/459行F1、135/1377行含对照，18复用/27新增阶段。推理退出0，报告首次退出1已通过独立源码锁修复，report_r1退出0。禁止重新启动p24driver或p24。

Final BA：HK F1 61.213、P61.128、C013.638；ISIC F1 58.276、P55.497、C046.401。主F1−P：HK+0.085pp，区间跨零；ISIC+2.778pp，[0.470,5.202]。无新增神经训练或test访问。详见FINAL_REPORT_ZH.md及results。

推理825.958秒，报告失败1.165秒、修复2.818秒全部保留；输出约22.39MB，未删除历史资产。六父/45W/预测锁、工程/修复证据、完整表格已同步。

分支analysis/ct5f-full-frozen-reference；原源码c466ff68625502fbb5fb9096c44df6785b2c7058，报告修复b023168。后续候选及严格前置条件见NEXT_DECISION.md；尚未启动新的训练。
