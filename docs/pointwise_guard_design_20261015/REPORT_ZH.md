# 逐样本训练边界约束：CPU证伪结果

状态：CPU二维SPD设计算术PASS，但完整native/高维/生产NOT_READY，不能启动GPU。本检查不是新医学实验或val校准。固定设计源 `af3f4ee7bdc1c1c971dfa6bfa5aa3e2bb91fc17f` 已先经显式代理公开并核实精确DESIGN HTTP200，再运行中性CPU入口。

1. 旧矩raw affine pushforward完全精确、新类统计不变时，新类真实margin从0.481960892降为-0.063243107，正确变错误。说明“当前统计不运输”本身不保证边界保护。这里只是二维SPD ridge特例（旧fixture不要求原始单样本L2单位范数），不是实际Core/LT归一化统计的完整重演，不能直接解释全部医学下降。
2. 单训练点约束可恢复margin至0.481960892，非公共head变化2.555983614，说明并非所有fixture都必然identity。但双点fixture的非公共变化仅1.26e-15，退回判别identity；完整head可能仍变化，不能拿全头范数冒充有效作用。也不能由这个例子断言所有高维训练约束必然identity。
3. 约束活跃时，遗漏约束导数的naive envelope误差1.298397462；Lagrangian导数与重新求最优值的有限差分误差5.58512613935e-09。因此若接入原解析训练，不能只替换头后沿用无约束固定头loss梯度。Wref块内必须冻结，修改刷新时序需要重新验证导数。
4. 均值margin保护反例、关闭精确、类别排列/正交坐标与公共logit规范不变检查通过。

0GPU/0医学影像/0医学checkpoint/0val/test/0optimizer。CPU pipeline 0.001488582930秒，包含于operation（含导入）0.059657083126及实测驻留0.099170416128秒，不能相加。此前一次手动合成算式预览未单测CPU驻留，不伪造费用；没有医学或GPU调用。

下一步仍只CPU：先构造完整native competition/proximal的矩阵自由约束/dual/KKT与正确envelope，再检查1536×23代表性支持秩、活跃约束下的非公共可行方向、全部约束的分离与资源界。任何“保护新类”的说法仅限已观察训练支持；未识别旧域外推和旧旧风险仍存在。不能因本例单点成功直接开训练，也不能因双点identity就按val挑点/放宽守卫。未形成可检验的非identity机制证据前维持NOT_READY。

本轮PMT医学结果已先完整交付080c2a3；旧轮不再轮询/收集/重评/交付。无新GPU科学轮、LAUNCH或正式deadline。
