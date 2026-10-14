# PMT 生产 CPU 验收

当前为生产 CPU PASS，尚无 GPU/医学图像或checkpoint/val/test/正式optimizer更新。数学与链路通过不是净分类收益。

关闭模块时，Core/LT 的完整参数、head、bank n/μ/Q/component 和 regional memory 相对原训练入口精确一致。新增链路通过27个合成阶段 checkpoint 重建；140次合成 optimizer 更新及6个合成零更新预检不计正式预算。

1536×23旧矩/当前统计、原proximal/Woodbury/competition求解通过。三个配置虚拟矩误差≤2.78e−17，最大全部链路solve残差8.54199919187e-10；显式满秩1536输入、21旧类、23最终类的CPU shadow也通过（残差6.37e−10以内）。四区域独立映射和global/region纯平移退化核对通过。原目标的envelope有限差分误差1.315e−11。当前新类不再运输；错位行、重复frame、非法scope/budget/早val和坏报告分母拒绝。

全阶段旧radius unknown+范围可经完整controller/competition/训练/checkpoint链路；不伪造精确协方差。配对纠正/破坏计数守恒及全部主方法/机制对照报告通过。最终report/collector可由无torch的bundled Python导入。

最终PASS pipeline 81.983961554 秒，含于operation 87.961869413及实测CPU驻留89.346059675秒；不相加。各独立CPU尝试实测驻留合计254.519952494秒，含保留的失败/先前PASS，全部尝试见CPU_ATTEMPT_AUDIT。首次wrapper语法失败在远端源执行前，驻留未测不伪造；另有APART打包依赖遗漏和合成预算断言不匹配，均已修复、0GPU/0formal，工程摘要见ENGINEERING_AUDIT。

后续六个真实GPU预检各需真实firstbatch backward且0optimizer/参数buffer不变/双池梯度有限非零，另做显式synthetic-map满秩最大旧bank峰值检查；全六PASS才能TRAIN。test继续封存，只有GPU0/1。旧类推广是假设，当前域拟合不等于旧风险概率，也无新类无损保证。
