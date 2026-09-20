# GSR-VILA 两路线实现包

本分支实现附件 `GSR_VILA_two_routes_plan_prompts.zip` 中的可复用数学与协议层：

- A / RASP：冻结 APART `a` 与冻结 CLIP `u` 的双特征、完整跨块矩、文本语义先验、谱正则 ridge，以及 A0--A8 全部读出。
- B / ACTM：冻结 CLIP 锚点漂移、可微条件仿射拟合、齐次矩传输、共享协方差旧类 margin loss，以及 B00/B01/B10/B11 配置校验。

入口脚本是 validator-only dry-run；不会隐式下载权重、启动训练、读取 test/reserved 或猜测类别名。它只返回 `CONFIG_VALID`，不把配置检查冒充资产或工程通过。真实执行前必须补齐配置中的权重、预处理、tokenizer SHA256，并通过项目自己的数据和 checkpoint 锁；本分支不包含医学训练驱动。

## 校验

```bash
python -m pytest tests/test_gsr_vila.py
python -m pytest tests/test_gsr_vila_review.py
python route_a/run_ct13.py --config configs/gsr_vila/CONFIG_A_RASP.json --dry-run
python route_b/run_ct14.py --config configs/gsr_vila/CONFIG_B_ACTM.json --dry-run
```

当前交付是代码与协议实现，不是 ISIC/HK 结果。附件本身不含图像、私有特征、权重或 checkpoint，因此没有把“缺失 ISIC”伪装成实验结论。

审阅修复后的合成验收为 32 tests passed；状态仍是 implementation-only，未进入 `ASSET_VERIFIED`、`ENGINEERING_PASS` 或 `READY_TO_RUN`。
