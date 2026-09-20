# GSR-VILA 审阅修复交付

本文件对应审阅包 `RASP_ACTM_74017f1_review.zip`（SHA256：`2ff39ddec25208997627037a5783021040913756a1523e94096921d5e65cec81`）。本轮只做合成数据上的实现、协议和测试修复，没有读取医学图像、私有特征、checkpoint 或 test/reserved，也没有启动训练。

## F01–F12

| 项目 | 状态 | 修复 |
|---|---|---|
| F01 | PASS | 旧类 margin 使用完整 `qᵀCq`，保留非对角和跨块协方差。 |
| F02 | PASS | 使用 `tau*log(1+sum exp(...))`、`epsilon=1e-8`，单旧类仍与新类竞争。 |
| F03 | PASS | query 风险改为 K 维平方误差、类内平均、类别等权。 |
| F04 | PASS | `DualMomentBank.homogeneous()` 返回真实齐次 `bar_S/bar_M`；独立样本重算验证非零平移。 |
| F05 | PASS | component 只归约为每类 count/sum/second-sum；持久状态不保存 component 向量或逐 component 2048² 矩阵。 |
| F06 | PASS | reliability 接口聚合每类全部 component，count 不再被单 key 截断。 |
| F07 | PASS | 采用显式局部列序、类内协方差、类均衡 pooled covariance、`omega=10/(n+10)` 和最危险文本差向量。 |
| F08 | PASS | A/u 单分支统计恢复 `sqrt(2)` 均值和 `2` 二阶矩，A2 保持联合 h 坐标。 |
| F09 | PASS | A3/A3s 提供基于 A2 的 top-k 输入相关预测；A5 支持只去掉 reliability gate。 |
| F10 | PASS | 增加 Torch ACTM episode 链：student→drift→H→transport→memory loss，保留 adapter 梯度。 |
| F11 | PASS | NumPy/Torch 漂移支持 labels 或固定 weights，并按类等权；teacher 明确 detach、统计 float64。 |
| F12 | PASS | seeds 恢复为 `[1993,1994,1995]`；任务、epoch、变体、fold、lambda 和权重/预处理/tokenizer SHA 校验严格拒绝不完整资产。 |

## 验收

```text
python -m pytest -q tests/test_gsr_vila.py tests/test_gsr_vila_review.py
32 passed

python route_a/run_ct13.py --config configs/gsr_vila/CONFIG_A_RASP.json --dry-run
status=CONFIG_VALID, training_started=false, test_access=0

python route_b/run_ct14.py --config configs/gsr_vila/CONFIG_B_ACTM.json --dry-run
status=CONFIG_VALID, training_started=false, test_access=0, trajectories=12
```

审阅包的 26 项回归检查全部通过；原始 6 项实现测试也全部通过。当前状态仍是 `CONFIG_ONLY/implementation-only`，没有伪造 `ASSET_VERIFIED`、`ENGINEERING_PASS` 或 `READY_TO_RUN`。

## 发布范围

原 PR #1 的 base 不是 `main`，包含历史 92 个提交和 1200 多个文件。修复分支从 `origin/main`（`7a089e4b30a3efa6e2943aa4b99c1f5e60652b16`）重新建立，只包含两个 GSR-VILA 提交和本报告，供 Pro 进行干净审阅。旧 PR/分支不强推、不删除、不覆盖。

真实 ISIC/HK 执行仍需独立提供已锁定的 CLIP 权重、预处理/tokenizer 字节、数据划分和父状态；本轮不启动医学实验。
