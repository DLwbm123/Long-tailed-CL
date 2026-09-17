# CT1 运行状态

状态：**FORMAL_RUNNING，尚未完成实验**。

- 启动时间：2026-09-17T21:38:29.771206+08:00。训练源提交：`8e63785896d17126114a436027d1cb0890b4658f`。
- 独立分支：`exp/ct1-no-base-continuous-adaptation`，未合并 main。
- hb01 / RTX 4090 D，单 GPU worker。P0 的连续更新、输运、严格恢复、pointwise 一致性及配对初始化均通过；完整报告的合成数据检查通过。
- 一次启动检查已确认 HK / seed 1993 / U / Task 1 完成第 1 个 epoch，后台进程及日志正常。该检查不等于完整实验完成。
- 固定矩阵：12 条神经训练轨迹、90 个任务 checkpoint、900 task-epochs、32,340 个 optimizer steps；最终 270 行指标、2,754 行逐类结果，其中主方法 135 / 1,377 行。
- 按实际吞吐预计约 9 小时，GPU-process residence 总上限 12 小时；硬超时已扣除 P0 消耗并留停止余量。分析硬超时 21,000 秒，低于总 CPU 预算 6 小时。
- 本轮新增完整任务状态由现有传输 worker 归档至 my-gpu；只有通过 SHA、可读性和严格源恢复校验的新临时副本才回收。历史资产未删除或迁移。
- 完成全矩阵后自动解封 val、执行固定 2,000 次 component 级配对 bootstrap、生成最终报告，然后 STOP。没有新建定时监测，不启动额外实验。
- 新增 test/reserved 预测和访问为 0；禁止资产访问 guard 已启用。

## 私有运行位置

- hb01 根目录：`/root/rivermind-data/LongTailedCL/ct1/q8m20`。
- 训练日志：`output/formal.log`；分析日志：`output/analysis.log`。
- 正式状态：`output/public/FORMAL_STARTED.json`，完成凭证 `GPU_PHASE_COMPLETE.json` 和报告阶段 `FINAL_REPORT_ZH.md` / `NEXT_DECISION.json`。
- my-gpu 归档：`/tmp/p20archive`（链接到本轮专用 remote-home 归档目录）。

本提交仅发布代码、协议和验收/启动证据，不发布图像、逐样本预测、权重、checkpoint 或认证信息。尚未产生可作为最终科学结论的 CT1 完整结果。
