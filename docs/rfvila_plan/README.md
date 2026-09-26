# NB2-RFVILA-12H-R1 — Codex 执行包

版本日期：2026-09-26。状态：**PROPOSAL_NOT_EXECUTED**。
本包是新实验的计划、执行指令与小规模数学参考，不是已完成的医学结果，也不是可直接投入真实数据的完整执行器。

**唯一主问题：在正常两类 Task1 后冻结的 APART 与固定 VLM 上，随机非线性特征能否增强类别均衡解析持续学习？**

必须同时保留通用 OpenCLIP（G）和 BiomedCLIP（B）。不能用医学提示词冒充医学 VLM。
本轮不执行 ConCM、ACTM、FD 搜索、LoRA、额外 Task1 或大基础阶段训练。

## 阅读次序
1. `01_EXPERIMENT_PLAN_ZH.md`：研究问题、方法定义、预算、统计解释。
2. `02_CODEX_PROMPT_ZH.md`：可直接交给 Codex 的完整操作指令。
3. `03_PROTOCOL_PROPOSAL.json`：待路径发现和验收后实例化的协议草案。
4. `04_METHODS.csv` / `04_EXPECTED_STAGE_MATRIX.csv`：固定方法及预期覆盖表。
5. `05_ACCEPTANCE_CHECKLIST.md`：工程、数据和交付验收。
6. `06_SOURCES_AND_ATTRIBUTION.md`：论文、公开代码和本项目来源边界。
7. `reference/`：无外部数据的 CPU 数学参考与 32 个单元测试。

优先级：本轮用户最新明确指令 > 02 与 01 的本轮定义 > 本轮 JSON。
旧 R1/MED12/V-ConCM 文件仅是历史材料。旧协议的“忽略所有时间限制”或多 GPU 设置**不适用于本轮**。
如 01/02/JSON 存在实质冲突，先修正并重新锁协议，不凭实现方便自行选一种。

## 时间与计算资源
从第一个本任务远端命令开始，最多 43,200 秒墙钟；包括准备、下载、排障、重启和报告。
只使用 1 个明确获分配的 GPU。不得继承历史的 GPU 0/2/3 多卡执行。
新方法集合在 T+2:00 之前按资源门槛确定；T+8:30 后不得启动新的拟合工作单元，T+10:30 后不得启动图像前向；T+11:00 结束模型/解析/评分计算，最后一小时只做保存、CPU 报告和归档。
提前完成就结束，不空转到 12 小时，不自动追加第二轮。

## 怎样使用
把本包交给有 my-gpu SSH 权限的 Codex，粘贴 `02_CODEX_PROMPT_ZH.md`。
Codex 需在既有 remote-home 工作区新增实际 runner。不要直接运行作者原始训练脚本。
本包的 `reference` 只验证数学定义；没有 SSH 逻辑、模型加载器、数据访问隔离或长时间运行服务。

本地参考检查命令：
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s reference -v
```
实际结果保存在 `LOCAL_SYNTHETIC_TESTS.txt`。这些测试不证明真实 GPU 适配、数据无泄露或方法有效。
