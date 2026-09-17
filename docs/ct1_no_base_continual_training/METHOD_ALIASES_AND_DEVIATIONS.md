# CT1 方法身份与实现边界

CT1 是用户批准的新协议。CT-J-CB 是 S-J-CB 的持续适配变体，不称原样复现；CT-ConCM 继承 K 的经验方差与旧/新类数回放系数，不使用历史 0.05 系数。

- 所有任务 10 epoch，AdamW 每任务重置，cosine 每任务执行；权重持续继承，同一组 adapter pools，不调用任务槽位复制。
- 真实三项损失直接复用锁定来源 APART 的 `_init_train`，设置 all_seen；pool assignment、pull、reduction、/3 不重写。ConCM 只覆写采样与回放系数。
- 训练保持原生算子及 batchwise 路由；独立提取/评价副本统一采用已批准 A1 pointwise 路由及 FP32 Linear 调度。CT-ConCM 用该副本的 raw main+few logits，CT-ConCM-CB 用同一副本的 joint 特征，不选择 head 或添加校准。
- 每个 epoch 的输运都从不可变 pre-task anchor bank 出发；任务末只提交一次。旧图像不重新编码。固定仿射模型的统计更新精确，不等于真实旧图像当前表征的精确矩。
- Risk 只在 W0 列空间求凸读出，完全复用 U 神经训练；风险项不反传适配器。新风险统计不改 CBRidge G/R。
- PT-CB 是预训练冻结参照。完整当前/未来类别的元数据仅用于协议容量和准入；真实训练/统计读取受任务访问边界约束。P0 按要求重现既有 PT-CB train/val 缓存，不读取任何 test/reserved 资产。
- 使用 4 个 PyTorch/BLAS 线程与 4 个 loader worker。默认 12 小时 GPU-process residence、6 小时 CPU 读出/报告、3 GiB hb01 活跃空间、10 GiB 新归档上限。
- 归档进程仅搬运本轮已完成任务状态，校验传输 SHA 与全部非共享 tensor 可读性；训练端已通过共享预训练 core 的严格模型恢复后才入队。回收范围只有本轮 hb01 临时 task 副本。
- 新依赖隔离安装到本轮目录：CVXPY 1.7.3、Clarabel 0.11.1、SCS 3.2.9、OSQP 1.0.4；训练环境 torch/CUDA 未升级。唯一求解器为 Clarabel，其他包是 CVXPY 导入依赖，不作回退求解器。

不声明原创性、临床风险保证、无记忆或独立确认。无论正负结果，固定矩阵完成后 STOP。
