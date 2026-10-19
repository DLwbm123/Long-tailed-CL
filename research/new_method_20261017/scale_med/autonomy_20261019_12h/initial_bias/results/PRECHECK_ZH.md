# 初始bias适配预检

固定训练源955f2bd；cd97f8b补齐正式batch32提取→CPU暂存→batch8取样的缓存路径；45a90ef只强化全窗空间守卫（显式含本地/Git预留与默认cache）和相应边界自检，train.py数学/优化规则与cd97f8b相同。实际环境：训练原中性解释器，未来fit现代中性解释器，不复制/修改环境。

实际head-only一步后g特征exact不变，正式缓存loss/head梯度与fresh误差0；bias实际一步finite、canonical absolute overrides往返一致，按q均匀minibatch的目标缩放检查通过。预检模型和8行训练更新全部丢弃，不作为正式初始化。

合成三臂六对比的成功/失败/缺失、JSON、增量统计、实际训练override现代映射通过；系统残差5.55e-17。两次同包装器故意注入RuntimeError均生成FAILURE、EXIT、资源及协调回执，退出1是有意测试，不是科学失败。全部预检费用保留，重叠墙钟不能相加。

正式比较只有原预训练初始状态、固定10epoch/batch8/AdamW .003、原共同G选alpha规则；不使用预检学后head/bias。完整G/P/C测试尚未执行，不声称性能收益。
