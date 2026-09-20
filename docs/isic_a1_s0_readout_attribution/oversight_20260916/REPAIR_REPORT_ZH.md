# A1 数值工程修复

原始 fba6a39 的 BLOCKED 报告不改写。本次按用户12小时自主工程维护授权继续，未新增训练或test访问。

相同输入的首个Transformer block算子取证发现，原始与双分支Linear的batch与单图输出有约3.6e-6至9.1e-6的局部差异；patch embedding、LayerNorm、GELU、适配attention的QK/PV两次bmm及softmax在本探针中逐位一致。差异经深层累计，触发raw特征门槛。不能据此推断表征损坏。

已安装PyTorch提交7269437的[Linear实现](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/Linear.cpp)会将连续三维输入展平后执行addmm，矩阵行数随batch改变。[官方数值说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)不保证批处理与逐片计算逐位相同。两项有限工程控制保留：关闭cuBLASLt无改善；以baddbmm保持每图矩阵维度也未通过native单图门槛，均未采用、未拟合S分类器。

采用的修复只改变隔离推理副本中三维B>1的Linear算子调度：逐图调用原生FP32 Linear，再沿batch维拼回。网络输入仍batch48，正常逐样本top-1、参数/buffers、dtype、所有数学公式及1e-5门槛保持；不是把全网络改成batch1，未增加网络模块。B1仍直接调用原生算子，reference未改。

三个父模型、B32/B48、反序、替换同行及B1抽查的全部固定工程布局中，两路routing ID一致，raw main/few最大绝对差均为0，原始backbone与A缓存通过；参数/buffers和RNG不变。GPU门槛通过不等于完整实验已完成。接下来仍须完整运行入口重新核验及吞吐/累计预算准入，然后在新目录实际执行固定矩阵。

先前A拟合18项及其私有分数原位复用，不重拟合；旧失败证据、权重、缓存、checkpoint和原CSV修改均保留。所有诊断资源累计到同一个A1预算，不因重启清零。诊断退出码0只表示相应工程检查完成；前两个控制的gate仍失败。见各子目录RESULT/ACCESS/RESOURCE及RESOURCE_LEDGER。
