# CT7-D R1 技术恢复

原进程0.166秒、exit1，在selfcheck加载numpy.testing的Python库路径时被过宽路径保护误拦。尚未加载任何sealed预测，无模型forward、拟合或样本统计产物。原output/FAILURE、log、launch和receipt重命名保留R0，原PROTOCOL_LOCK不覆盖。

修复仅把test/reserved路径禁止作用于数据资产读取；白名单仍限制所有npz/npy/pt/pth/图像/csv文件，Python库不当作数据集。禁写白名单输出外保持不变，运行设置PYTHONDONTWRITEBYTECODE避免导入写缓存。所有输入hash、比较、阶段、seed及指标保持原锁。PROTOCOL_LOCK_R1绑定修复源码；在新output执行，原成本保留，不重训或新增预测。
