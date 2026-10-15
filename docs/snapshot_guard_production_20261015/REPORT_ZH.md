# 守卫生产 CPU 首轮：工程失败完整交付

科学源 `0c707f92424810d407587b39771a22d8e2dc991c`。八个独立子进程均在解析测试源码时失败，0项生产CPU通过。不是守卫算法或医学效果失败，不能称GPU-ready。完整失败状态和逐次费用保留CPU_RESULTS.json；私有stderr/traceback只留服务器。

原因：服务器Python限制静态嵌套块数量，测试包装多层with patch超限。本机新版Python的py_compile通过未覆盖目标解释器限制。修复拟使用ExitStack；不修改已退休source，不同轮静默重试，不改科学算法/参数/数据。

八次实测驻留合计2.190434727818秒，user 0.524286000/system 0.213773000秒，peak 13.308593750MiB；父wall 2.334103900939秒另列，不相加。全部在import模型与数值代码前失败，0医学/公共预训练模型加载、0图像、0val/test、0optimizer/GPU。pipeline未执行，无实测client驻留，保持缺失。

生产入口与主四格/PMT贡献协议、预算账已经前瞻冻结，但关闭精确、真实ViT反传、最大阶段checkpoint/eval等检查全部missing，不填0/不冒充通过。当前不启动GPU；未来修复新source新round完整重做CPU，保留本轮所有失败/费用。
