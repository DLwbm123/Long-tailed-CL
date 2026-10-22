# NIH 固定参考空间／患者类别等权解析原型：真实GPU通过并启动

RUNNING_QUALIFIED_BASE_ENCODING。科学源`1b211b20989e26d03d8a3ab83340ba30c2d45f5a`，完整机制/公平对照/13门/预算见[PLAN](PLAN.md)，当前源码启动公开不是科学完成，不承诺正向。

三臂LINEAR_BALANCED强完整视觉线性解析控制、KERNEL_NATURAL同核坐标自然患者质量控制、主KERNEL_BALANCED。共享真实当前数据和固定基类landmark权限、患者内影像权重、类Gram/均值统计；主等类别质量。固定基类参考坐标与精确全seen凸风险联合解，非旧局部字典refit/非线性旧矩运输/合成特征替换。2048真实独立基类患者landmark权限大于前有限memory，不能混称同旧A/B；线性1537/核2049活跃维数不同，不假称等容量/等FLOPs。参考带宽只由base landmark距离下中位数定、center只由其核行均值定，ridge1e-4冻结无val选择；旧统计不可变，当前全feature独立核对后释放。

## 实际全链检查

20合法基类不同患者真实图reader→GPUFP16完整encoder→GPUFP64正式1537/2049维映射/患者矩/全seen解析求解→8+2到达→三臂临时head各64行前反向/clip/SGD实际参数变化后丢弃→保存加载→独立CPU NumPy全矩系统/score PASS。qualification为测试正式维数重复landmark、全部丢弃，不在正式landmark中用重复患者；模拟base8参考不包含后来2类。6次解析资格solve，3丢弃SGD；正式没有SGD，不能把其算4500神经更新或把0optimizer说成0计算。

score误差max9.7998e-14，正常方程残差max1.249e-15，实际丢弃optimizer delta线性3.50021e-06/核1.14112e-05。资格显式编码20，shared linear/kernel映射及矩各20行（核两臂共享不要重复加）；GPU优化额外每臂64行/独立score每臂64行，6GPU资格解析解/3NumPy资格矩解、基类正式大小参考距离/center、20线上NumPy矩复算另计。编码2.921521s、编码后10.022853s，共原钟非ETA。

## 原钟预算与状态

first2026-10-23T02:21:51.772415+08:00—deadline2026-10-23T10:21:51.772415+08:00，8h不重置不延长；准备/资格/等待/故障/训练/验收共钟，更早本地准备未知不记0。3GB/2.7GB预停，RSS16GiB/GPUreserved8GiB，实际NFS余量写读/GPU8GiB/host16GiB准入PASS，5秒全组guard截止60s前TERM/30s必要KILL。guard统计本新根/q674与已退asymmetric/q672、structured/q673，窗口累计<30GB/远端27GB预停，本地文献源码保留100MB，基线不重置/旧证据不删除。

实测2026-10-23T02:23:36.779103+08:00：science928/68058、formal solve0、val0，无科学结果；根+entry采样152129096B、窗口远端518342642B、RSS采样峰2547187712B/GPUreserved1277165568B。guardian1277632/ticks469592432、worker1277634/ticks469592496完整中性argv/GPU PID现场匹配健康。run0自动verify，不能手动重复验收。

18模型val前seal/300val编码一次，final NumPy独立固定参考/完整类矩求解/18score<=2e-6同预测、方程残差<1e-9、真实probe和患者/metadata/counts/resources。到达统计当前全表线上独立NumPy复算后释放，不声称重新读取历史图像。必须全组退+13门与全指标/全成本/中文报告及代理push/SHA/匿名访问后才科学交付，无需人工审核。所有landmark/类矩/head/逐图score/身份/图像/原日志私有。
