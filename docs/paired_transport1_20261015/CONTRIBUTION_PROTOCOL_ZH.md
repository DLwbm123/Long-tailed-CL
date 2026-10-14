# PMT 的最小贡献对照

主方法始终为 local_transport；Core/base 是固定完整控制，不为显示增益而减弱。两数据集均报告：

|比较|解释|
|---|---|
|PMT − Core|全局旧矩运输相对原全局 translation 的净贡献|
|LT+PMT − LT|主组合相对各自控制的净贡献|
|LT+PMT global_only − LT|保留原区域 translation 时，全局矩运输贡献|
|LT+PMT − LT+PMT global_only|独立四区域映射的增量贡献|
|LT+PMT − PMT|组合相对单独|
|(LT+PMT−LT)−(PMT−Core)|主交互，不能用组合的绝对成绩代替互补贡献|
|global_only 的对应交互|区分原区域先验与新全局运输的叠加|
|每条新臂 − AIR|BA/avgBA/tail/forget/old/new及准确率；资源不同，不作等预算胜负|

两数据集、全部六指标及每类/每stage的正、零、负变化全部保留。救回和破坏按同一stage、同一 canonical val 的各自控制配对计数；`correct_on−correct_control = rescued−damaged`。全部匿名混淆计数、old/new角色和真实类对最大其他类的平均间隔公开；没有新类准确率保证，不将均值间隔当个体保护。

预先沿用开发门：相对各自控制 BA≥+1pp、tail≥−.5pp、forget≤+1pp、old/new≥−1pp；主组合与单独在两数据集都逐项报告，不能按val挑臂、挑类或重新定义主方法。full-regional 优于 global_only 才能支持区域映射的额外作用；拟合 rank/residual 非零仅是激活证据。失败与全部 identity/无分类变化都如实公开，不追加参数、seed或预算档位。
