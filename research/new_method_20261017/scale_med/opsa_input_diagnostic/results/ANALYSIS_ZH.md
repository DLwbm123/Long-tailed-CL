# OPSA 输入与精度诊断：支持32图定位到 cuDNN TF32 设置

独立工程诊断已完成，性能实验未执行，method_success=false。原失败实验及原容差保留；未重跑 OPSA 训练。

固定同一32张 ISIC producer0支持图，缓存像素与当前PIL原图解码像素逐元素相同，maxabs=0。Pillow10.4.0、jpg ABI6.2、libjpeg-turbo3.0.3与认证环境一致。

| 输入 | cuDNN TF32 | 特征maxabs | 原F支持分数maxabs | 原回放门 |
|---|---|---|---|---|
| 缓存像素 | 关闭 | 0.00024188394763818843 | 0.0005049140215747094 | 未通过 |
| 重新解码 | 关闭 | 0.00024188394763818843 | 0.0005049140215747094 | 未通过 |
| 缓存像素 | 开启 | 8.326672684688674e-17 | 3.3306690738754696e-16 | 通过 |
| 重新解码 | 开启 | 8.326672684688674e-17 | 3.3306690738754696e-16 | 通过 |

matmul TF32始终关闭、batch32/no_grad/原冻结模型相同。此固定四格将本32图的回放差异定位到cuDNN TF32设置；未定位具体内部kernel，也不推广到全数据、不同batch或训练梯度。不是缓存损坏、OPSA性能负结果或已修复训练的证明。

共4次逻辑双流encoder wrapper调用、128图次，0backward/optimizer/fit/query图。四格有限值与FEATURE_SEAL先于比较，参数版本和冻结状态保持。原门feature1e-6/score1e-5不变。未序列化或公开像素、特征、逐例身份、模型或第三方材料。

源9b1fa66199e8616ab7b0606e04efb056af062c48先封存并验证远端/匿名协议访问。父1114179、子1114180正常退出并退休；中性活动argv与GPU名已捕获，600s/CPU1800s/RSS8GiB/新增128MiB及17/20GB窗口守卫未触发。无历史数据删除或模型/环境复制。

下一步须另封数值绑定与全部producer支持回放合格的训练协议，保持原F、线性/无MAPC等控制、全模型先封后query、少数类及全类风险门；本工程完成不自动复活失败pilot或修改旧结果。

成本一次计入（准备/阅读/网络/报告CPU未知）：
{
  "activity_wall_seconds": 38.59640121459961,
  "known_worker_wait4_CPU_seconds": 46.305065,
  "coordinator_CPU_seconds": 2.7342999999999997,
  "peak_RSS_bytes": 1609760768,
  "round_remote_bytes": 38764,
  "window_remote_bytes": 14169988810,
  "unmetered": "Preparation, source-reading, environment introspection, network and report CPU unknown; worker body not double-counted"
}

累计：
{
  "activity_wall_seconds": 32376.905967295956,
  "known_worker_wait4_CPU_seconds": 40744.99602699999,
  "coordinator_CPU_seconds": 1273.7930139999999
}
