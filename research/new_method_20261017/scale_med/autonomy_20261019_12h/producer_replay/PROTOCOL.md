# 支持缓存生产者与运行时：固定工程复演

真实读取/复演前封存。此为排查norm_geometry方向不一致的工程比较，不是新的性能实验或旧范数门重试。旧结果714ccd2及门1e-4原样保留；不训练分类器、不选择参数、不评分验证/测试。

已静态核查原producer run_real_query_node.extract：原ApartFeatures、同权重、transform(False)、seed=原PROTOCOL_LOCK、batch32、num_workers0、FP32；无encoder updates。原prior_rows与当前development_images解析为同一根，prepare按cache.y/components绑定行。原record extract_fit亦明确未训练适配器。原运行时Python3.10/Torch2.1.2+cu121/torchvision0.16.2，当前Python3.12/Torch2.6.0+cu124/torchvision0.21.0；timm1.0.15和Pillow10.4相同，matmulTF32 false/cudnn true。此处版本是当前存在旧环境的读取，不认证其安装内容自历史以来未变化。

固定同norm_geometry的64条初始~valid训练图（每类原顺序前16），无筛选。ORIGINAL用现有旧运行时原APART重放并保留一个私有解码像素Tensor，然后同旧运行时bare ViT FP32提取最终CLS。MODERN用当前运行时bare ViT FP32读同一像素Tensor；消除第二次解码差异。三条路径每图一次，共192图次，不切换额外精度，不改变batch。私有输出仅用于定位差异，公开聚合，不发布像素/特征/身份。

原APART对历史cache归一化1536特征逐元素最大误差<=1e-5为历史重放工程接受线（沿用历史B_FEATURE_PARITY误差尺度）；失败为历史重放未决，不自动判数据损坏，不覆盖cache。相邻两bare或原APART/bare单位方向L2>1e-4时，只能标明路径/运行时差异存在，不直接归因特定CUDA kernel。若均低于线却旧norm_geometry失败，其原因仍未决，不得忽略或删掉旧结果。不以三条路径择一个去做性能宣传。

沿用完整包装器，三个阶段独立退出/资源回执。阶段失败即停止依赖阶段、不自动换解释器/算法，保留私有checkpoint。预计本轮新增<128MiB（64图FP32像素约38.5MB且NFS保守allocated计费），不复制环境/模型/cache；全窗<20GB/17GB软停，单文件512MiB上限。GPU0显存/挂载/容量/写读探针在启动前核查。CUDA/Torch等cache与tmp均本窗NFS，PYTHONDONTWRITEBYTECODE=1。无时间/GPU预算上限但如实记费，旧失败CPU缺失保留。完成后公开源码/协议/所有聚合/成本/实质分析和失败，核push/SHA/匿名访问。不因复演通过自动重跑旧测试或启动范数方法。
