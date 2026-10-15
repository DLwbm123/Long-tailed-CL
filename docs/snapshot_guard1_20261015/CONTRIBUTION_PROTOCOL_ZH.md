# 前瞻贡献与结果交付规则

固定主方法local_transport，不按val换主方法。主表六配置Core/LT/PMT/LT+PMT/守卫运输/LT+守卫运输 × ISIC/HK=12行；AIR仅外部已公开参考另列完整差距，不因预算/预训练不同宣称等预算胜负。

全部六指标(finalBA/avgBA/tail/forget/old/new)、final/avgAcc及已有F1、不补旧缺F1。六主要对比每dataset：守卫单独−Core、组合−LT、PMT−Core、LT+PMT−LT、守卫−PMT、组合−LT+PMT；再报告组合−单独与交互。所有正零负保留。六指标全向、逐stage/31类nrecall、旧新混淆/救回破坏/错误换错误/true margin和新类变化完整报告。

same-state OFF与proposal仅隔离该快照head开关，不能替代独立训练轨迹贡献。至少报告guard vs same-state OFF/proposal、独立Core/LT和PMT控制的纠正破坏及类别均衡净值；存储float32对照vs double精度变化须单列，不能当守卫收益。AIR差距正负均列。预声明发展门：各数据集组合对LT需finalBA>0且tail>=0、new>=0；守卫相对各自PMT需new>=0且类均衡净纠错>=0；这些是开发筛选，不是独立检验或患者/临床显著证据。失败/缺失不补0，不按单臂称完整比较，不用val调参数。

本轮只代码、协议与CPU验收；未来医学训练/eval仍需明确GPU授权、科学源码/协议公开冻结、完整生产CPU通过和全部真实零更新预检。当前CPU合成分类或参数变化不冒充医学贡献，全部identity、未解或代价过高也完整保留。当前无医学结果，不新增正式实验。
