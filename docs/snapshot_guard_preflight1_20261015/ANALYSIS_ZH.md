# 四条真实零更新预检：完整分析与限制

四项全部通过；PROGRAM 为 PREFLIGHT_COMPLETE_STOPPED / STOPPED_NO_TRAIN，active 空，父进程 q247c 与全部 q247w 实际退出。一次终态收集。正式更新、policy 更新、正式训练和val/test访问均0，不因通过而解锁训练。实际预检部署源16ad48063895e58fadd93f6eba744d9d82599521，生产数学源7a077b69f16d843573b2bc7af5a46d9701c29bb0不变；报告源码独立，不同步进科学源。

## 真实反传与完整模型不变性

四条真实首task、首batch64，1536维ViT生产路径及epoch guard证书均通过。两个adapter pool梯度均有限非零，总梯度范数约0.01196–0.01683，原clip5未触发数值异常。optimizer.step被独立入口硬性拦截，四次均optimizer0/policy0；最大旧bank shadow之后又与模型初始化state_dict逐tensor比较。

每个模型839个参数tensor精确不变；注册buffer为0，buffer集合不变是空集合检查，不能声称实测了非空buffer更新路径。不能把初始化、首batch或模拟/合成预检当正式训练更新。

## 最大旧bank：真实与合成证据分别报告

真实首task没有旧bank，不读旧医学图像或旧医学checkpoint。每项另有原PMT GPU synthetic最高旧bank/输入rank1536 probe，以及新guard最高bank的合成full-fit生产Session：ISIC6旧类/8总类/Nfit9434/allcurrent11835，HK21旧类/23总类/Nfit1038/allcurrent1298。guard支持rank分别1536/1038；HK这一项不能称满输入rank1536，满rank1536证据来自另一个PMT probe。完整约束始终保留、无active cap，不调强度/阈值。固定已知旧affine(2.8,.65...)，不是对真实旧域漂移的验证。

shadow将bank/offbank/regional/R/pairs/完整fit输入暂驻该GPU，再调用原CPU full-dual guard投影，真实模型和第一batch梯度同时保留在GPU。guard并非GPU上运行L-BFGS-B；测到的是这一混合路径的驻留、显存和费用。fixture包含固定的scope/version拒绝及合成checkpoint重建，全部时间都在实际驻留内。

| 项目 | 旧/总类 | fit/支持rank | 完整约束/正dual | 最小slack | stationarity | retained gain % | 判别变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre_ISIC_guard | 6/8 | 9434/1536 | 66038/515 | -3.4226e-09 | 2.4487e-12 | 99.979941 | 0.967192632 |
| pre_ISIC_LT_guard | 6/8 | 9434/1536 | 66038/514 | -3.5381e-09 | 2.5087e-12 | 99.979941 | 0.967341794 |
| pre_HK_guard | 21/23 | 1038/1038 | 22836/149 | -2.4527e-09 | 1.1880e-12 | 99.987993 | 0.900890076 |
| pre_HK_LT_guard | 21/23 | 1038/1038 | 22836/148 | -2.2111e-09 | 1.2084e-12 | 99.988032 | 0.901260013 |

四条合成guard均非identity，输入统计不变。最差slack−3.538094328e−9、stationarity<=2.508742132e−12、互补<=8.600294317e−14、绝对gap<=5.023537142e−12；原冻结KKT门全部满足。PMT virtual moment最大误差6.938893904e−18，PMT residual最大8.642027873e−10，全真实首batch及shadow综合maxsolve最大8.924413626e−10。

这些是工程/已知合成算术证据，不是分类或医学改善。该预检sidecar未导出救回/破坏指标，不能把未提供填0；先前独立CPU相同fixture的0/0不算本轮新医学结果。旧PMT开发结果中新类损害/HK弱仍未被本次预检否定。

## Checkpoint、主机输入与显存

四条shadow实际私有合成checkpoint序列化字节数见RESULTS/PREFLIGHT_SUMMARY（不含真实ViT参数或optimizer，不代表未来完整模型文件）。force重建相对误差最多1.297951438e−10；没有完整支持，加载时只能核对重建/驻留和当时标量证书，不能独立重新验全部margin。

成功guard投影的逻辑GPU到CPU输入字节分别437258704、437651920、900994008、902124504，合计2678029136。它不是全fixture CPU到GPU/GPU到CPU总流量或实测PCIe带宽，额外负向调用和序列化也未拆分传输计数；耗时已完整包含。不能把逻辑字节数当实际带宽或全部复制开销。

PyTorch per-process peak allocated分别11.926920/12.110437/12.295539/12.476758 GiB，reserved分别12.673828/12.892578/17.457031/17.640625 GiB。allocated和reserved峰值未必同一时刻，二者峰值差不能当某时刻碎片大小。初始NVML曾显示0MiB，未独立采样物理进程显存峰值，因此NVML peak保持null；不用该0MiB否定allocator实测。当前24GiB准入下四项通过，不保证完整多task正式训练/评价的峰值或24h总预算。

## 逐次费用与全部失败

四次成功驻留：ISIC Core569.213006735、LT560.321662188；HK Core561.475058079、LT576.873324633秒。合计2267.883051634 GPU秒＝0.629967514 GPU小时。父wholewall1146.708442926秒（19分06.708），因两卡并行不与GPU秒相加。worker operation合计2223.846936496秒、其中guard shadow722.154845930秒，都是驻留内部区间，不另加。父检测轮询分辨率约0.25秒，驻留是spawn到父观察exit的时间，含导入/提取/反传/解算/序列化和检测尾差；未单测纯GPU kernel时间。

本次新增CPU小harness独立驻留45.171773958秒（user9.915315/system2.192733、peak467.0078125MiB），单列，不混入GPU秒。原8项生产CPU与更旧失败轮已交付，不在本轮重跑/重计。四GPU失败0、重试0、新CPU失败0；所有实际驻留均保留，未按job或operation去重。旧闭合391490.24710153416秒只加本轮一次，当前全部闭合393758.130153168秒；不能再叠加旧cumulative。没有单价，金额未提供，不虚构货币费用。

工程事故未隐藏：首27f3a59因本轮忽略规则遗漏发布两份私有配置中的路径元数据，无凭证/医学数据/模型；16ad480已从最新树移除并忽略local_runtime，历史仍残留，未强推改史。启动report首次代理TLS push失败，publisher先因旧remoteHEAD拒绝NAS写，第二代理ls-remote TLS失败，随后同代理HTTP1.1成功。均无新科学operation或GPU重启；完整私有收据保留，公开仅匿名原因。最终结果交付也只经显式代理核实后NAS独立保存，未确认时不称完成。

本次授权到此停止。完整四项表、双pool梯度、839参数/0buffer、完整约束、显存/墙钟/费用和所有工程负面已公开；没有正式训练、val调参或test读取，不自动新增训练/科学轮。通过预检不等于方法有效或临床/患者独立/统计显著性证明。
