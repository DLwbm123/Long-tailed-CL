# 同一响应目标的固定预算 Newton-CG 对照

## 依据与唯一改变

q400坐标等价检查通过，原W梯度比q399下降约48/137倍，但ISIC2.413765135e−5、HK7.655887995e−6仍未达到1e−6；200迭代终点均已公开交付。更低优化误差未带来一致query收益，不能用未收敛结果判响应目标优胜。

本轮唯一改变求解器：在同一q400可逆坐标中，用矩阵自由Newton-CG替代L-BFGS。保持原ridge初值W0，全部H+新S、beta8、.064系数、原Z的response_scale、G+.001I预条件化、数据访问与主判定。q399/q400均不续跑、不替换、不扫预条件系数；重新从同一原W0开始的独立固定预算算法对照。既往查询均为开发依据。

## 不变的目标与数据

p_i=1/(K*n_yi)，s2=sum p||z||²，Rnorm=d/(2*s2)*sum_i p_i sum_j[z_i@(delta_w_yi−delta_w_j)]²，目标类平衡CE(8zW)+.064/2 Rnorm，中心原W0。全部类的全部竞争列含旧旧/旧新/新新，self为0，无额外ridge目标项/筛正确/选竞争者。

G=sum p z z^T，L L^T=G+.001I，V=L^T W，Ztilde=Z L^-T，V0=L^T W0。原Z决定response_scale，不按Ztilde重算。最终恢复W=L^-T V，原W梯度L*gV用于收敛。保留Cholesky/特征变换残差<1e−10、初末原/变换目标差<1e−10。

四份centroid/ridge/affine/parameter DELIVERY绑定保持。候选seal前禁止query、q398参数头/结果、任何结果参考；q398参数头仅封存后加载作对照，不重训。q400仅已公开汇总用于历史比较，不加载私有头。禁止旧S/teacher/E/raw、新标签或额外数据。

## Newton-CG 与成本边界

每外步以自动微分算一次目标/梯度。用当前softmax概率缓存和解析Hessian向量乘解H d=−g，绝不构造完整H或显式求逆。对方向P，CE的HVP为beta² Ztilde^T[p*s*(ZtildeP−sum_j s_j(ZtildeP)_j)]；响应项以同一pair差构造精确二次型HVP。无阻尼项、无曲率超参搜索。

每步CG至多50次，残差相对阈值min(.5,sqrt(||g||_2))；非正曲率截断，如果组装方向不是下降方向则使用−g。Armijo c1=1e−4，初始步长1，每次乘.5，至多20次。线搜索失败或预算耗尽保存当前端点，不重启、不延期。内部V梯度∞≤1e−8可停止；最终仍原W梯度∞≤1e−6才算收敛。

每fit外步≤200，严格合计目标+梯度次数、解析HVP次数、线搜索目标次数、曲率概率缓存次数≤300，另一次终点目标+梯度沿用既定预算。即HVP不是免费的额外工作：总2fit合计≤600+2终点梯度；该计数是保守操作预算，不等于FLOPs。保留计数/停止原因/原和V梯度trace。每CPU任务900秒、总3600秒、2线程、0GPU保持。若本轮仍未收敛，禁止为追求成功进行无依据求解器/精度/预算搜索，先总结这一瓶颈。

工程仍2个二维fit。原梯度create_graph后用一次方向二阶导代替q400第二次变换梯度调用，与原坐标解析HVP及变换映回HVP对照atol1e−12；不增加梯度/导数调用数，每工程fit为closure+3。另记录2个显式解析HVP自检/数据集。保留非零中心解析梯度、显式pair、共同平移、2/3类trace、W/U链、复制目标/scale与旧默认目标检查。单位fit恢复W后仍对80步二分根及原W收敛检查；不预跑额外工程fit。

## 评价、资源和交付

三头raw_ridge/parameter_ce(q398)/linear_ce(current)，同2155query/6465logit；原ridge唯一主对照，原W梯度收敛且两片BAall严格增加、old/new各下降≤1pp才datasetpass，metric_gate_pass与criterion_pass分列。保留全类/组BA/CE/errors、margin/rank/top2/3/ties/competitors、ridge和参数到候选的实际逐样本配对及支持响应42类/12组。

支持ISIC2354/HK264，共2618；H原herding直接读无复制，零adapter1536DT/finite/count、完整mean<1e−12/ridge残差≤1e−7检查保持。2矩重构、2Cholesky/8formal三角solve，工程另8二维三角solve。封存后支持响应诊断5236个delta-logit样本。所有优化内CE/概率/HVP代价计入wall/CPU/RSS，公开各类次数，不以三头query次数冒充总运算次数。

2新W32256double258048字节，0额外持久推理模块、0基线重训、0adapter/encoder/teacher/memory更新，0GPU。成本从q400累计。完成退休后公开源码、完整汇总、成本、解析HVP自检/拟合trace/conditioning；与q400全42类/12组及2fit终点比较，不声称历史样本级救回。

这是优化机制诊断，不是已证实的老方法提升。H已压缩、query已曝光且患者分组未知；无全历史oracle、独立验证或医学可靠性结论。先冻结公开协议再启动，所有负结果保留。
