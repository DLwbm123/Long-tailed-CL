# 固定预算的响应目标优化坐标对照

## 依据与唯一改变

q399两组到达200迭代，原W梯度∞分别.001168742/.001050786，未满足1e−6；工程检查通过不等于目标充分优化。HK两片指标改善仍未通过完整主判定。原记录与费用已经交付，本轮不延长、恢复或替换q399，而是另一个固定预算的优化坐标对照。

保持同一个响应目标、原ridge初值W0、支持H+新S、系数.064、beta8、响应尺度规则、优化次数和wall上限。仅用已经构建的类平衡支持Gram作可逆预条件化：G=sum_i p_i z_i z_i^T，L L^T=G+.001I，V=L^T W，Ztilde=Z L^-T，中心V0=L^T W0。使用原ridge已有.001，不搜索预条件系数、不截秩、不作特征选择。

Ztilde V=ZW，Ztilde(V−V0)=Z(W−W0)，因此CE和全部类别间响应目标严格不变。response_scale仍由原Z计算为d/(2*s2)，不按Ztilde重新归一化；.001只用于坐标变换，不是新增目标正则。最终恢复W=L^-T V，推理依然使用原特征、原头形状。

## 目标、收敛与可比性

原目标为类平衡CE(8zW)+.064/2*Rnorm，Rnorm=d/(2*s2)*sum_i p_i sum_j[z_i@(delta_w_yi−delta_w_j)]²，p_i=1/(K*n_yi)。全部合法H+新S与全部竞争列均保持，含旧旧/旧新/新新。

L-BFGS在V变量中仍lr1、max_iter200、max_eval250、history20、strong_wolfe、tolgrad1e−8、tolchange1e−14，closure硬上限300/fit。坐标会改变搜索路径及内部tolgrad含义，不能称优化过程相同。每次trace同时记录V梯度和原W梯度gW=L*gV；最终通过条件仍原W梯度∞≤1e−6，不能因V梯度较小放宽标准。若仍未收敛，只保留预算终点，不延长/重试/换预条件系数。

同一目标和初值下，比较q399与本轮的原目标、支持CE、保护值、原W梯度和预算内迭代效率。不同收敛程度下的query差异仍是预算端点描述；本轮不自动解决参数保护与响应保护在匹配新类拟合后的方法比较。

## 数据、评价和检查

数据/来源/访问守卫沿用q399：ISIC2354/HK264的H+新S，原零adapter、1536DT/finite/count、完整均值与ridge重构；四份DELIVERY绑定保留。参数对照仍复用q398，不读q399私有头、不重复拟合对照。拟合前禁止q398头、query和结果参考，先保存候选seal再加载与评价。禁止原旧S/teacher/E/raw/newlabel。

三头仍raw_ridge/parameter_ce(q398)/linear_ce(current)，共享2155query/6465logits，完整配对ridge→candidate和parameter→candidate。主规则不变：原W梯度收敛且每片BAall严格改善、old/new各下降≤1pp，两个query均过才datasetpass；不部署、不切主。完整报告逐类/组CE与错误去向、margin/rank/top2/3/ties/competitors、配对和支持响应。另从q399已公开RESULTS做全部42类/12组历史汇总和优化终点比较，不对历史汇总声称逐样本救回/伤害。

工程二维用非对角下三角L=[[2,0],[.3,1.5]]检查变换前后目标和gW=L*gV，恢复W后对照原80步二分根及收敛。保留显式响应求和、2/3类trace、共同平移、W/U关系、旧默认目标和类内复制目标/scale检查；本轮第二次工程autograd用于坐标梯度链，类复制梯度已由q399验证，不新增反传。每工程fit梯度数仍closure+3。

正式拟合前Cholesky和feature变换相对残差<1e−10、原/变换初始目标差<1e−10，恢复W后最终目标差<1e−10。新输出solver_coordinates、conditioning、原/优化梯度、分解/三角solve计数；旧协议缺字段默认native不改变原行为。

## 预算与资源

2CPU任务各900秒、总3600秒、2线程、0GPU；2正式fit+2工程fit，正式closure≤600+2终点梯度。相比q399新增2个1536维Cholesky分解和8次正式三角solve（每data feature变换1次、终点/初始/中心映回3次），全部计入同一wall上限与实际CPU/峰值RSS，不声称等FLOPs；没有显式求逆。工程每data另有4个二维三角solve，无工程Cholesky，固定L直接给出。

2完整支持矩重构、0基线重拟合，支持2618/查询2155/三头6465logit，封存后支持响应诊断5236个delta-logit样本保持。2新W32256double=258048字节，无额外持久推理模块；L、Ztilde和优化历史仅工作内存。0adapter/encoder/teacher/memory更新；成本从q399累计。

所有query已用于开发、患者分组未知。没有保留E或新独立证据，不声称新方法成功、全历史oracle、完整CL或医学可靠性。q399全部负结果保留，先公开本轮完整源码/协议再启动，结果完成后报告并继续按证据推进。
