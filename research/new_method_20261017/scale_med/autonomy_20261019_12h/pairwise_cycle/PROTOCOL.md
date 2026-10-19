# 预先封存协议：成对边界是否受共享分数一致性约束

## 为什么不是继续调OvR

原ridge、softmax CE和独立OvR最终都是argmax(xW)，任意三类的边界差满足d_ij+d_jk+d_ki=0。OvR的负结果只说明独立binary loss和正负平衡没有自动解除这条几何约束。独立one-vs-one允许每对边界单独拟合，且概率投票可能产生一个共享线性分数矩阵表达不了的决策区域。尚无证据说此约束是真瓶颈；这是可证伪的问题，非因果结论。

已有方法本身完全不新：[Hastie/Tibshirani NIPS1997](https://papers.nips.cc/paper_files/paper/1997/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)和[Wu/Lin/Weng JMLR2004](https://jmlr.csail.mit.edu/papers/v5/wu04a.html)已有成对分类/概率耦合；后者比较投票与耦合。本次核对摘要/引言，尚未宣称逐页读完全文或实现其全部算法。若做固定概率胜率和投票基线，应准确命名，不能称实现Wu论文的概率耦合。

## 最小可区分控制

独立成对logistic产生反对称矩阵D_ij(x)。同一已拟合D分解为势差a_i−a_j与循环余项，a_i=(1/k)sum_j D_ij。主读出固定soft Borda=sum_j sigmoid(D_ij)；强控制Hodge势分数a_i和固定hard vote，不再训练门控或选择读出。对势差做soft Borda与argmax(a)等价，因为sigmoid严格单调；因此同一头的full soft Borda与Hodge差异能定位为保留循环余项的固定读出差异。它不证明训练过程的唯一因果瓶颈，也不意味着所有最优循环/耦合读出。

G/P/C原完整特征、原划分、原权重、同一lambda候选规则。每对只用对应两类，各类质量1/2，其他类权重0；这改变条件训练分布，需明确与OvR不同。所有pair共享一个臂级lambda，由初始681验证的固定主Borda BA选择后冻结。全部头封存才读测试。用已有本窗特征缓存，不再提特征或访问图像。

原三ridge及全部CE/S/B仍完整报告。主Borda各臂对三ridge、同头Hodge及hard vote共15比较；10,000类内已知组bootstrap，固定新seed74156，家族15调整。候选须对G ridge J≥1pp、对P/C ridge与Hodge/vote点差均正、5项调整下界全正；两个未来阶段BA不降；类4出生/转旧与类5出生不降；旧类宏代价≤1pp；类0/1在三个阶段各自代价≤1pp（相对自身ridge）。不能以投票/Hodge辅助最佳取代主Borda。通过也只是全历史强基线headroom，不宣称新方法或原有限记忆部署。

## 资源及未决/停止

不设耗时预算，继续原12h启动截止；新增文件预估<128MiB，复用已有约447MB窗口资产。单文件512MiB，全部本窗合计17GB软停/20GB硬限，1GB本地/Git预留。新env全部缓存指向NFS；纯CPU、CUDA_VISIBLE_DEVICES空。15矩阵拟合（初始9、未来6）覆盖129个二分类任务；原binary W梯度≤1e−7、解析梯度/HVP、行序/统计parity、全聚合JSON与失败回执预检缺一不可。

任何拟合未认证、输入不对齐、封存/隔离失败、空间风险记未决；完整却无臂过门停止该机制，不增seed、lambda、投票规则或换耦合方式追逐。与前轮共用已暴露测试，结论探索性，不能声称独立确认。

本文件在真实拟合前封存。若synthetic不能验证势差/Borda等价、保留循环影响和接受点认证，则先解决工程问题，不读真实测试。

数值规则冻结：复用前轮float64解析BCE/HVP及固定对角缩放，L-BFGS-B maxiter5000/maxfun20000/maxcor20/maxls40/ftol0/gtol1e-9；仅原W梯度>1e-7时最多10固定Newton-CG修正，rtol.05/maxiter1000，真实残差≤.1原梯度最大值且下降，Armijo1e-4最多30次二分。未认证记未决。独立每对目标为加权binary logistic加lambda/2非bias系数平方；矩阵求和与各对独立最小化等价。hard vote在0时判低编号类，最终票数并列也按低编号；Borda/Hodge最终并列按低编号。无按结果改规则。
