# 局部查询域兼容性：前瞻CPU审计

上一局部loss记忆c6e24cfb已完整交付：局部界与有限比较过门，但每类anchor不同，linear控制也极精确，不能直接接共享分类器。本轮只检查共享参数域的必要条件，不再拟合记忆、不训练分类器或调参。

对每类a_c=2normalize(current mean)，有效球B(a_c,r)，r=1/8冻结。若同一个head方向v需要同时查询多个类的loss，必须v属于这些球的交集。任意pair ||a_i-a_j||>2r就由三角不等式证明交集为空；所需共同半径至少max_pair distance/2。均值center到全部anchor的最大距离提供最小包围球半径上界；它<=r则是可行见证。pair无违反不当充分可行条件，其他情况UNRESOLVED。这里考察完整参数空间，不额外施加||v||=2；交集在宽域为空更能排除norm2子域。

该条件只针对同一个v需要查询多类的实现。如果未来机制提供不同可行参数耦合/额外anchor/不同表示，需要新协议，不能把本轮结论推广到所有多类方法。对one-vs-rest或pairwise head在多个类上的共享查询尤其相关；本轮没有实际分类轨迹。

4CPUcase：几何算术与三规模。算术含同anchor可行、距离2r边界接触、正交anchor不可行、边长.24等边三角形pair都<=.25却circumradius>.125的反例、排列不变。三规模复用原确定性合成支持生成器/seed263/264/265/public比例：32d512K4、1536d9434K8、1536d1038K23。仅生成当前协议需要的均值；不读取旧NAS/旧节点/旧loss结果、不重跑旧拟合或评价。阶段2+1+1、4+2+2、13+2+2+2+2+2，仅按已见anchor前缀审计。全合成审计非医学current-train访问或隐私保证。

输出12stage几何行及最终所有287匿名类pair距离/下界，不输出向量、支持或逐样本ID。完全相同anchor的代数control必有交集，但不宣称其loss/分类质量，也不事后换主方法。0memory/query optimizer、0医学images/features/checkpoint/val/test/GPU；没有新训练seed/预算档位。case1800/总5500秒仅资源上限，每次CPU驻留/user/system/RSS全保留。

本轮结果是参数域可行性证据，不是新方法effectiveness gate。半径、存储和旧门不改；先完整交付再决定共享表示机制，不能以局部成功直接上医学GPU。
