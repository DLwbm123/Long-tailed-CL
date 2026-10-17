# 冻结响应更新的旧H空间投影消融

## 证据与唯一操作

q402在全部类/结局中发现：响应头旧类受损ISIC4+6例、HK1+1例的固定基线竞争者扰动绝对值均以旧H空间外分量为主；但ISIC旧类救回6+4例及新类救回15+16例同样由空间外分量主导。竞争者切换还可能使固定竞争者的正outside分量伴随实际损伤，不能仅据分解断言删除outside有益。

本轮只执行单一固定消融W_projected=W0+QQ^T(W_response−W0)。Q取同一合法旧H的固定数值秩basis，W0原ridge、W_response为已交付q401未收敛预算端点。保留空间内更新，空间外恢复原ridge；强度固定为完整投影，无混合系数、网格、数据选择或重新拟合。不加载参数头，三头预算用于ridge/full_response/projected_response。

q401未收敛与完整主判定失败不变。投影头未经优化收敛检查，本轮不宣布方法成功、不部署；只检验上述空间分量是否同时承载伤害和收益，为后续是否值得定义约束优化提供依据。

## 封存与恒等式

从旧H原归一化特征thin SVD构建Q，阈值max(H.shape)*float64epsilon*smax；不以query选秩，支持仅使用原H+合法新S。先验证Q正交与H重构<1e−10；投影后更新位于Q空间（元素误差<1e−10），且旧H上全已见类别beta8 logits与完整响应头一致<1e−10。随后保存derived head并seal，才读q401结果参考和query。

来源绑定centroid/ridge/response三份DELIVERY；response为q401源码8fb4ab6b2a8c5d0966f902ff8af89d3d34a75ed5、交付b8cc1aa3eca875ff718e76256ffa86cd9df17e34。均1536DT/finite/no bias/零adapter/model一致；H原herding直接读取，合法新S、逐类count保持。禁止旧S/teacher/E/raw/新标签或额外数据，主动测试访问守卫。原头不覆盖，投影头仅保存到新运行根。

二维固定例子检查投影保留H响应、幂等、空间外回到原ridge且与完整更新确有差异，无fit/gradient/HVP。新S不参与选择Q，query不参与投影。

## 评价与固定判定边界

实际重算三头全部query分数、竞争者、预测、margin/rank/top2/3/ties、CE、errors old/new、全类/组BA。ridge与full_response聚合须匹配q401原结果（整数精确、CE1e−10）。支持三头每类CE/correct也保留，原ridge/full_response的全类平均CE分别匹配q401初始/最终支持CE，投影在旧H上的响应一致不代表新S保持拟合。

直接逐样本配对ridge→projected、full_response→projected。另对全部类记录full_harmed及其中repaired/remaining、full_rescued及其中retained/lost、原ridge与full均正确却被projected破坏、原两头均错被projected救回，以及ridge/full到projected最强竞争者切换数。包括所有新类，不能只报旧伤害被修复。

原ridge仍唯一主对照，描述性metric gate维持每片BAall严格增加且old/new下降≤1pp；不改变门槛、不换微准确率。输出full_method_success_evaluated=false和method_success_claim=false，清楚说明没有新的优化收敛证据，metric通过不构成完整方法通过。保留q401来源收敛/主门失败事实；不把未测梯度写成已收敛或已失败的新求解结果。

## 预算、交付和边界

2CPU任务各900秒、总3600秒、2线程、0GPU。0正式/工程fit、0梯度/HVP，2thin SVD、2确定性派生头；2W32256double258048字节，未来推理无需Q模块，现阶段仅诊断。支持特征2618、共享query2155；支持三头logit7854、query三头6465，全部记CPU/wall/RSS，不声称免费或严格等FLOPs。

0adapter/encoder/teacher/memory更新，成本从q402累计。预计支持class63、query class126、配对84、三头margin126、转移42、head21、supportcounts21、basis2及全组汇总、竞争者明细和描述性gate4。所有结果和负结果保留、完成退休后公开去身份化汇总源码报告；不公开私有head/特征、个体标识或服务器路径。

所有query已经曝光、患者分组未知。旧H空间外不等于完整H+newS响应目标零空间；消融改变完整支持目标，不能以它的性能代表该目标的最优解。它不证明压缩损失、全历史oracle、完整CL或独立医学可靠性。根据修复与损失的完整结果决定下一步，不扫描投影强度或恢复既有拟合。
