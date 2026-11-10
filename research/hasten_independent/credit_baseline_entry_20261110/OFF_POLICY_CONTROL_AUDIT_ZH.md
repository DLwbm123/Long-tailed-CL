# 有界信用入口完成：已知gold注入不是新旧任务信息

这是R130负资格后的主线可行性审查，0新模型/编码/优化调用，未改R130结果/协议。三篇限额已全部用完；没有R131协议、root、native或训练。以下区分论文事实、静态代码、条件解析推导和未测结论，不以来源阅读当模型效果。

## 第二篇：LUFFY

[LUFFY v5 §3.1/3.2、§4/C.1](https://arxiv.org/html/2504.14945v5)混合外部正确推理轨迹和on-policy响应；实践令专家概率为1、去off-policy clip，使用f(p)=p/(p+0.1)。实验采用7 on+1 off，同时KL为0、有entropy项并改变归一化；它提供同轨迹SFT及RL+SFT对照。不能把整篇当成仅加一条gold答案。

### 静态路由与分支核对

当前作者仓库快照SHA 01f4ae066249e6965ce0d5512ccdf99a12895c74。最初两个通用verl文件不是混合路径；随后核对6个专用文件，合计8文件138639B，未读取ExGRPO子树。入口[main_mix_ppo](https://github.com/ElliottYan/LUFFY/blob/01f4ae066249e6965ce0d5512ccdf99a12895c74/luffy/verl/verl/mix_src/main_mix_ppo.py)经混合worker选择[mix_actor](https://github.com/ElliottYan/LUFFY/blob/01f4ae066249e6965ce0d5512ccdf99a12895c74/luffy/verl/verl/mix_src/mix_actor.py)，其use_off_policy_loss分支调用[mix_core_alg](https://github.com/ElliottYan/LUFFY/blob/01f4ae066249e6965ce0d5512ccdf99a12895c74/luffy/verl/verl/mix_src/mix_core_alg.py)。在target_probs缺省、off_policy_reshape为p_div_p_0.1且无额外clip的分支，使用带梯度的p/(p+.1)，不是detach后的任意权重。prefix_mask/eos_mask选择on/off项，归一化有条件分支。[默认YAML](https://github.com/ElliottYan/LUFFY/blob/01f4ae066249e6965ce0d5512ccdf99a12895c74/luffy/verl/verl/mix_src/config/mix_ppo_trainer.yaml)的off-policy开关为False，不等于论文实验启动命令；原始论文时点commit、完整launch override、运行和收敛证明均未验证，不能说默认即完整LUFFY或误报通用路径缺shaping为作者bug。API初次返回301后只跟进已核同仓库ID，全部GitHub访问显式代理。

## 第三篇：GOLD

[GOLD v2 §3.1–3.3/A.4](https://arxiv.org/html/2009.07839v2)采用示范离线策略梯度；把完整序列importance product近似成逐动作比率，再用经验行为分布忽略分母，正文承认该近似有偏。算法用滞后策略权重及floor、估计回报与baseline。只读相关论文节，作者代码未审；不是精确无偏importance correction或医疗continual结果。

## 对本项目的条件解析推导（不是作者运行证据）

当前任务已经给出完整类别label。把canonical name+EOS当off-policy gold，会给已到达的被查询样本一个正的完整答案；它没有提供额外推理轨迹/新临床反馈，也不会令从未抽到的类别产生query。是否改善优化仍可经验检验，不能说信息相同即梯度相同；但不得把这份直接答案监督记为独立RL发现或凭空解决旧图风险。

设p为同student prompt下gold token概率，A和mask/归一化常数c在本次更新detach。对上述无clip分支，L=-c A f(p)，f(p)=p/(p+gamma)。有

    grad L = -c A gamma*p/(p+gamma)^2 * grad log p.

因此同一参数点可用L_CE=-stopgrad[c A gamma*p/(p+gamma)^2]*log p构造局部精确的有符号加权似然控制。若A>0它是CE方向，若A<0则反向；不能仅因gold正确就假定RaPO total reward优势必为正（本任务还含retention）。正的token概率下gamma*p/(p+gamma)^2<=1/4，是解析界，不是新增医疗toy结果或参数扫描。GOLD固定滞后权重及回报时也呈加权gold-loglikelihood梯度，不能不带同信息加权监督控制就把收益全称RL。

对on-policy项，在我们每步只积累一次梯度、参数未更新且ratio=1的特定点，其梯度为-A_i grad log p_i；可与同rollout、有符号权重的似然/负样本控制比较。上述是局部一阶关系，不证明跨更新的采样分布、数据依赖权重、裁剪、KL、Adam轨迹或整个LUFFY等于普通CE；BF16执行路径也可能不同。它说明未来资格至少要分离gold监督、混合组baseline和on-policy负反馈，而不是重命名一个现成loss。

已知label的delta行为只在gold路径有支持，不能把行为概率写1就默认为覆盖整个自由生成空间的精确IS；如果用混合proposal，需明确序列/逐token密度和支持。该点是本项目确定性短答案适配的条件边界，不声称作者定理有bug。作者代码未核其全部preset/theorem前提，也不把GOLD的近似称无偏。

## 已作可行性决定

1. 拒绝单换global normalization作为独立后继（第一篇已审，不重做）。
2. 不准入“canonical gold+LUFFY/RaPO/TARS”简单拼接：gold信息等于当前完整标签监督，尚无区别于上述控制的长尾保留机制；不能追加reasoning/教师长轨迹/SFT来补这个差异，不能视为R128所有辅助监督已被否的反面证明。
3. 不准入“GOLD置信度重加权”简单拼接：它既不创造缺失query，也无新的旧分布反馈，floor或温度扫描不能代替机制。没有因此测试、失败或证明算法在医学上无效。

当前没有可运行的后继，而不是用户暂停或外部资源阻塞。R126仍受更早20h钟及原完整forecast限制；R130单转换吞吐不构成它的新容量证据，micro2写入原协议，不能暗改micro16称同次工程加速。候选若仍用旧反馈必须遵循原钟；真正新问题也必须完整冻结反馈/动作/访问/强CE或精确控制/成本/停止，再真实native与1.5forecast+reserve准入。已完成的3/3来源入口到此关闭，下一轮不能再包装成2篇未读或另开同类三篇扫描。
