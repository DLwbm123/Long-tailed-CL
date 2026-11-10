# R127后：三项新primary的context反馈筛选

本次实际审阅SDFT、SIPO、SDAR的所列方法/附录；未复现作者运行。SDAR固定commit仅核三个源文件33670B，全部GitHub显式7897proxy，不clone/install/下载模型或PDF。第三方源文本私有保留，不混进公开交付。以下作者方法与本地推断分开。

## 三个原始来源

[SDFT §3/§5/A](https://arxiv.org/html/2601.19897v1)：同一模型加入示范作教师，学生采样后作反向KL蒸馏；默认EMA教师，强调ICL质量，讨论小模型与语言伪迹限制。它可作同信息on-policy蒸馏控制；不能因on-policy或IRL解释就把全文答案监督算成独立验证奖励。本地只有类别标签，没有独立医学推理示范，原样医学推理/EMA训练方案并未获得验证。

[SIPO §3/A/B](https://arxiv.org/html/2609.36742v1)：用正确/错误答案教师的logp差作有界token证据，加到序列优势。有效错误回答在全错组也可有teacher项；无可判定错误答案时不加该项，全对组不造负答案。额外teacher前向仍须计费。原文用去group-std的二元优势、K10同步教师；这里与RaPO连续奖励/CTAN结合是明示适配，不能搬用其1/G符号裕量论证。

[SDAR §2/A及固定代码](https://arxiv.org/html/2605.15155v1)：RL加gate加权的学生token似然辅助项；论文自己在附录证明该形式，选定代码一致。它不是漏乘gap的代码bug，也不是完整反向KL梯度；g和teacher detach后，dL/dlogp_student为负的聚合权重乘g。不能把负gap软衰减写成该token被直接负系数惩罚，更不能从单项梯度断言整个softmax/Adam或任务效果。原任务的多轮技能库在本地不存在；类别名是完整答案，不伪造独立技能或多轮环境。

## 实际选择

选择一次**当前完整标签条件下，on-policy对比反馈相对直接答案监督和正教师蒸馏的增量资格实验**，进入R128完整冻结。不是R126旧图重放/old-reward mask的新名字：它仍replay-free、只当前TRAIN，不访问旧buffer；新变化在学生自己生成的错误token能否得到已知标签的对比监督。它也不是R113记忆编辑actor/R105全候选选择，没有学习样本选择器。原R125学生prompt、到达顺序、seed、G8/max24、TARS/RaPO与原失败门均保持相关约束。

标签是完整类别答案，teacher项明确算辅助监督，不能记成纯RL发现新临床证据。四臂是相同TARS_RaPO骨干、加全强度CE、加正教师sampled-KD、加对比teacher主；CE/KD获同一当前标签，查询/更新/学生生成预算相同，实际teacher/TF/token/时间不同完整计费。不存在从teacher向student直接传入答案。

R127证明了共同首模型菜单变化，不证明teacher在不同policy/context下必有正确临床依据，因此不采用旧prompt教师或更改部署菜单。SIPO当前标签可获得，目标问题是对比密集辅助监督在原严格全流程门下是否优于这些强控制；不能仅以非零teacher优势或正确组份额宣称成功。R125没存TRAIN逐token文本，不能从all-wrong组数补造valid-wrong teacher资格率，必须在新协议中保存远端私有判定并独立核算。

SDFT默认EMA与full-vocab estimator不机械冒充已复现；R128正教师sampled-KD为明示机制控制。SDAR暂不加入，以免把gate/额外技能或正确响应放大变成新模块堆叠。没有创新/独立临床声明，允许负结果关闭本构建，不扫κ/δ/teacher周期/CE系数/提示或新seed救分。

R128首次准备保守记本次入口心跳2026-11-10T02:19:39.063Z，独立23h<24h。该新问题没有重置R126旧反馈候选的原20h钟；R126仍未时间准入且0科学训练。只有完整冻结、CPU数学与访问/teacher资格检查、真实GPU1 native和完整矩阵外推准入后才进入训练。