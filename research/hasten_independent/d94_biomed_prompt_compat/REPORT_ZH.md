# D94 BiomedCLIP双prompt适配：原生兼容性检查

2026-11-02。工程检查完成exit0，软件与真实TRAIN原生检查均PASS。**这是接口与梯度兼容性结果，不是完整RLAP已训练或准确率改善。** 0optimizer/actor/VALTEST，未加载CPC/R93已训练权重。

## 实际适配路径

视觉：既有Biomed timm ViT-B16的patch_embed/_pos_embed后追加4×768 prompt，保持patch_drop/norm_pre/blocks/norm/trunk.forward_head和外部visual.head原投影。没有绕过原pool或fc_norm。文本：原BERT word embeddings的CLS之后插4×768 prompt，随后走原position/type embedding、attention mask、CLS pooler及MLP projection；仅在原尾4位均为padding时截尾，保留有效SEP，不用CLIP EOT argmax。NOOP不添加token，非添加零token。

## 数值和冻结检查

| 检查 | 实际值/结果 |
|---|---|
| 原encode_image与手工NOOP最大误差 | 0，门≤2e−5 |
| 原encode_text与手工inputs_embeds NOOP最大误差 | 0，门≤2e−5 |
| 视觉prompt梯度norm | 0.301202267，有限非零 |
| 文本prompt梯度norm | 2.07605553，有限非零 |
| base冻结且无参数梯度 | PASS |
| 小prompt状态strictrestore | PASS |
| 物理GPU1 ownPID绑定 | PASS |
| tokenizer长度/BERT位置上限 | 256/512 |
| 实际pad/CLS/SEP token ID | 0/2/3 |

软件另检查mask/shape/CLS保留、有效尾token截断必须拒绝、None为真NOOP。额外添加零token相对NOOP的最大输出差：视觉1.09731007、文本1.74339378；这是接口区别诊断，不是效果比较。

## 预算与信息边界

两个初始stage合法TRAIN S图像、不同identity-component，两个公开通用文本。实际8 image-row、8 text-sequence累计暴露（各distinct2），分别包含原生reference、手工NOOP、prompt梯度路径和零token诊断；不是8个独立病例。无新优化步骤、VAL或TEST，也没有医学分类指标。所有原图、IDs、features、基座/小prompt权重、原始日志保持私有。

仅双prompt有6144个训练参数；文本4×768相比R88的4×512每branch多1024，不伪称容量严格相等。完整RLAP还包含原型权重策略/专家/融合/阶段分支/其他投影，本次没有将其移植训练，不能宣称完成RLAP复现或GRPO改进。

## 费用

worker CPU27.062167s、wall76.313616s、原1800秒clock至worker结束224.860130s。原生coreCPU20.363656/wall44.424448已含worker，不重复加。峰RSS2052517888B/group1908322304B，GPUreserved1272971264B，disk106841B。准备、导入、SSH、guard、文档、发布CPU非0未完整采样；最初准入脚本引号导致语法失败、尚未模型运行，成本原clock包含但CPU未单独采样。随后更正传输方式，不更改科学/原生门。

## 后继还缺什么

通过本检查只消除了接口阻塞。完整医学骨干RLAP仍需唯一监督/memory预算、公平对照与参数差异披露、标准GRPO与RLPO区分、宏/尾/tiny/oldnew门及最坏训练/评估24h实测预算，不能直接从两个样本耗时外推全计划。原CPC/R93各轮保持关闭，不因本工程PASS恢复。骨干替换是主线诊断候选，不自动构成最终创新。

安全source/协议/工程聚合交付，不发布第三方全文或私有运行路径。VERIFICATION仅核保存收据和计数，没有重复native或hash基座。
