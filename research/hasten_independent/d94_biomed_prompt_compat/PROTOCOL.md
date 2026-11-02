# D94 BiomedCLIP双prompt原生兼容性检查

2026-11-02。用户此前提出RLAP替换医学骨干，持续授权范围内先验证必要接口。独立工程检查，非完整RLAP/GRPO/方法效益run；不恢复R93/R91/D92，不读其科学端点/VAL，不复用其已训练权重。新FIRST起1800秒含实现、准入、软件、原生、修复、分析交付，不重置。0正式optimizer/actor/VALTEST，0新训练。

唯一适配：两个prompt各4×768；视觉在timm _pos_embed后追加tokens，后续patch_drop/norm_pre/blocks/norm/trunk.forward_head/visual.head保持原路径。文本在原word embeddings的CLS之后插4tokens，再走原BERT position/type embeddings/CLS pooler/MLP投影；插4个mask有效位，只截原尾4个padding，不可静默截有效token或SEP。NOOP为不加tokens，零prompt不是NOOP。仅检查底座与prompt，不实现整个RLAP专家/权重优化，也不将双prompt成功当识别增益或创新。

软件一次检查：拼接位置与shape/mask、尾有效token时拒绝、CLS保留、None不增加tokens；不重复旧R90/R91数学toy。

原生使用既有只读Biomed pretrained和本地tokenizer，无下载或模型复制。两个既有初始stage合法TRAIN S样本、不同identity-component，两个公开通用文本。检查原encode_image/text与手工无prompt实际路径maxabs<=2e−5；文本pad与tokenizer一致/CLS0/SEP未截；两个prompt有限非零梯度、所有base冻结无grad；小prompt状态strictrestore；额外零token输出差仅诊断不作科学门。原生最多8 image-row和8 text-sequence暴露（distinct2各），0optimizer/VAL/TEST，不报告准确率。固定seed74002，初始小prompt私存不hashbase。

资源：GPU1only，64GiB RAM/14GiBfree准入，reserved<12GiB；CPU4/400000s/file1GiB、worker28/group29每2s总30GB、共享独占memorylock；新增disk1GB+512MiB早停，每server<50GB。remote-home，NFS小probe和free先检查，不复制底座、不rootspill/swap/quantize、不动他人或删除历史。独立守卫不依赖chat。若准入不满足，不启动worker/等待越过原1800秒。失败先核原clock/全部费用/forwardcounts，科学0只同钟工程修复，不盲重复native。

成功需软件、全部native数值/冻结/梯度/restore/ownGPU检查通过、exit0、8或更少vision/text暴露实际计数完整报告与安全source/协议/工程摘要交付。不得发布图/ID/逐例输出/权重/特征/原始日志/privatepaths。发布仍本机显式7897proxy、原deliverygit/branch、exactremote/匿名REPORT200/source+localDELIVERY。

本工程即使通过，完整RLAP的监督/memory预算、公平对照与长尾/tiny/oldnew门、最坏正式训练CPU/GPU/24h预算尚未冻结，绝不自动启动科学fit。医学文本prompt相比R88每branch多1024参数，如实披露，不用无梯度占位参数伪配平。
