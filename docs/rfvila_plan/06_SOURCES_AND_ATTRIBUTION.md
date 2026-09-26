# 来源与本轮适配边界

## S1 用户上传论文（本轮主要方法来源）
Zhao et al., *Advancing Analytic Class-Incremental Learning through Vision-Language Calibration*。
用户PDF为 arXiv:2602.13670v3，2026-05-07；20页。
本地原PDF SHA256：`3035ee0994db5e8c30f493438576ec5e3c3e2a444c4fa546545f05c65b23edab`。
论文链接：https://arxiv.org/abs/2602.13670

对应位置：p.3固定Gaussian+ReLU/解析目标与Gram不一致；p.5 UGC式(4)、CSE式(5)/(6)及加法；p.6维度16384/Task1 LOOCV；p.8二次内存限制；p.14–15算法与文本缓存。
本轮无需转载原论文表格或共享用户PDF；代码/报告应引用论文。

## S2 作者公开实现（已核对版本）
仓库：https://github.com/byzhaoAI/VILA
固定提交：`ad4293af236d86e05d8e850d2ce33638fc16c7ed`。
文件：https://github.com/byzhaoAI/VILA/blob/ad4293af236d86e05d8e850d2ce33638fc16c7ed/models/vila.py
https://github.com/byzhaoAI/VILA/blob/ad4293af236d86e05d8e850d2ce33638fc16c7ed/utils/VILA.py

本次核对差异：代码Task1 80/20特征切分与1e-8…1e8 ridge grid；初始W乘0.9；CSE用0.8s+0.2v。
论文描述Task1 LOOCV及1e-8…1 grid，CSE为s+v。不可默默择一充当严格复现。
本轮不复制0.9初始缩放，也不运行作者默认dataset/test/Task1训练脚本。引用随机特征+UGC+CSE思想，不声称新发明。

## S3 本项目医学实验
提交：`b01ba12f86706c44dfcdbed81c7305493ce354dc`。
报告：https://github.com/DLwbm123/Long-tailed-CL/blob/b01ba12f86706c44dfcdbed81c7305493ce354dc/docs/nb2_medvlm_r1_20260926/FINAL_REPORT_ZH.md
协议：https://github.com/DLwbm123/Long-tailed-CL/blob/b01ba12f86706c44dfcdbed81c7305493ce354dc/docs/nb2_medvlm_r1_20260926/protocol_lock.json
核心线性/先验代码：https://github.com/DLwbm123/Long-tailed-CL/blob/b01ba12f86706c44dfcdbed81c7305493ce354dc/tools/medvlm_math.py
现有source_lock列出了G/B运行与模型依赖，应在新机器实核，不重新发明路径。

历史protocol存在ignore-time override，resource_report记录GPU0/2/3；这些只属于旧run。本轮明确单GPU、12小时，不能继承。
历史backup_report仅确认NFS流式复制和可读回执，并写destination_bytewise_verification=false。本轮需补核验关键集合。

## S4 技术实现参考
PyTorch官方Cholesky文档：https://docs.pytorch.org/docs/stable/generated/torch.linalg.cholesky.html
用于对称正定线性系统的稳定求解；支持double。正式环境版本沿用并锁定已验收环境，不因网站最新版本强制升级。

## 本轮设计而非来源直接结论
类别均衡目标及Kλ、m4096尺度归一化、两独立RP种子、Task1组件CV9点网格、Split/RPLIN对照、utility门槛、时间/备份规则属于本次协议设计。
没有任何来源保证这些适配在ISIC/HK有效；全部作为可证伪假设。
BiomedCLIP与Generic的系统差异包含语料/文本编码器/预处理等，不能单因果归因为医学语料。
