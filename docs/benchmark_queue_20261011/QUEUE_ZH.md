# 用户授权的顺序队列

Tissue完整结束并GitHub/NAS交付 → NIH-CXR-LT单标签配对 → RFMiD多标签配对。所有新运行仅GPU0/1，GPU2/3禁止；当前Tissue冻结源不改动。各轮启动前独立协议/数据审计/CPU检查/源码冻结，结束先完整公开正负与成本。不会依据前轮val改候选或添加CIFAR实验，RL暂停，所有医学及MedMNIST test封存。

NIH作者20列单标签与train/val已核对，所有影像现有缓存齐全；协议见[NIH](../nih_cxr_lt1_20261011/PROTOCOL_ZH.md)。排队状态不等于已启动，真实GPU预检尚待Tissue交付及资源准入。

RFMiD官方训练、验证下载链接已交给用户，等待用户提供保存目录；不能把数据或29类映射核对称已完成。其设计与待冻结项见[RFMiD](../rfmid_ml1_20261011/PROTOCOL_SPEC_ZH.md)。保持multi-hot共病标签，不能套用单标签argmax/六BA门来伪造多标签结果。实际CSV映射、支持度、阶段与预算核对后才能发布最终LOCK并启动。
