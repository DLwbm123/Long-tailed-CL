# 快照守卫生产 CPU 检查完整报告

科学源 `7a077b69f16d843573b2bc7af5a46d9701c29bb0`，当前八项CPU验收：全部通过。没有GPU启动或真实GPU预检。

| 检查 | 状态 | 完整约束/正dual | 判别变化 | 收益保留 | 驻留秒 | 峰值MiB |
|---|---|---|---|---|---|---|
| production_chain | PASS | 未提供/未提供 | 未提供 | 未提供 | 58.0833258 | 504.410156 |
| actual_ViT_CPU_zero | PASS | 未提供/未提供 | 未提供 | 未提供 | 34.8166111 | 2490.87891 |
| ISIC_T3_epoch_init_Core | PASS | 66038/513 | 0.96745015 | 0.999800356 | 127.694738 | 2336.22266 |
| ISIC_T3_epoch_init_LT | PASS | 66038/512 | 0.967601721 | 0.999800354 | 170.508851 | 2162.125 |
| ISIC_T3_boundary_final_Core | PASS | 66038/515 | 0.967192632 | 0.999799407 | 143.986718 | 2269.73047 |
| ISIC_T3_boundary_final_LT | PASS | 66038/514 | 0.967341794 | 0.999799406 | 151.750326 | 2333.85938 |
| HK_T6_boundary_final_Core | PASS | 22836/149 | 0.900890076 | 0.999879933 | 174.038685 | 4528.10156 |
| HK_T6_boundary_final_LT | PASS | 22836/148 | 0.901260013 | 0.999880318 | 195.817663 | 4529.46484 |

本轮独立CPU驻留1056.696917263秒；user 1949.487690000/system 62.374770000；峰值4529.46484MiB。父wall 1056.841117570秒另列，pipeline/operation均包含于驻留，不相加。首轮八次解析失败2.190434728秒已77bd9e29完整交付，本生产请求两轮累计实测驻留1058.887351991秒。无GPU费用，旧历史391490.24710153416秒不重计。

关闭Core/LT和PMT的精确回归、全stage训练/guard-aware评分、失效/非法scope/预算/早val及guard schema结果见CPU_RESULTS。模拟eval仅合成数据，不是医学val。真实ViT两张合成图CPU零optimizer反传与高维全部fit/最大旧bank是分别记录的检查，不能合称四条真实GPU预检。

预算保持新四轨迹1288更新；Core/LT及两背景PMT复用，不训练对照。54个守卫事件不加图像提取，但增加CPU原生逆、主机逻辑输入量和私有证书；未来GPU worker驻留已包含CPU等待，配对控制推理不为0。原canonical/student/teacher曝光及存储/推理预算见BUDGET_SPEC。

守卫权威double头与旧存储float对照精度差异必须报告，same-state OFF/proposal不替代独立控制贡献。只保护固定encoder快照fit margin；encoder更新即失效。不保证val无损、旧旧风险识别或医学净纠错。

当前生产入口已接线，CPU通过也不自动GPU-ready：未运行GPU、未读医学图像或checkpoint、未读val/test、未有新医学结果；只有明确后续GPU授权与全真实零更新预检后才能启动。所有缺失保持null，不补0。

完整调用链、保护时点和失效、checkpoint精度契约、全约束负结果及预算分析见 [ANALYSIS_ZH.md](ANALYSIS_ZH.md)，实际资源与未测项见 [BUDGET_AUDIT.json](BUDGET_AUDIT.json)。六条高维合成检查救回/破坏（相对同状态无约束ON proposal）全部0/0；可行非identity不构成分类改善。18个stage/54个事件/36次证书失效、四合成eval和四模拟零更新入口通过；12层真实预训练ViT两合成图CPU反传通过，0optimizer及参数buffer精确不变。

本轮六条高维共824次native逆，最差slack−3.538094328e−9；stationarity≤2.508742132e−12，互补≤8.600294317e−14，绝对gap≤5.023537142e−12。checkpoint没有支持不能独立全点重验。当前CPU无host传输不等于未来GPU无传输；完整医学checkpoint/GPU峰值/全campaign成本尚未测，null。
