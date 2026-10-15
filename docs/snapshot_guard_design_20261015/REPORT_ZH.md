# 固定快照全 fit 守卫：完整 CPU 结果

科学源 `ce45194b5ad04161fd2854faf1fc9e8456c785c7`。全部 9 个 CPU case 已终态。无 GPU、医学图像/checkpoint、val/test、optimizer；不是正式医学效果。

| case | 状态 | 约束/正 dual | 全点最小 slack | 已认证收益 | 判别变化 | 驻留秒 | peak MiB |
|---|---|---|---|---|---|---|---|
| small_oracle_schema | PASS | 不适用 | 不适用 | 不适用 | 不适用 | 11.529396 | 442.640625 |
| legacy_ISIC_T3_Core | CERTIFIED | 66038/109 | -2.26438645e-09 | 0.304487743 | 0.894925071 | 149.981984 | 1399.35156 |
| legacy_ISIC_T3_LT | CERTIFIED | 66038/109 | -2.15222268e-09 | 0.304512117 | 0.894967102 | 152.231767 | 1601.96875 |
| ISIC_T3_epoch_init_Core | CERTIFIED | 66038/513 | -2.0879204e-09 | 0.309675665 | 0.96745015 | 96.4029494 | 1986.42578 |
| ISIC_T3_epoch_init_LT | CERTIFIED | 66038/512 | -2.9082457e-09 | 0.309680332 | 0.967601721 | 144.480031 | 1859.79688 |
| ISIC_T3_boundary_final_Core | CERTIFIED | 66038/515 | -1.6973261e-09 | 0.309681343 | 0.967192633 | 104.163253 | 2057.90234 |
| ISIC_T3_boundary_final_LT | CERTIFIED | 66038/514 | -3.61224908e-09 | 0.309685578 | 0.967341794 | 97.4068229 | 1931.05859 |
| HK_T6_boundary_final_Core | CERTIFIED | 22836/149 | -2.7794788e-09 | 0.104123501 | 0.900890075 | 193.801219 | 3733.51953 |
| HK_T6_boundary_final_LT | CERTIFIED | 22836/148 | -2.39650814e-09 | 0.104154639 | 0.901260011 | 216.579228 | 3725.83203 |

全部详细目标/保留比例/identity/合成救回破坏、KKT/gap、支持秩与逐次资源见 CPU_SCALE_SUMMARY.csv；活跃增长见 ACTIVE_DUAL_GROWTH.csv。未解的最后迭代不是可用头，不填 identity 或0。小枚举、排列/公共logit、关闭精确与证书结果完整保留于 CPU_RESULTS.json。

独立 CPU 驻留合计 1166.576650228 秒；user 2365.323555000 / system 53.651161000 秒；peak 3733.51953 MiB；父 wall 1166.709580658 秒。pipeline/operation 为包含区间，不相加。历史 GPU 不变。

## 限制

保护仅固定 encoder/frame 的 epoch 初始化与最终 boundary fit 支持，相对同状态 OFF；encoder 更新后失效。合成判别变化和二次收益不证明医学分类收益。已知旧 affine 不验证真实运输外推。checkpoint force 可检查 stationarity，无全支持不能独立复验所有 margin。未接生产，不是 GPU-ready。所有失败与未运行范围如实保留，不能用单 case 通过代替整体可行。

## 直接结论与全部负结果

八条1536维规模问题均通过独立全点/KKT/gap数值门，全部非identity，0未解、0超时、0科学失败。旧同一ISIC最大问题在新算法下109正dual，四条ISIC快照512–515正dual，HK148/149，均未依赖旧64逆方向缓存上限。可行性容差为2e-8，最差slack−3.612249085e-9；不是精确实数无损保证。

八条合成救回/破坏全部0/0；判别头发生变化且目标改善，不等于分类有益。设计没有观察真实医学fit、真实PMT拟合误差、独立Core/LT轨迹效果或val新类间隔，不能据此预测医学增益。全部positive/negative和限制见ANALYSIS_ZH.md。

工程记录：报告helper首次代理push连接失败后同代理成功；最初本地等待collector退出且无落盘记录，原因未确定，远端终态确认后一次收集成功。没有重跑科学CPU、没有丢失已记录的科学成本。等待进程驻留未单测，保留null，不伪造。详见ENGINEERING_AUDIT.json。
