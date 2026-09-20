# CT9-P CURRENT_STATUS

COMPLETE_CT9P / STOP — 2026-09-19 20:45 UTC监测确认工程、训练、评价、报告全部exit0。六条Task4轨迹、60epochs/2750steps/6新checkpoint、12W及12新预测锁齐备；30/240含控制结果、200行配对区间已完成。worker、驱动与归档服务均已退出，禁止重新启动p28。

科学源e7f9b9afa36139d6ceffe818d0840fbee3fd7082保持不变，分支exp/ct9p-fixed-fd-perturbation。B1T−C3：HK 0.000pp [0,0]；ISIC +0.537pp [−0.238,1.370]。ISIC当前类+2.624pp、旧类−0.159pp；主BA仅2/3顺序改善，尾类未改变。不能宣称稳健成功。完整结论见FINAL_REPORT_ZH.md，NEXT_DECISION=STOP。

内部GPU residence5383.686秒，外层5398.083秒；峰值分配18.613GB。六新末状态归档my-gpu /tmp/p28archive真实NFS，约0.765GiB；六原Task3父仍在/tmp/p22archive。终态空闲3.23GB，无需清理历史资产。所有失败/部署证据及资源保留；旧fit/未来fit/test/reserved/oracle均0。本轮不扩HKTask5–11、不继续beta搜索。

公开交付只含源码、配置、谱系哈希、聚合指标与工程资源证据；私有图像、样本身份、特征/W/checkpoint/逐样本分数保留私有。实际发布提交以GitHub分支核验为准。
