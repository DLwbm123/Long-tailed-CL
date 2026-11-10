# R128 当前完整标签对比辅助监督资格实验

已知SIPO机制的TARS/RaPO适配，不是作者原样复现、独立临床或创新结论。完整标签teacher项是辅助监督，不把它的收益自动记成纯RL必要性。

|arm|postmacro %|posttail %|forgetting pp|
|---|---:|---:|---:|
|BASE_TARS_RAPO|23.1254|3.9394|16.0038|
|CE_AUX_TARS_RAPO|16.0751|4.1669|40.4195|
|POS_KD_TARS_RAPO|15.5163|2.9464|36.0767|
|CONTRAST_TARS_RAPO|22.7765|2.9248|16.7067|

九门通过3/9：{"post_macro_over_BASE_ge_1pp": false, "post_tail_over_BASE_ge_2pp": false, "initial_task_head_drop_ge_minus1pp": false, "worst_old_tail_over_BASE_ge_minus1pp": true, "post_macro_over_CE_AUX_TARS_RAPO_positive": true, "post_tail_over_CE_AUX_TARS_RAPO_positive": false, "post_macro_over_POS_KD_TARS_RAPO_positive": true, "post_tail_over_POS_KD_TARS_RAPO_positive": false, "classwise_forgetting_less_than_BASE": false}
原完整菜单/参数的首taskhead门保持；R127条件归因未用于更换head基线。全部旧实验失败与正信号保留，负不扫参数/提示/额外SFT/seed/teacher周期。
所有4臂同query/update/学生G8生成预算；CE_AUX得到完整答案、全强度1.0 CE且额外TF/backward，POS_KD得到正教师条件答案，主得到正负条件。实际teacher行/token/时耗分别计费，不声称计算或信息强度完全相等。
SIPO原1/G符号裕量不适用RaPO连续奖励；本轮只用显式token符号投影。无效/未来schema/truncated回答不造负例、不丢组；全错组也不是必有teacher信号，合法率/资格率由私有保存输出独立核算。
复用唯一R125合法generic首taskSFT前缀，306历史步/4868queries/624历史VAL不当新调用或免费；本轮新2048步+131072RLresponses，校准17548/VAL22812/native80合171512新响应，0额外SFT/4TRAIN原生编码/0新VAL编码。2354是加历史前缀的完整链，非2354新更新。
只当前TRAIN、无历史replay；动态seen菜单、全臂seal后seenVAL、各臂连续Adam/sigma/teacherK10、teacher/actor/reference还原与actualL2/资格数学均核。20表行含4共享前缀历史读出=17unique其中16新，288类行/72group类行。
generation各kind未逐项保存padded长度，base TF物理response行数精确、总input-token cells给出由实际总decode cells推得的区间，不伪称全forward token精确值；所有这些算子的CPU/wall/硬预算均已实际计入。
校准只有aggregate正确数，teacher logp/逐类梯度未保存，不补造独立重评分、variance或梯度效果；组份额/teacher项非零不等泛化。所有failed attempts若发生须另记且原钟不重置。

完整saved-only CPU子进程（含torch导入）4.121732秒/wall 4.300573348999023秒；先前三次interim与本次全部保留ALL_ATTEMPT_COSTS/INTEGRITY_PULSES。

独立聚合算术CPU子进程0.054174秒/wall 0.08836793899536133秒；先前三次interim与本次全部保留ALL_ATTEMPT_COSTS/INTEGRITY_PULSES。

完整结论：主对BASE平均post macro−0.3489pp/tail−1.0146pp，classwise forgetting16.7067pp劣于BASE16.0038pp，首task head从75.1903%至31.9968%（损43.1935pp，含菜单变化）。对CE_AUX宏+6.7014pp/尾−1.2421pp，对POS_KD宏+7.2603pp/尾−0.0216pp。完整联合门失败，不能只挑对CE/KD的宏优势称成功。最终主macro16.3567/head30.5567/tail.8658/tiny0%；BASE16.8141/29.3500/3.1385/5%，其tiny仅UCgrade2-3的1/4、其余四tiny0。POS_KD阶段3的UCgrade0-1为3/5，最终0/5，阶段峰值不等持续保留。

正常退出0；原准备至退出33431.458756秒，whole worker wait4 CPU29527.183524秒（body CPU29523.639640为其子集，不相加）。本轮171512生成/2048新曲线/288类行与全部状态保存核过，4臂各512实际非零步；主正负teacher各26112科学TF行，POS_KD正教师32768，CE_AUX科学4096教师查询；不据TRAIN正确数或teacher机会推定效果。完整算术另由CLASS_RESULTS整数分母第二种实现独立核过，详见INDEPENDENT_AGGREGATE_AUDIT/PAIRED_CLASS_CONTRIBUTIONS/MECHANISM_ANALYSIS_ZH。

CPU软件4.769550秒及五个saved-only审计/分析完整child费用保留ALL_ATTEMPT_COSTS；SSH/准备shell/Git CPU未全面仪器计时，未知非0，不宣称总CPU精确。历史首task前缀306/4868/624有成本且不重复生成。所有生成prefill35774940tokens/active923990/decoded1447886cells、contextteacherTF19610631cells实记；base TF393408响应行精确、inputcells区间[83766378,86552322]，per-kind padded长度未存不补造。

后继具体准入决策见ROUTE_DECISION_ZH：关闭当前辅助配方网格，不复活R126原钟不足的同候选；继续新机制反馈/目标兼容性入口筛选，未伪造新科学冻结/native/启动，不等待用户逐轮批准。
