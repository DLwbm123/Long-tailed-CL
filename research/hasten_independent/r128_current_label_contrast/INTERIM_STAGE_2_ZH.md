# R128 stage2（第3任务）保存结果中间核验

仅本stage，不判定完整四post平均/最终head/遗忘九门；不据中间reward/VAL改参。

|arm|seenmacro %|tail %|oldmacro %|newmacro %|firsttask head4 %|tiny %|
|---|---:|---:|---:|---:|---:|---:|
|BASE_TARS_RAPO|22.0315|1.9048|20.9011|24.2922|35.7489|0.0000|
|CE_AUX_TARS_RAPO|16.7885|0.4878|5.1886|39.9884|12.9714|0.0000|
|POS_KD_TARS_RAPO|16.9008|1.9048|11.4706|27.7612|25.4787|0.0000|
|CONTRAST_TARS_RAPO|21.8438|1.9048|20.1043|25.3227|35.4227|0.0000|

主相对各控制增量pp：
BASE_TARS_RAPO：macro -0.1877, tail +0.0000, old_macro -0.7968, new_macro +1.0305
CE_AUX_TARS_RAPO：macro +5.0553, tail +1.4170, old_macro +14.9157, new_macro -14.6657
POS_KD_TARS_RAPO：macro +4.9429, tail +0.0000, old_macro +8.6337, new_macro -2.4385

4臂各128attempt/8192回答；连续Adam step256、sigma接自己上一stage、K10最后同步250、当前TRAIN同tape/seen-only VAL在全4臂seal后、整数类分母与EOS负例/众数平票均独立核对。
完整类别标签是辅助监督；CE_AUX/POS_KD和主得到同标签，实际计算不同。组份额、正确TRAIN回答数、非零L2或teacher资格不等同梯度贡献/VAL效果。
oldmacro不等firsttask head4；完整类表保留tiny与所有类分母。单order开发VAL组件不是独立患者证据。
0新模型调用/生成/optimizer；私有预测/normalized rollout/ACCESS/索引/STATE仅远端CPU读取，不fetch/public。校准逐例correct/teacher原始logp/每类梯度未存，不补造。
后续所有臂继续原冻结协议，23h原钟/累计10GB/hardguard不变；未运行full finalizer。

本次CPU审计完整子进程计时（含torch导入）2.666331秒，wall 3.658371925354004秒；0新模型调用，成本保留INTEGRITY_PULSES。

本阶段主对BASE macro−0.1877pp、tail持平；oldmacro−0.7968pp/newmacro+1.0305pp，未出现整体宏均值提升。对CE_AUX宏尾更高、对POS_KD宏更高而尾持平，同时newmacro均更低；不能据此宣称主全面获益或替换原九门。前两个post均未胜BASE macro不是完整四post最终失败判定。全部tiny为0，类分母见CSV。
