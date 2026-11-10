# R128 stage3（第4任务）保存结果中间核验

仅本stage，不判定完整四post平均/最终head/遗忘九门；不据中间reward/VAL改参。

|arm|seenmacro %|tail %|oldmacro %|newmacro %|firsttask head4 %|tiny %|
|---|---:|---:|---:|---:|---:|---:|
|BASE_TARS_RAPO|18.6817|1.1905|17.4846|23.1707|31.2777|0.0000|
|CE_AUX_TARS_RAPO|12.2629|10.2273|3.6287|44.6415|1.9929|0.0000|
|POS_KD_TARS_RAPO|13.1806|7.5000|6.2998|38.9837|14.5512|20.0000|
|CONTRAST_TARS_RAPO|18.1916|0.5952|16.6470|23.9837|30.9284|0.0000|

主相对各控制增量pp：
BASE_TARS_RAPO：macro -0.4901, tail -0.5952, old_macro -0.8376, new_macro +0.8130
CE_AUX_TARS_RAPO：macro +5.9287, tail -9.6320, old_macro +13.0184, new_macro -20.6578
POS_KD_TARS_RAPO：macro +5.0110, tail -6.9048, old_macro +10.3473, new_macro -15.0000

4臂各128attempt/8192回答；连续Adam step384、sigma接自己上一stage、K10最后同步380、当前TRAIN同tape/seen-only VAL在全4臂seal后、整数类分母与EOS负例/众数平票均独立核对。
完整类别标签是辅助监督；CE_AUX/POS_KD和主得到同标签，实际计算不同。组份额、正确TRAIN回答数、非零L2或teacher资格不等同梯度贡献/VAL效果。
oldmacro不等firsttask head4；完整类表保留tiny与所有类分母。单order开发VAL组件不是独立患者证据。
0新模型调用/生成/optimizer；私有预测/normalized rollout/ACCESS/索引/STATE仅远端CPU读取，不fetch/public。校准逐例correct/teacher原始logp/每类梯度未存，不补造。
后续所有臂继续原冻结协议，23h原钟/累计10GB/hardguard不变；未运行full finalizer。

本次CPU审计完整子进程计时（含torch导入）2.105665秒，wall 2.1932644844055176秒；0新模型调用，成本保留INTEGRITY_PULSES。

本阶段主相对BASE macro−0.4901pp、tail−0.5952pp；oldmacro−0.8376pp/newmacro+0.8130pp，尚无整体增益。主对CE_AUX/POS_KD虽macro更高，tail分别低9.6320/6.9048pp，且新类明显较低，不能挑macro称成功。POS_KD tiny20%来自UC grade0-1的3/5，其余两tiny均0/1；BASE/CE_AUX/主的三tiny全0。小分母组件VAL不是独立患者或稳定尾类改善证据。前三post均未胜BASE宏，不据此提前结束剩余阶段或改原九门。
