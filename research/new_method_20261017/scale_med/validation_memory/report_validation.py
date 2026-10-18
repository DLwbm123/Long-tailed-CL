"""Render aggregate evidence; private artifacts are never copied into delivery."""
import json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
result=json.loads((root/'results.json').read_text());exit_info=json.loads((root/'EXIT.json').read_text());assert exit_info['exit_code']==0
costs=json.loads((root/'costs.json').read_text())
for phase in ['fit','audit','test']:costs[phase]=json.loads((root/(phase+'.costs.json')).read_text())
prior_failures=json.loads((root/'PRIOR_FAILURES.json').read_text()) if (root/'PRIOR_FAILURES.json').exists() else []
public=dict(results=result,costs=costs,exit=exit_info,prior_failures=prior_failures,cumulative_wall_seconds=exit_info['wall_seconds']+sum(f['wall_seconds'] for f in prior_failures),data_audit=json.loads((root/'DATA_AUDIT.json').read_text()),model_seal=json.loads((root/'MODEL_SEAL.json').read_text()),audit_seal=json.loads((root/'AUDIT_SEAL.json').read_text()))
(out/'aggregate.json').write_text(json.dumps(public,indent=2,allow_nan=False))
lines=['# 历史验证记忆：最小证伪结果','',f"退出码：0；成功轮墙钟：{exit_info['wall_seconds']:.2f} 秒；含失败累计 {public['cumulative_wall_seconds']:.2f} 秒。主检验 ISIC；HK 验证覆盖 14/15 类，只作辅助诊断。",'', '首次启动因路径字符串与Path混用，退休拦截未生效，被自检及时停止；保留失败成本并修正，在独立目录重跑，科学协议未改变。' if prior_failures else '', '', '旧测试已经暴露、患者隔离未知；这不是盲测，也不是隐私保证。所有选择器永久排除验证行，未进行选参后全量重拟合。','']
for n,res in result.items():
    v=res['validation'];lines += [f'## {n}', '',f"停止判断：`{res['decision']}`。完整验证候选 BA 跨度 {v['validation_BA_spread_pp']:.4f} pp；精确矩与直接 MSE 最大误差 {v['quadratic_max_abs_error']:.3g}。",'', '| 选择器 | alpha | 验证 BA % | 选择遗憾 pp | 测试 BA % | 旧类 BA % | 新类 BA % |','|---|---:|---:|---:|---:|---:|---:|']
    for k,a in v['selection'].items():
        m=res['metrics'][k]['summary'];lines.append(f"| {k} | {a:g} | {100*v['validation_BA'][str(a)]:.4f} | {v['regret_pp'][k]:.4f} | {100*m['all']['BA']:.4f} | {100*m['old']['BA']:.4f} | {100*m['new']['BA']:.4f} |")
    ref=v['strongest_simple_control'];ci=res['intervals'][ref]['MOMENT']['all']['delta_BA_pp_CI95'];delta=100*(res['metrics']['MOMENT']['summary']['all']['BA']-res['metrics'][ref]['summary']['all']['BA'])
    lines += ['',f"在测试打开前，由完整验证选出的最强简单控制为 {ref}（特权诊断参照）。MOMENT−该控制：{delta:+.4f} pp，配对95%区间 [{ci[0]:+.4f}, {ci[1]:+.4f}]。",'']
    c=costs[n];lines += [f"验证矩实际稠密状态 {c['validation_dense_bytes']/2**20:.3f} MiB；上三角理论存储 {c['validation_packed_symmetric_bytes']/2**20:.3f} MiB；同批验证特征 float64 等价 {c['raw_validation_equivalent_float64_bytes']/2**20:.3f} MiB。后者仅用于成本参照，不能作为合规部署记忆。",'']
decision=result['ISIC']['decision']
meaning={'STOP_NO_SELECTION_HEADROOM':'完整历史验证没有比最强简单选择器提供超过0.5pp的验证选择空间。当前候选族下，新增记忆的必要性未成立；不能据此宣称所有在线选参均无效。','STOP_SURROGATE_MISMATCH':'存在简单选择器未恢复的验证选择空间，但精确平方损失摘要未通过BA选择遗憾门。失败不能归因于压缩；本轮根本没有压缩。','STOP_TRANSFER_UNCERTAIN':'验证摘要通过了前两门，但独立评测的预设收益与旧新类约束未同时满足。不能把验证排序改善写成泛化收益。','EXPLORATORY_GATES_PASS_NO_AUTOMATIC_EXTENSION':'预设探索性门均通过，但历史测试暴露和单次转换限制仍在，需要独立确认；本授权不自动启动压缩或扩展实验。'}
lines += ['## 机制解释与边界','',meaning[decision],'','FIRST 检查首阶段固定参数是否已经足够；CURRENT 检查只看新类的选择；BIRTH 检查旧分数归档；FULL_BA 检查历史验证的选择空间；FULL_MSE 与 MOMENT 分开识别替代指标误差和统计实现误差。FULL_BA/FULL_MSE 由隔离审计进程计算，不能作为满足旧逐样本退出约束的部署方法。','','平方损失精确可重算是标准统计恒等式，工程一致性不构成方法成功。出生归档仅是相同候选头下的机制控制，不是SLARC复现。HK 缺失类仍作为所有头的输出竞争者，但不进入验证均值，不能据其14类结果声称15类BA选参可靠。','','没有运行压缩、额外种子、网格扩展、表示修改或后续实验。缓存生成和编码器历史成本不含在本轮时间内。']
(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
