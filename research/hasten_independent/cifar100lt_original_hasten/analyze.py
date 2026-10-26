"""Saved-output verification and same-task comparison; no model scoring."""
import json,math
from pathlib import Path
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_text())
result=read('RESULTS.json');records=read('STAGE_RESULTS.json');spec=read('SPLIT_SPEC.json');control=read('CONTROL_EXIT.json')
assert control['exit_code']==0 and result['complete'] and len(records)==6
assert control['elapsed_seconds']<24*3600 and control['peak_allocated_bytes']<4_000_000_000
counts=spec['train_counts_in_incremental_order'];start=0;expected_steps=0
for i,rec in enumerate(records):
    seen=50+i*10
    expected=10*math.ceil(sum(counts[start:seen])/64);expected_steps+=expected;start=seen
    assert rec['stage']==i and rec['seen']==seen and rec['optimizer_steps']==expected
    assert rec['total_per_class']==[100]*seen
    assert sum(rec['correct_per_class'])*100/(100*seen)==rec['accuracy']
assert result['optimizer_steps']==expected_steps
assert abs(result['average']-sum(x['accuracy'] for x in records)/6)<1e-10
lines=['# HASTEN / APART：同一 CIFAR100-LT 任务结果','',
       'B50+5×10，seed1993，训练10847行，不平衡比100，平衡官方测试集。HASTEN保留官方机制与配置；语义树按用户授权由当前模型依官方提示逐阶段生成并通过官方校验，构树模型与论文不同。',
       '', '|方法|阶段平均准确率%|最终准确率%|', '|---|---:|---:|',
       f"|APART历史完整实测|{result['APART_average']:.4f}|{result['APART_final']:.2f}|",
       f"|本次HASTEN|{result['average']:.4f}|{result['final']:.2f}|",
       f"|HASTEN−APART (pp)|{result['average_delta_pp']:+.4f}|{result['final_delta_pp']:+.2f}|",'',
       '|阶段|HASTEN%|APART历史%|差值pp|','|---|---:|---:|---:|']
for i,(x,y) in enumerate(zip(result['curve'],result['APART_historical_curve'])):lines.append(f'|{i}|{x:.2f}|{y:.2f}|{x-y:+.2f}|')
lines+=['','两者标准预训练骨干、训练配方和预算不同：HASTEN为LAION CLIP，APART为ImageNet AugReg ViT。结论限于该任务系统性能，不是机制单因素消融。仅1seed，不声称统计显著性。APART历史逐例输出/身份快照缺失，不能给出逐例配对检验；本次固定相同的确定性抽样规则并保存HASTEN原行索引。没有调参或按测试选择epoch。',
        '', '官方发布的初始树与代码类别顺序不匹配：缺26个当前类并含26个未来类。本次未使用这套不匹配树，不把原始发布物问题静默隐藏。原方法代码未改；抽样/资产绑定/资源与记录在外部包装器内。环境版本见ENVIRONMENT.json，与论文环境差异保留。',
        '',f"实际优化更新{expected_steps}，总历时{control['elapsed_seconds']:.1f}s，源端峰值{control['peak_allocated_bytes']}bytes。R64配置与进程未改动。"]
(r/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
(r/'VERIFICATION.json').write_text(json.dumps(dict(passed=True,stages=6,optimizer_steps=expected_steps,saved_outputs_only=True,paired_APART_predictions_available=False),indent=2))
print(json.dumps(result,indent=2))
