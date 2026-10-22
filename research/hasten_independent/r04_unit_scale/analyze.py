import json,statistics
from pathlib import Path
root=Path(__file__).parent;data=json.loads((root/'results/aggregate.json').read_text());rows=data['results'];arms=['H-raw','H-unit','H-unit-no-proj'];seeds=[74002,74003,74004]
summary={}
for a in arms:
 out=[]
 for seed in seeds:
  rr=sorted([r for r in rows if r['arm']==a and r['seed']==seed],key=lambda v:v['stage']);m=[r['metrics']['final']['BA']*100 for r in rr]
  out.append(dict(seed=seed,mean=statistics.mean(m),final=m[-1],stages=m,final_correct=sum(rr[-1]['metrics']['final']['correct']),tail=rr[-1]['metrics']['final']['tail_BA']*100,perclass_correct=rr[-1]['metrics']['final']['correct']))
 summary[a]=dict(seeds=out,mean=statistics.mean(v['mean'] for v in out),final=statistics.mean(v['final'] for v in out),tail=statistics.mean(v['tail'] for v in out))
diffs=[dict(seed=s,mean=summary['H-unit']['seeds'][i]['mean']-summary['H-raw']['seeds'][i]['mean'],final=summary['H-unit']['seeds'][i]['final']-summary['H-raw']['seeds'][i]['final']) for i,s in enumerate(seeds)]
gate=statistics.mean(v['mean'] for v in diffs)>=1 and statistics.mean(v['final'] for v in diffs)>=1 and sum(v['mean']>0 and v['final']>0 for v in diffs)>=2
(root/'results/ANALYSIS.json').write_text(json.dumps(dict(summary=summary,paired_unit_minus_raw=diffs,preregistered_development_gate_pass=gate,independent_confirmation=False),indent=2))
lines=['# r04：输入尺度修正的三种子配对结果','',f'预注册开发门通过：{gate}。这不是独立测试或统计显著性结论。','', '| 路线 | 阶段平均BA | 最终BA | 尾7 BA |','|---|---:|---:|---:|']
for a,v in summary.items():lines.append(f"| {a} | {v['mean']:.4f}% | {v['final']:.4f}% | {v['tail']:.4f}% |")
lines+=['| Spatial（旧单seed） | 33.2341% | 28.0000% | 23.8095% |','| CLS＋patch（旧单seed） | 34.0053% | 27.3333% | 19.0476% |','','| seed | raw平均/最终 | unit平均/最终 | unit-no-proj平均/最终 |','|---|---:|---:|---:|']
for i,s in enumerate(seeds):
 vals=[summary[a]['seeds'][i] for a in arms];lines.append(f'| {s} | '+' | '.join(f"{v['mean']:.4f}/{v['final']:.4f}" for v in vals)+' |')
lines+=['','三个种子的unit−raw平均与最终均为正，平均提升3.9425pp、最终提升4.3333pp，满足固定双指标>=1pp、至少2/3种子同向门。单位化全部新阶段训练诊断距离上限饱和率0；raw seed74003全部为1，seed74004前两阶段0、之后1，保留该差异。输入尺度修正恢复双曲分支区分能力，低损失本身不作为科学门。','', 'unit相对unit-no-proj平均+0.3413pp、最终+1.3333pp，三种子两指标方向一致；阶段平均增幅不到1pp，不夸大为强机制优势。unit最终每seed正确77/79/81张，无投影72/76/77张。','', 'unit仍低于Spatial平均0.8792pp、最终1.6667pp；低于CLS＋patch平均1.6504pp、最终1.0000pp。尾7只有12.0635%，弱于旧两条路线；内部改善不是全面胜出。所有逐类正确数和各seed阶段指标见ANALYSIS.json，不能隐藏尾类损害。','', '54阶段中6个raw seed74002阶段只读复用r03，实际新训练48阶段、85360更新；总轨迹更新96030含复用，不把它计作本轮新算力。首次运行3375.61秒（56分16秒），exit0，新增计费4158486803字节。54阶段所有三种分数独立重算计数及恢复检查通过。r03+r04累计约7.03GB，在30GB窗口预算内。','', '下一轮固定均衡当前类的优化采样，保持训练批次数/epoch不增加，原型和回放统计仍使用全部原训练数据。用三种子配对检验尾类与整体收益，继承unit尺度，保留本轮结果。只做一个明确假设，不搜索温度/融合/树，不优化旧基线。','', '验证集已反复开发，无独立患者测试结论。旧路线骨干/训练不同且旧结果为单seed，仅作路线数值比较；当前三seed不能消除这些混杂。']
(root/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({a:{k:v[k] for k in ['mean','final','tail']} for a,v in summary.items()}))
