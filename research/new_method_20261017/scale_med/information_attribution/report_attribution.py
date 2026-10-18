"""Aggregate report only; never select a new arm after test results."""
import csv,json,sys
from pathlib import Path
p=Path(sys.argv[1]);rs=json.loads((p/'results.json').read_text());assert json.loads((p/'status.json').read_text())['state']=='COMPLETE'
lines=['# 固定信息归因比较','', '七项固定比较，首转换、历史暴露测试。主比较 S−N 和 S−K；不依据测试选择新公式或追加设置。','', '| 数据集 | 方法 | BA % | 旧类 BA % | 新类 BA % |','|---|---|---:|---:|---:|'];table=[];deltas=[];gates={}
for r in rs:
 for arm,v in r['metrics'].items():
  s=v['summary'];lines.append(f"| {r['dataset']} | {arm} | {100*s['all']['BA']:.4f} | {100*s['old']['BA']:.4f} | {100*s['new']['BA']:.4f} |")
  for group,m in s.items():table.append(dict(dataset=r['dataset'],arm=arm,group=group,**m))
lines+=['','| 数据集 | 对比 | BA差 pp | 95% CI pp | 旧类差 pp | 新类差 pp | 纠正/损害 |','|---|---|---:|---|---:|---:|---|']
for r in rs:
 gates[r['dataset']]={}
 for label,interval in r['intervals'].items():
  a,b=label.split('-');delta={g:100*(r['metrics'][a]['summary'][g]['BA']-r['metrics'][b]['summary'][g]['BA']) for g in ['all','old','new']};lo,hi=interval['all']['delta_BA_pp_CI95'];pair=r['paired'][label]['all']
  lines.append(f"| {r['dataset']} | {label} | {delta['all']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {delta['old']:+.4f} | {delta['new']:+.4f} | {pair['rescue']}/{pair['harm']} |")
  for g in delta:
   l,h=interval[g]['delta_BA_pp_CI95'];deltas.append(dict(dataset=r['dataset'],contrast=label,group=g,delta_BA_pp=delta[g],CI95_low_pp=l,CI95_high_pp=h,**r['paired'][label][g]))
  if label in ['S-N','S-K']:gates[r['dataset']][label]=dict(positive_lower_CI=lo>0,no_old_new_point_loss_over_1pp=min(delta['old'],delta['new'])>=-1)
strong=all(all(x.values()) for ds in gates.values() for x in ds.values())
lines+=['',f"预先约定的较强探索性二阶增量条件：{'满足' if strong else '不满足'}。",'', 'G0−B 改变了加权、截距、归一化与正则流程，不能单独归因某一个因素。G1−G0 检查第二全局路径，M−G1 检查均值，N−M 检查均值二次非线性，S−N 检查同方向图像内方差，S−K 检查是否超出有限的等维全局/均值非线性容量控制。', '', '256方向不能代表全部协方差信息；K不能代表所有全局非线性模型。没有检测到收益不等于证明所有局部信息无效。原测试已参与研究分析，bootstrap区间是条件性描述，不校正历史选择、多重比较，也不能证明极小HK类别未来安全。患者级隔离未知。', '', '每个新头仅通过旧类支持的内部组留出选择alpha，然后封存归一化、投影和alpha；单组类别无验证覆盖，新类不参与选参。旧类阶段结束后只带统计进入新类阶段，退休输入保护检查见RETIREMENT文件。不是对历史运行的续跑，也没有重新把历史暴露数据称为盲测。', '', '成本见costs.json：实际统计矩阵、头、归一化和投影分别记账；不含编码器参数和旧gB缓存生成成本。共享提取时间不能当作各方法独立部署成本。没有追加门控、种子、维度或公式扫描。']
for name,data in [('summary.csv',table),('contrasts.csv',deltas)]:
 with (p/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
(p/'CONCLUSION.json').write_text(json.dumps(dict(stronger_exploratory_support=strong,conditions=gates),indent=2)+'\n');(p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
