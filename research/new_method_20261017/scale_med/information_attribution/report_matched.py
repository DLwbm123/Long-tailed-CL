import json,csv,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'results.json').read_text());assert json.loads((p/'status.json').read_text())['state']=='COMPLETE'
lines=['# HK 固定正则归因检查','', '统一alpha=.01，取自首轮旧类支持内部选出的N参照。只从原联合统计重新求解G0/G1，不访问旧图像、不重新选参。本检查在首轮测试分析后提出，只作探索诊断。','', '| 方法 | BA % | 旧类 BA % | 新类 BA % |','|---|---:|---:|---:|'];rows=[]
for a,m in r['metrics'].items():
 s=m['summary'];lines.append(f"| {a} | {100*s['all']['BA']:.4f} | {100*s['old']['BA']:.4f} | {100*s['new']['BA']:.4f} |")
 for g,v in s.items():rows.append(dict(arm=a,group=g,**v))
lines+=['','| 对比 | 总体变化 pp | 95% CI pp | 旧类变化 pp | 新类变化 pp | 纠正/损害 |','|---|---:|---|---:|---:|---|']
for label,interval in r['intervals'].items():
 a,b=label.split('-');d={g:100*(r['metrics'][a]['summary'][g]['BA']-r['metrics'][b]['summary'][g]['BA']) for g in ['all','old','new']};lo,hi=interval['all']['delta_BA_pp_CI95'];c=r['paired'][label]['all'];lines.append(f"| {label} | {d['all']:+.4f} | [{lo:+.4f},{hi:+.4f}] | {d['old']:+.4f} | {d['new']:+.4f} | {c['rescue']}/{c['harm']} |")
lines+=['','原G0/G1分数及预测在重提取CLS后通过固定容差复现。区间为精确图像身份组内配对bootstrap10000次seed74003，非同时区间；HK患者隔离未知，极小类别风险及历史测试暴露限制保留。','', '本检查不改变首轮S−N/S−K的结果，不追加alpha、特征维度、投影或门控。不能把重新求解后效果较好的配置事后改成主方法。成本仅包含本次统计求解与HK测试路径提取，不含历史支持统计构建。']
(p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
with (p/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
