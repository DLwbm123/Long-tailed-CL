import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]);rs=json.loads((root/'results.json').read_text());summary=[];classes=[];retained=[];gates={}
text=['# Decision-entry veto and equal-mean fusion','',
      'Frozen B, original scales and lambda=0.5. Equal mean only when both prior fusions leave B. No parameter scan, model/image processing, scale refit or E/test access.','',
      '| Dataset | Query | Entry BA % | Delta B all/old/new pp | Delta consensus all pp | Rescue/harm vs B | Rescue/harm vs consensus | Harm bound |',
      '|---|---|---:|---|---:|---|---|---:|']
for r in rs:
    n,q=r['dataset'],r['query'];m=r['metrics'];s=m['entry']['summary'];ds={base:{g:100*(s[g]['BA']-m[base]['summary'][g]['BA']) for g in ['all','old','new']} for base in ['B','density_fusion','distance_fusion','consensus']}
    pb,pc=r['paired']['B']['all'],r['paired']['consensus']['all'];bound=r['retention']['all']['harm_bound']
    text.append(f"| {n} | {q} | {100*s['all']['BA']:.4f} | "+' / '.join(f"{ds['B'][g]:+.4f}" for g in ['all','old','new'])+f" | {ds['consensus']['all']:+.4f} | {pb['rescue']}/{pb['harm']} | {pc['rescue']}/{pc['harm']} | {bound['upper_bound']} |")
    gates.setdefault(n,{})[q]={base:dict(delta_pp=d,pass_condition=d['all']>0 and min(d['old'],d['new'])>=-1) for base,d in ds.items()}
    for arm,v in m.items():
        for group,z in v['summary'].items():summary.append(dict(dataset=n,query=q,arm=arm,group=group,**z))
    classes.extend(dict(dataset=n,query=q,**c) for c in r['classes'])
    for group,v in r['retention'].items():
        for base in ['density_fusion','distance_fusion','consensus']:retained.append(dict(dataset=n,query=q,group=group,reference=base,**v[base]))
text+=['','## Retention of consensus rescues','',
      '| Dataset | Query | Retained/previous rescues | Lost | Additional |','|---|---|---|---:|---:|']
for r in rs:
    z=r['retention']['all']['consensus'];text.append(f"| {r['dataset']} | {r['query']} | {z['retained']}/{z['reference_rescues']} | {z['lost']} | {z['additional_rescues']} |")
text+=['','## Target audit and interpretation','',
'''Both preidentified ISIC Qfit targets recover: mapped class3 and mapped class4. However, a different mapped class4 sample loses a previous rescue. Only three predictions change across all four conditions: two wrong-to-correct and one correct-to-wrong versus consensus, all on ISIC Qfit. All three were wrong under B. Thus B-relative rescue increases from18 to19 while B-relative harm remains4. The target class4 gain is canceled by the other class4 loss; the net BA gain is exactly one extra class3 correct image, 100/(311*6)=0.05359 pp. Old BA rises0.08039 pp, new BA is unchanged. No other condition changes predictions, and the HK target recoveries persist because all HK predictions remain identical to consensus.

The score evidence explains both benefits and cost. For the recovered class3 sample, wrong class4's advantage over true class3 is +0.14096 under density and -0.15156 under distance; averaging selects true class3 by only0.00530. For the recovered class4 target, wrong class0's advantage is -0.09600/+0.01561; averaging selects class4 by0.04020. For the newly lost class4 rescue, wrong class2's advantage is -0.05479/+0.08012; averaging instead selects class2 by0.01266. These are small compromise margins, not evidence that equal averaging identifies the reliable branch. Per-row identities remain private.

The label-free gate invariant and harm subset pass on every row and old/new subgroup. Observed B harms are ISIC4/2 and HK0/0, matching the fixed upper bounds. This guarantee controls damage relative to B, not loss of rescues relative to another fusion; the newly lost rescue does not violate it. Current zero HK harm is dataset-specific, not a universal generalization guarantee.

Both datasets still improve overall BA versus B on both queries without old/new degradation. However, this change barely increases retained rescue coverage: ISIC Qfit keeps17/18 consensus rescues and adds2; all other conditions retain exactly the same rescues. Versus density/distance it remains a conservative tradeoff, not a broadly stronger method. No lambda, threshold, class-specific protection, seed expansion or follow-up experiment was performed. Reused development queries and post-hoc design preclude independent performance claims.''',
      '', '## Gates','']
for n,g in gates.items():
    for base in ['B','density_fusion','distance_fusion','consensus']:
        passed=all(g[q][base]['pass_condition'] for q in ['Qfit','Qgate']);text.append(f"- {n} versus {base}: {'PASS' if passed else 'FAIL'} (strict overall improvement on both queries and old/new loss at most1 pp).")
text+=['','## Validation and artifacts','',
      'Synthetic checks cover either-branch veto, averaging across all classes, permitted return to B and the harm subset. All three reference runs reproduce full metric dictionaries, labels, B and scales. Public files include per-class comparisons, all/old/new metrics, confusion matrices, rescue retention and harm bounds. Private changes retain row indices, labels, predictions and score gaps. Hybrid-score CE is descriptive and can change without prediction changes. CPU runtime in status.json excludes all prior cache construction. Previous consensus delivery:61ea835df655feda64f40206f4326e44ce93f21f.']
for name,rows in [('summary.csv',summary),('classes.csv',classes),('retention.csv',retained)]:
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(root/'gates.json').write_text(json.dumps(gates,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(text)+'\n')
