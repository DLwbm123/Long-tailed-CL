import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]);rs=json.loads((root/'results.json').read_text());summary=[];classes=[];retained=[];gates={}
text=['# Conservative common-support fusion','',
      'Frozen B, original scales, lambda=0.5. No parameter scan, scale fit, model fit, image forward or E/test access. Both prior fusions and B reproduce exactly.','',
      '| Dataset | Query | Consensus BA % | Delta B all pp | Delta B old pp | Delta B new pp | Rescue/harm | Harm bound |',
      '|---|---|---:|---:|---:|---:|---|---:|']
for r in rs:
    n,q=r['dataset'],r['query'];m=r['metrics'];s=m['consensus']['summary'];b=m['B']['summary'];delta={g:100*(s[g]['BA']-b[g]['BA']) for g in ['all','old','new']};p=r['paired']['B']['all'];bound=r['retention']['all']['harm_bound']
    text.append(f"| {n} | {q} | {100*s['all']['BA']:.4f} | {delta['all']:+.4f} | {delta['old']:+.4f} | {delta['new']:+.4f} | {p['rescue']}/{p['harm']} | {bound['upper_bound']} |")
    gates.setdefault(n,{})[q]={}
    for base in ['B','density_fusion','distance_fusion']:
        d={g:100*(s[g]['BA']-m[base]['summary'][g]['BA']) for g in ['all','old','new']}
        gates[n][q][base]=dict(delta_pp=d,pass_condition=d['all']>0 and min(d['old'],d['new'])>=-1)
    for arm,v in m.items():
        for group,z in v['summary'].items():summary.append(dict(dataset=n,query=q,arm=arm,group=group,**z))
    classes.extend(dict(dataset=n,query=q,**c) for c in r['classes'])
    for group,v in r['retention'].items():
        for base in ['density_fusion','distance_fusion']:retained.append(dict(dataset=n,query=q,group=group,reference=base,**v[base]))
text+=['','## Rescue retention (same samples, relative to B)','',
      '| Dataset | Query | Density retained/total | Density lost | Distance retained/total | Distance lost |',
      '|---|---|---|---:|---|---:|']
for r in rs:
    a=r['retention']['all']['density_fusion'];b=r['retention']['all']['distance_fusion']
    text.append(f"| {r['dataset']} | {r['query']} | {a['retained']}/{a['reference_rescues']} | {a['lost']} | {b['retained']}/{b['reference_rescues']} | {b['lost']} |")
text+=['','ISIC Qfit additionally rescues one sample absent from each respective reference rescue set. Other conditions add no rescues outside the respective sets. Counts of total rescues therefore must not be mistaken for retention counts.','', '## Gates','']
for n,g in gates.items():
    for base in ['B','density_fusion','distance_fusion']:
        passed=all(g[q][base]['pass_condition'] for q in ['Qfit','Qgate'])
        text.append(f"- {n} versus {base}: {'PASS' if passed else 'FAIL'} on both-query overall BA and old/new protection criterion.")
text+=['','## Analysis','',
'''The harm subset invariant holds per row and separately for old/new groups. Observed harms equal the all-group bounds: ISIC 4/2, HK 0/0. Both targeted HK Qgate samples (mapped classes12 and13) remain correct. Zero harm here is a property of the observed reference-harm intersection, not a guarantee on future data. The formula preserves B when either prior fusion preserves B, but cannot guarantee that a changed prediction is correct.

Relative to B, ISIC improves overall/old/new BA on both queries. HK improves overall/old BA while new BA is unchanged. Both datasets pass the descriptive B-reference criterion. However, consensus sacrifices many corrections: only 17/24 and 14/21 density rescues, and 17/33 and 14/35 distance rescues survive on ISIC. HK retains 2/5 and 1/2 density rescues, and 2/6 and 1/3 distance rescues. HK's zero harm comes with small positive BA gains of 0.4048/0.2381 pp, not preservation of all previous gains.

ISIC class4 harm falls to 2/2 versus B (distance fusion had 12/11), but rescues fall from 12/12 to 9/7. New-class BA lies between the density and distance versions: versus density -1.1905/-0.6969 pp, versus distance +1.1905/+0.6969 pp. Consensus overall BA is below density by 0.2896/0.4835 pp and below distance by 0.5895/2.1804 pp. Thus the conservative rule reduces damage but gives up useful decisions; it is not a uniformly superior method.

HK Qgate improves 8.7759 pp versus density by avoiding its rare-class failures, but remains 0.2420 pp below distance. HK Qfit is 0.4607/0.7896 pp below density/distance. Against either prior fusion, neither dataset passes the both-query overall-improvement criterion.

This is a post-hoc mechanism diagnostic on reused development queries. There is no independent validation, confidence interval or multi-seed evidence. The two local variants are correlated views of the same model, not independent evidence. Do not tune thresholds or select a per-dataset rule from these observations. No subsequent experiments were run.''',
      '', '## Validation and delivery','',
      'Synthetic checks cover sign disagreement, shared positive support, exact ties retaining B, rescue accounting and randomized label-free invariant checks. Live assertions verify identical labels/B/scales, previous full metrics, per-row invariant and both target recoveries. CE is descriptive for anchored rectified scores; it is not evidence of calibration. Margin diagnostics local refers to distance-only local. Public files are aggregate-only; per-row labels, indices, predictions and scores stay private. Cached CPU runtime is in status.json; prior cache construction is excluded. References: density delivery 13b28296482e8f6cd1b8b65d395d87d76379ce22; distance delivery 8aa53c92e012b72a635c805466f51b6bed459035.']
for name,rows in [('summary.csv',summary),('classes.csv',classes),('retention.csv',retained)]:
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(root/'gates.json').write_text(json.dumps(gates,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(text)+'\n')
