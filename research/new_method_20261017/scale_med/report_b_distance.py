import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]);rs=json.loads((root/'results.json').read_text());summary=[];classes=[];gates={}
text=['# Uniform volume removal with frozen B and scales','',
      'Lambda 0.5. No parameter scan, scale refit, model fitting, image forwards or E/test access. Previous B and fusion metrics reproduce exactly. All class offsets were sealed before opening query scores.','',
      '| Dataset | Query | B BA % | Previous fusion % | Current fusion % | Delta vs B pp | Delta vs previous pp | Rescue/harm vs B | Rescue/harm vs previous |',
      '|---|---|---:|---:|---:|---:|---:|---|---|']
for r in rs:
    n,q=r['dataset'],r['query'];m=r['metrics'];current=m['B_distance']['summary'];b=m['B']['summary'];p=m['previous_fusion']['summary']
    ds={base:{g:100*(current[g]['BA']-m[base]['summary'][g]['BA']) for g in ['all','old','new']} for base in ['B','previous_fusion']}
    gates.setdefault(n,{})[q]={base:dict(delta_pp=d,pass_condition=d['all']>0 and min(d['old'],d['new'])>=-1) for base,d in ds.items()}
    pb,pp=r['paired_B']['all'],r['paired_previous']['all']
    text.append(f"| {n} | {q} | {100*b['all']['BA']:.4f} | {100*p['all']['BA']:.4f} | {100*current['all']['BA']:.4f} | {ds['B']['all']:+.4f} | {ds['previous_fusion']['all']:+.4f} | {pb['rescue']}/{pb['harm']} | {pp['rescue']}/{pp['harm']} |")
    for arm,v in m.items():
        for group,s in v['summary'].items():summary.append(dict(dataset=n,query=q,arm=arm,group=group,**s))
    for base,key in [('B','classes'),('previous_fusion','classes_previous')]:
        for c in r[key]:classes.append(dict(dataset=n,query=q,reference=base,**c))
text+=['','## Gates','']
for n,g in gates.items():
    for base in ['B','previous_fusion']:
        passed=all(g[q][base]['pass_condition'] for q in ['Qfit','Qgate'])
        text.append(f"- {n} vs {base}: {'PASS' if passed else 'FAIL'} (both queries improve overall BA, neither old/new loses more than 1 pp).")
text+=['','## Targeted HK audit','']
for r in rs:
    for t in r.get('target_audit',[]):text.append(f"- Mapped class {t['class_index']}: B correct, previous fusion wrong; current fusion {'correct (recovered)' if t['current_correct'] else 'wrong'}. No row identity published.")
text+=['','## Interpretation','',
'''Both targeted HK errors recover under the uniform, class-independent formula. HK Qgate rises 9.0179 pp vs previous fusion and 0.4801 pp vs B. This supports the volume-offset mechanism for these two cases; it is a post-hoc diagnostic, not independent confirmation. Removing the term does not merely preserve prior decisions: vs previous fusion HK Qgate rescues four and harms two (one previous rescue is lost and one previously correct B prediction becomes wrong), while vs B the new result rescues three and harms one. The new harm is in mapped class6 (73 query images), not either targeted tiny class. HK Qfit also introduces one harm in class6; its rescue/harm vs B changes from 5/0 to 6/1.

ISIC overall BA increases on both queries, but harm counts versus B worsen: Qfit 5 to 14 and Qgate 7 to 11. Rescue counts rise from 24 to 33 and 21 to 35. The new-class group loses 2.3810 and 1.3937 pp relative to previous fusion, so the old/new protection criterion fails against that previous fusion despite higher overall BA. The same group remains at B on Qfit and +0.1742 pp on Qgate. Class4 accounts for 12 harms on Qfit and 11 on Qgate; it has 12 rescues on each. Qfit class3 gains 17 net correct images and Qgate class3 gains 20. Thus the modification redistributes benefits toward old classes and is not a uniformly safer fusion.

Distance-only local BA is lower than the original density-only local BA in all four conditions: ISIC 49.7627/48.7202 vs 60.5042/57.1859, HK 52.0751/57.3252 vs 65.0962/63.1727. The volume term carries useful discrimination as well as harmful bias; wholesale removal improves this frozen fusion's overall BA but does not establish a superior standalone local classifier. Broader-covariance preferences and the retained original scale remain limitations. No further tuning or experiment was performed.

Conclusion: both datasets pass the frozen criterion versus B on these reused development queries. Both HK target samples recover. Overall BA improves in all four conditions vs previous fusion, but ISIC harm counts and its new-class BA worsen. Do not claim universal non-degradation, independent test performance, or deployment readiness.''',
      '', '## Diagnostics and cost','',
      'classes.csv contains per-class rescue/harm against B and previous fusion. For classes_previous, the field B_correct denotes the reference previous fusion; local_correct denotes distance-only local. results.json includes confusion matrices, old/new metrics, paired outcomes and normalized margin quantiles. Per-row disagreements, labels, scores and indices remain private. The synthetic test verifies exact Gaussian-volume cancellation against the direct Mahalanobis expression with regularization. CPU runner wall time is recorded in status.json and excludes prior cache generation. Source reference: frozen B-local delivery 13b28296482e8f6cd1b8b65d395d87d76379ce22.']
for name,rows in [('summary.csv',summary),('classes.csv',classes)]:
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(root/'gates.json').write_text(json.dumps(gates,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(text)+'\n')
