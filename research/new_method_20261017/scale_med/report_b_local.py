"""Aggregate-only report; per-row diagnostics remain private."""
import csv,json,sys
from pathlib import Path

root=Path(sys.argv[1]);records=json.loads((root/'results.json').read_text());gates={};summary=[];classes=[]
text=['# Frozen B + local shrink: fixed lambda 0.5','',
      'First transition, reused development queries. Frozen B and local_shrink references reproduce exactly. No fitting, image inference, lambda search or E/test access. Both support-derived scales sealed before query scores opened.','',
      '| Dataset | Query | B BA % | Local BA % | Fusion BA % | All delta pp | Old delta pp | New delta pp | Rescue | Harm |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in records:
    n,q=r['dataset'],r['query'];m=r['metrics'];b=m['B']['summary'];f=m['B_local']['summary']
    delta={g:100*(f[g]['BA']-b[g]['BA']) for g in ['all','old','new']};p=r['paired']['all']
    passed=delta['all']>0 and min(delta['old'],delta['new'])>=-1
    gates.setdefault(n,{})[q]=dict(delta_pp=delta,pass_condition=passed)
    text.append(f"| {n} | {q} | {100*b['all']['BA']:.4f} | {100*m['local_shrink']['summary']['all']['BA']:.4f} | {100*f['all']['BA']:.4f} | {delta['all']:+.4f} | {delta['old']:+.4f} | {delta['new']:+.4f} | {p['rescue']} | {p['harm']} |")
    for arm,v in m.items():
        for group,s in v['summary'].items():summary.append(dict(dataset=n,query=q,arm=arm,group=group,**s))
    for c in r['classes']:
        classes.append(dict(dataset=n,query=q,**c,BA_contribution_pp=100*(c['rescue']-c['harm'])/(c['count']*len(r['classes']))))
for n,g in gates.items():g['both_queries_pass']=all(v['pass_condition'] for v in g.values())
text+=['','## Frozen descriptive criterion','']+[f"- {n}: {'PASS' if g['both_queries_pass'] else 'FAIL'} (both queries must improve BA; old/new loss cannot exceed 1 pp)." for n,g in gates.items()]
text+=['','## Disagreement diagnostics','',
       '| Dataset | Query | Branch disagreements | B only correct | Local only correct | Both wrong | Fusion changes |',
       '|---|---|---:|---:|---:|---:|---:|']
for r in records:
    sums={k:sum(c[k] for c in r['classes']) for k in ['branch_disagreement','B_only_correct','local_only_correct','both_wrong']}
    text.append('| '+ ' | '.join(map(str,[r['dataset'],r['query'],*sums.values(),r['paired']['all']['changed']]))+' |')
text+=['','## Interpretation and limits','',
       'Per-class rescue/harm and BA contributions are in classes.csv. results.json includes full confusion matrices and margin quantiles for branch disagreements, fusion changes, rescue and harm. Top2 gaps measure confidence, true margins measure correctness, and B_vs_local margins measure the contest between the two branch winners. All public margin quantiles use normalized branch scores; local margins exclude lambda, fusion margins include it. Per-row margins, labels, predictions and row indices remain private on the approved server. Calibration uses fitted-support scores and all seen model columns; it is not held-out calibration or proof that a fixed scale generalizes. CE depends on score scale. This full-history B diagnostic does not validate compressed-memory deployment. No independent test or multi-seed claim.']
for name,rows in [('summary.csv',summary),('classes.csv',classes)]:
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
(root/'gates.json').write_text(json.dumps(gates,indent=2)+'\n')
(root/'REPORT.md').write_text('\n'.join(text)+'\n')
