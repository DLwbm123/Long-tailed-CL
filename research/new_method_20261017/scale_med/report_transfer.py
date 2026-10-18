"""Aggregate-only transfer reporting after all four query conditions finish."""
import csv,json,math,sys
from pathlib import Path

def main(root):
    state=json.loads((root/'status.private.json').read_text());data=json.loads((root/'results.json').read_text())
    assert state['state']=='COMPLETE' and len(data)==4
    assert {(d['dataset'],d['query']) for d in data}=={(n,q) for n in ['ISIC','HK'] for q in ['Qfit','Qgate']}
    lines=['# V2 ISIC / HyperKvasir matched-query comparison','',
           'First transition only: ISIC 4+2 and HyperKvasir 13+2 classes. Exact historical S_old/S_new and Qfit/Qgate rows; no E/test access. '
           'Fixed v2 parameters and base-support scale fitting; both models sealed before either query was opened. '
           'Four internal arms plus historical frozen A/B; A/B BA and CE must reproduce published results. '
           'All observations are reused-development evidence, not independent confirmation.','',
           '| Dataset | Query | Arm | All BA % | Old BA % | New BA % | Micro % | Macro CE |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    rows=[];pairs=[]
    for d in data:
        assert len(d['metrics'])==6
        for arm,m in d['metrics'].items():
            s=m['summary'];assert all(math.isfinite(v) for group in s.values() for v in group.values())
            row=dict(dataset=d['dataset'],query=d['query'],arm=arm,all_BA=s['all']['BA'],old_BA=s['old']['BA'],new_BA=s['new']['BA'],micro=s['all']['micro'],macro_CE=s['all']['CE']);rows.append(row)
            lines.append('| '+d['dataset']+' | '+d['query']+' | '+arm+' | '+' | '.join(f'{row[k]*100:.4f}' for k in ['all_BA','old_BA','new_BA','micro'])+f" | {row['macro_CE']:.5f} |")
        for reference,groups in d['paired'].items():
            for group,p in groups.items():pairs.append(dict(dataset=d['dataset'],query=d['query'],reference=reference,group=group,**p))
    gates={}
    lines+=['','## Fusion deltas and paired outcomes','',
            '| Dataset | Query | Reference | All delta pp | Old delta pp | New delta pp | Rescue | Harm |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for d in data:
        for ref in ['ridge','A','B']:
            a=d['metrics'][ref]['summary'];b=d['metrics']['ridge_local']['summary'];delta={g:b[g]['BA']-a[g]['BA'] for g in ['all','old','new']}
            ok=delta['all']>1e-12 and min(delta['old'],delta['new'])>=-.01-1e-12
            gates.setdefault(d['dataset'],{}).setdefault(ref,{})[d['query']]=dict(delta=delta,pass_descriptive_gate=ok)
            p=d['paired'][ref]['all'];lines.append('| '+d['dataset']+' | '+d['query']+' | '+ref+' | '+' | '.join(f'{delta[g]*100:+.4f}' for g in ['all','old','new'])+f" | {p['rescue']} | {p['harm']} |")
    lines+=['','Descriptive criterion: both queries improve overall BA, with neither old nor new BA losing more than 1 pp. This does not select or deploy a winner.','']
    for name,refs in gates.items():
        for ref,queries in refs.items():
            passed=all(q['pass_descriptive_gate'] for q in queries.values());lines.append(f'- {name} versus {ref}: '+('PASS' if passed else 'FAIL')+' on the frozen two-query criterion.')
    lines+=['','## Limits and interpretation','',
            'The ridge comparison isolates adding the fixed local term to the current v2 global head. Historical B uses the same support-source rows, but differs in representation, sample weighting, regularization, bias and retained state. A additionally has compressed history. Therefore comparisons against A/B are pipeline comparisons, not equal-memory or single-module ablations. '
            'Known identity groups are not verified patients. CE is secondary because score scales differ. Tiny classes can strongly affect BA; full class counts, correct counts and confusion matrices are in results.json, and group rescue/harm counts are in paired.csv. '
            'Calibration uses only base support, but uses fitted-training scores and has no held-out calibration guarantee. No claim of old-score invariance for the full ridge fusion. No test-set, multi-seed or clinical generalization claim.']
    for name,items in [('summary.csv',rows),('paired.csv',pairs)]:
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    (root/'gates.json').write_text(json.dumps(gates,indent=2));(root/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]))
