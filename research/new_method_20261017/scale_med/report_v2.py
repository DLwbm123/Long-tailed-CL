import json,csv,sys
from pathlib import Path

def main(root):
    state=json.loads((root/'status.private.json').read_text());data=json.loads((root/'results.json').read_text())
    assert state['state']=='COMPLETE' and len(data)==6
    rows=[]
    for arm in data[-1]['metrics']:
        rows.append(dict(arm=arm,mean_stage_macro=sum(d['metrics'][arm]['macro'] for d in data)/6,final_macro=data[-1]['metrics'][arm]['macro'],final_micro=data[-1]['metrics'][arm]['micro']))
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Fixed-scale ridge plus uniform local evidence: v2','',
           'NIH author20 single-label development split; same one seed/order as v1. '
           'Reuses v1 frozen features and sufficient statistics; newly extracts only base-training scale-calibration examples. '
           'No nonlinear reference mixture. Local covariance shrinkage is unchanged. '
           'Scales are frozen base-training class-balanced RMS class-centered scores; lambda=0.5. '
           'No validation parameter selection. The global ridge is updated with arriving classes, so only local scores have old-score invariance.','',
           '| Arm | Mean stage macro | Final macro | Final micro |','|---|---:|---:|---:|']
    for r in rows:lines.append('| '+r['arm']+' | '+' | '.join(f'{r[k]:.4f}' for k in list(r)[1:])+' |')
    lines+=['','| Stage | Fusion changes vs ridge | Harm | Rescue |','|---|---:|---:|---:|']
    for d in data:
        e=d['fusion_effect'];lines.append(f"| {d['stage']} | {e['changed']} | {e['harm']} | {e['rescue']} |")
    delta=data[-1]['metrics']['ridge_local']['macro']-data[-1]['metrics']['ridge']['macro']
    lines+=['',f'Final fusion minus ridge: {delta*100:+.2f} percentage points. '
            'This is an exploratory development-set result, not an independent replication or a medical-safety claim. '
            'Control per-class accuracies must match v1 at every stage; the runner checks this before accepting a stage. '
            'Original v1 files are read-only. No private data or per-example outputs are included.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]))
