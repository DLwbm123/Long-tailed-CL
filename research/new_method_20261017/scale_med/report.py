"""Generate a public aggregate report; never includes individual predictions or paths."""
import csv
import json
from pathlib import Path
import sys

def main(root):
    data=json.loads((root/'results.json').read_text())
    state=json.loads((root/'status.private.json').read_text())
    assert state['state']=='COMPLETE' and len(data)==6
    support=json.loads((root/'support.json').read_text())
    lines=['# SCALE-Med v1: NIH author20 development experiment','',
        'Frozen ImageNet ViT-B/16; 128 fixed projected coordinates; one Gaussian per class. '
        'Patient-equal training weights. Reference covariance is fixed from base classes only. '
        'One class order and one seed; no validation parameter search. '
        'The author20 single-label development split is not the original multilabel clinical task, '
        'an independent test, or evidence of medical safety.','',
        '| Arm | Mean stage macro | Final macro | Final micro | Mean forgetting |',
        '|---|---:|---:|---:|---:|']
    records=[]
    for arm in data[-1]['metrics']:
        final=data[-1]['metrics'][arm]
        forgetting=[]
        for c in data[-2]['seen']:
            past=[s['metrics'][arm]['per_class'][str(c)] for s in data[:-1] if c in s['seen']]
            forgetting.append(max(past)-final['per_class'][str(c)])
        row=dict(arm=arm,mean_stage_macro=sum(s['metrics'][arm]['macro'] for s in data)/len(data),
                 final_macro=final['macro'],final_micro=final['micro'],mean_forgetting=sum(forgetting)/len(forgetting))
        records.append(row)
        lines.append('| '+arm+' | '+' | '.join(f'{row[k]:.4f}' for k in list(row)[1:])+' |')
    lines += ['', '## Rank diagnostics', '',
              'Fixed old scores do not prevent new classes from winning. Entropy and ridge can change old scores. '
              'The following counts compare predictions restricted to old classes on the same old validation images.', '',
              '| Arm | Old-winner flips | Harm | Rescue | New-class wins (stage-summed) | Max old-score change |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in data[-1]['metrics']:
        ds=[s['drift'][arm] for s in data[1:]]
        lines.append('| '+arm+' | '+' | '.join(str(sum(d[k] for d in ds)) for k in ['old_pair_winner_flips','harm','rescue','new_class_wins'])+f" | {max(d['max_old_score_change'] for d in ds):.6g} |")
    lines += ['', 'Stage-summed counts can include the same image at multiple stages. '
              'No confidence interval or multi-order robustness claim is supported. '
              'Patient identifiers, images, embeddings, raw predictions, and model states are excluded from public delivery.', '',
              f"Training support ranges from {min(v['patients'] for v in support.values())} to {max(v['patients'] for v in support.values())} patients per class."]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

if __name__=='__main__':main(Path(sys.argv[1]))
