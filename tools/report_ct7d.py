"""CPU-only paired error decomposition of locked terminal validation scores."""
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from ct2d_math import classify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def events(a, b, y):
    ac, bc = a == y, b == y
    out = dict(both_correct=ac & bc, corrected=ac & ~bc, harmed=~ac & bc,
               both_wrong_same=(~ac & ~bc) & (a == b),
               both_wrong_different=(~ac & ~bc) & (a != b))
    assert np.all(sum(out.values()) == 1)
    return out


def summarize(ev, y, labels):
    mask = np.isin(y, labels)
    assert len(labels) and mask.any() and all((y == k).any() for k in labels)
    out = dict(n_images=int(mask.sum()), n_classes=len(labels))
    for name, v in ev.items():
        out[name + '_n'] = int(v[mask].sum())
        out[name + '_sample'] = 100 * float(v[mask].mean())
        out[name + '_macro'] = 100 * float(np.mean([v[y == k].mean() for k in labels]))
    for mode in ('sample', 'macro'):
        assert abs(sum(out[n + '_' + mode] for n in ev) - 100) < 1e-10
        out['agreement_' + mode] = out['both_correct_' + mode] + out['both_wrong_same_' + mode]
        out['net_' + mode] = out['corrected_' + mode] - out['harmed_' + mode]
    return out


def selfcheck():
    y = np.array([0, 0, 0, 0, 1, 1])
    a = np.array([0, 0, 1, 1, 0, 2])
    b = np.array([0, 1, 0, 1, 2, 1])
    ev = events(a, b, y); s = summarize(ev, y, [0, 1])
    assert [s[n + '_n'] for n in ev] == [1, 1, 2, 1, 1]
    np.testing.assert_allclose(s['net_macro'], -25)
    np.testing.assert_allclose(s['net_sample'], -100 / 6)
    assert summarize(events(a, a, y), y, [0, 1])['agreement_macro'] == 100
    order = np.array([2, 0, 1]); raw = np.array([[0., 0., 0.], [4., 2., 3.]])
    pred = order[classify(raw, order)]; perm = np.array([2, 0, 1])
    assert np.array_equal(pred, [0, 2])
    assert np.array_equal(pred, order[perm][classify(raw[:, perm], order[perm])])
    return dict(status='PASS',checks=['event_partition','class_balance','gain_loss_identity',
                                    'identity_pair','label_permutation','tie_original_label'])


def writecsv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def main():
    start = time.monotonic()
    root = Path(os.environ['P26_ROOT']); out = root / 'output'
    lock = json.loads((root / 'PROTOCOL_LOCK.json').read_text())
    assert sha(__file__) == lock['source_sha256']
    for file, expected in lock['dependencies'].items():
        assert sha(file) == expected
    for file, expected in lock['source_locks'].items():
        assert sha(file) == expected, 'BLOCKED_SOURCE_LOCK'
    assert not out.exists(), 'BLOCKED_ALREADY_STARTED'
    assert os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 2**30
    resource.setrlimit(resource.RLIMIT_CPU, (170, 180))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    out.mkdir()
    allowed = {str(Path(e['path']).resolve()) for e in lock['inputs']}
    allowed.add(str(Path(lock['reference_csv']).resolve()))
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve(); flags = args[2] if len(args)>2 else 0
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            assert p.is_relative_to(out.resolve()), 'BLOCKED_WRITE'
        elif p.suffix.lower() in ('.npz', '.npy', '.pt', '.pth', '.jpg', '.jpeg', '.png', '.csv'):
            assert str(p) in allowed, 'BLOCKED_ASSET_ACCESS'
        assert not any(x.lower().startswith(('test', 'reserved')) for x in p.parts), 'BLOCKED_TEST'
    sys.addaudithook(guard)
    try:
        (out / 'ENGINEERING_GATE.json').write_text(json.dumps(selfcheck(), indent=2))
        assert sha(lock['reference_csv']) == lock['reference_csv_sha256']
        with open(lock['reference_csv']) as f:
            ref = {(x['dataset'], int(x['seed']), x['method']):x for x in csv.DictReader(f)}
        data = {}; lineage = []
        for e in lock['inputs']:
            assert sha(e['path']) == e['sha256'], 'BLOCKED_PREDICTION_HASH'
            with np.load(e['path'], allow_pickle=False) as f:
                p = {k:f[k].copy() for k in ['raw','y','ids','order','original','component']}
            c=e['seen']; p['order']=p['order'][:c]
            assert p['raw'].shape == (len(p['y']),c) and np.isfinite(p['raw']).all()
            assert len(np.unique(p['order'])) == c and np.all((p['y']>=0)&(p['y']<c))
            assert np.array_equal(p['order'][p['y']],p['original'])
            assert len(np.unique(p['ids']))==len(p['ids']) and np.array_equal(p['ids'],np.sort(p['ids']))
            pred = classify(p['raw'],p['order'])
            key=(e['dataset'],e['seed'],e['method']); assert key not in data
            ba=100*np.mean([(pred[p['y']==k]==k).mean() for k in range(c)])
            np.testing.assert_allclose(ba,float(ref[key]['balanced_accuracy']),atol=1e-10,rtol=0)
            p.update(pred=pred,known=e['known']);data[key]=p
            lineage.append({k:e[k] for k in ['dataset','seed','method','file','sha256','seen','known']})
        assert len(data)==24
        rows=[];classes=[];groups=[];audit=[]
        for ds,c in [('HK',23),('ISIC',8)]:
            for seed in (1993,1994,1995):
                for a,b in lock['comparisons']:
                    x=data[ds,seed,a];z=data[ds,seed,b]
                    for k in ['ids','y','order','original','component']:
                        assert np.array_equal(x[k],z[k]), 'BLOCKED_LAYOUT'
                    assert x['known']==z['known']
                    ev=events(x['pred'],z['pred'],x['y']);meta=dict(dataset=ds,seed=seed,contrast=a+'-'+b)
                    summ=summarize(ev,x['y'],list(range(c)))
                    row=dict(meta,**summ); rows.append(row)
                    delta=float(ref[ds,seed,a]['balanced_accuracy'])-float(ref[ds,seed,b]['balanced_accuracy'])
                    accdelta=float(ref[ds,seed,a]['accuracy'])-float(ref[ds,seed,b]['accuracy'])
                    np.testing.assert_allclose([summ['net_macro'],summ['net_sample']],[delta,accdelta],atol=1e-10,rtol=0)
                    audit.append(dict(meta,BA_delta=delta,decomposed_BA_delta=summ['net_macro']))
                    for k in range(c):
                        classes.append(dict(meta,original_label=int(x['order'][k]),**summarize(ev,x['y'],[k])))
                    scopes=dict(all=list(range(c)),old=list(range(x['known'])),current=list(range(x['known'],c)))
                    scopes.update({name:np.flatnonzero(np.isin(x['order'],labs)).tolist() for name,labs in lock['frequency_groups'][ds].items()})
                    for scope,labels in scopes.items():
                        groups.append(dict(meta,scope=scope,**summarize(ev,x['y'],labels)))
        assert (len(rows),len(classes),len(groups))==(24,372,144)
        writecsv(out/'pairs.csv',rows);writecsv(out/'per_class.csv',classes);writecsv(out/'groups.csv',groups)
        (out/'AUDIT.json').write_text(json.dumps(dict(status='PASS',pairs=audit,lineage=lineage),indent=2))
        size=sum(p.stat().st_size for p in out.iterdir()); assert size < 10*2**20
        elapsed=time.monotonic()-start;assert elapsed<180
        (out/'RESOURCES.json').write_text(json.dumps(dict(wall_seconds=elapsed,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,output_bytes=size,
            saved_prediction_files=24,new_forwards=0,new_predictions=0,train_steps=0,GPU_seconds=0,
            test_reads=0,reserved_reads=0,image_reads=0,fit_reads=0,checkpoint_reads=0),indent=2))
        (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE_CT7D',pairs=24,per_class=372,groups=144,NEXT_DECISION='STOP'),indent=2))
    except Exception as e:
        (out/'FAILURE.json').write_text(json.dumps(dict(status='BLOCKED',error=type(e).__name__,reason=str(e),wall_seconds=time.monotonic()-start),indent=2))
        raise


if __name__ == '__main__':
    if os.environ.get('P26_MODE') == 'selfcheck':
        print(json.dumps(selfcheck()))
    else:
        main()
