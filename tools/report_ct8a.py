"""Stdlib-only retrospective analysis of previously recorded adapter gradients."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import time


def geometry(real, weighted_fd, cosine):
    assert math.isfinite(real) and real >= 0 and math.isfinite(weighted_fd) and weighted_fd >= 0
    assert cosine is None if real*weighted_fd == 0 else cosine is not None and abs(cosine)<=1.00001
    c=0 if cosine is None else max(-1.,min(1.,cosine))
    total=math.sqrt(max(0.,real**2+weighted_fd**2+2*real*weighted_fd*c))
    return dict(weighted_ratio=None if real==0 else weighted_fd/real,total_norm=total,
                total_real_cosine=None if real==0 or total==0 else (real+weighted_fd*c)/total)


def check():
    a=geometry(3.,4.,0.); assert a['total_norm']==5 and a['total_real_cosine']==.6
    b=geometry(1.,2.,-1.);assert b['total_norm']==1 and b['total_real_cosine']==-1
    assert geometry(1.,1.,-1.)['total_real_cosine'] is None
    assert geometry(0.,2.,None)['weighted_ratio'] is None
    return dict(status='PASS',checks=['orthogonal_vector_sum','opposition','exact_cancellation','zero_denominator'])


def csvwrite(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def summary(rows, keys):
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    out=[]
    for key,rs in groups.items():
        d=dict(zip(keys,key));d['n']=len(rs)
        for col in ('real_norm','weighted_FD_norm','weighted_ratio','cosine','total_norm','total_real_cosine'):
            v=[x[col] for x in rs if x[col] is not None]
            d.update({col+'_null':len(rs)-len(v),col+'_median':st.median(v) if v else None,
                      col+'_min':min(v) if v else None,col+'_max':max(v) if v else None})
        d['FD_dominant_count']=sum(x['weighted_FD_norm']>x['real_norm'] for x in rs)
        d['opposed_count']=sum(x['cosine'] is not None and x['cosine']<0 for x in rs)
        d['total_against_real_count']=sum(x['total_real_cosine'] is not None and x['total_real_cosine']<0 for x in rs)
        d['real_zero_count']=sum(x['real_norm']==0 for x in rs)
        d['FD_zero_count']=sum(x['weighted_FD_norm']==0 for x in rs)
        out.append(d)
    return out


def main():
    start=time.monotonic();root=Path(os.environ['P27_ROOT']);lock=json.loads((root/'PROTOCOL_LOCK.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(__file__)==lock['source_sha256']
    for file,h in lock['inputs'].items():assert sha(root/file)==h
    out=root/'results';out.mkdir(exist_ok=False)
    try:
        (out/'ENGINEERING_GATE.json').write_text(json.dumps(check(),indent=2))
        grad=[json.loads(x) for x in (root/lock['gradient_log']).read_text().splitlines()]
        logs=[json.loads(x) for x in (root/lock['loss_log']).read_text().splitlines()]
        key=lambda x:(x['dataset'],x['seed'],x['task'],x['epoch'])
        expected={(ds,s,t,e) for ds,tasks in [('HK',range(4,12)),('ISIC',[4])] for s in (1993,1994,1995) for t in tasks for e in range(1,11)}
        assert len(grad)==len(logs)==len(expected)==270
        assert {key(x) for x in grad}=={key(x) for x in logs}==expected
        assert sum(x['optimizer_steps'] for x in logs)==5570
        assert all(x['teacher_unchanged'] and math.isfinite(x['loss']) for x in logs)
        rows=[]
        for x in grad:
            assert set(x['groups'])=={'main','few'}
            for group,g in x['groups'].items():
                calc=geometry(g['real_norm'],10*g['FD_norm'],g['cosine'])
                assert (calc['weighted_ratio'] is None and g['weighted_ratio'] is None) or math.isclose(calc['weighted_ratio'],g['weighted_ratio'],rel_tol=1e-10,abs_tol=1e-12)
                rows.append(dict(dataset=x['dataset'],seed=x['seed'],task=x['task'],epoch=x['epoch'],group=group,
                    phase='epoch1' if x['epoch']==1 else 'epoch2to10',real_norm=g['real_norm'],weighted_FD_norm=10*g['FD_norm'],
                    cosine=g['cosine'],**calc))
        losses=[dict(dataset=x['dataset'],seed=x['seed'],task=x['task'],epoch=x['epoch'],
            main_CE=x['main_CE'],sum_CE=x['sum_CE'],pool_weighted_few=x['pool_weighted_few'],assignment=x['assignment'],pull=x['pull'],weighted_FD=10*x['FD_mean'],total=x['loss']) for x in logs]
        tables={'gradients':rows,'summary':summary(rows,['dataset','group','phase']),
            'by_seed':summary(rows,['dataset','seed','group','phase']),
            'by_task':summary(rows,['dataset','seed','task','group','phase']),'loss_components':losses}
        assert [len(x) for x in tables.values()]==[540,8,24,108,270]
        for name,table in tables.items():csvwrite(out/(name+'.csv'),table)
        elapsed=time.monotonic()-start;size=sum(x.stat().st_size for x in out.iterdir());assert elapsed<30 and size<2**20
        (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE_CT8A',rows={k:len(v) for k,v in tables.items()},NEXT_DECISION='STOP',
            wall_seconds=elapsed,output_bytes=size,new_GPU_seconds=0,new_forwards=0,new_optimizer_steps=0,new_predictions=0,new_image_fit_test_reserved_reads=0),indent=2))
    except Exception as e:
        (out/'FAILURE.json').write_text(json.dumps(dict(status='BLOCKED',error=type(e).__name__,reason=str(e),wall_seconds=time.monotonic()-start)))
        raise


if __name__=='__main__':main()
