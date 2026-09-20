"""Read only locked aggregate epoch logs; no ML or data/model imports."""
import csv,hashlib,json,math,os,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decompose(r,beta):
    components=dict(CE_main_sum=(r['main_CE']+r['sum_CE'])/3,
                    CE_few=r['pool_weighted_few']/3,assignment=r['assignment'],
                    pull=r['pull'],FD=beta*r['FD_mean'])
    total=sum(components.values());residual=r['loss']-total
    assert all(math.isfinite(v) for v in [*components.values(),total,residual])
    assert abs(residual)<=1e-4*(1+abs(r['loss'])), ('LOSS_ACCOUNTING',residual)
    return dict(**components,reconstructed=total,recorded=r['loss'],residual=residual)
def save_csv(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=time.monotonic();root=Path(os.environ['P29_ROOT']).resolve();out=root/'docs/ct10o_objective_accounting'
    lock=json.loads((out/'PROTOCOL_LOCK.json').read_text())
    assert sha(Path(__file__))==lock['worker_sha256']
    allowed={str((root/p).resolve()) for p in lock['inputs']}
    # Limit asset reads and all output writes; imports are stdlib and already loaded.
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
        if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):assert p.is_relative_to(out)
        else:assert str(p) in allowed,('UNLOCKED_READ',str(p))
    import sys
    sys.addaudithook(guard)
    for p,h in lock['inputs'].items():assert sha(root/p)==h
    s=dict(main_CE=3.,sum_CE=6.,pool_weighted_few=9.,assignment=2.,pull=-.1,FD_mean=.2,loss=9.9)
    assert abs(decompose(s,10)['residual'])<1e-12
    try:decompose(dict(s,loss=11),10)
    except AssertionError:pass
    else:raise AssertionError('CORRUPT_RECORD_ACCEPTED')
    rows=[];summary=[]
    for beta,path in lock['logs']:
        items=[json.loads(l) for l in (root/path).read_text().splitlines()]
        selected=[r for r in items if r['task']==4]
        assert len(selected)==60
        assert sum(r['optimizer_steps'] for r in selected)==2750
        for ds in ('HK','ISIC'):
            for seed in (1993,1994,1995):
                q=sorted((r for r in selected if r['dataset']==ds and r['seed']==seed),key=lambda r:r['epoch'])
                assert [r['epoch'] for r in q]==list(range(1,11))
                rr=[dict(beta=beta,dataset=ds,seed=seed,task=4,epoch=r['epoch'],steps=r['optimizer_steps'],**decompose(r,beta)) for r in q]
                rows+=rr;s=dict(beta=beta,dataset=ds,seed=seed,task=4)
                for name in ('CE_main_sum','CE_few','assignment','pull','FD','recorded'):
                    s[name+'_e1']=rr[0][name];s[name+'_e10']=rr[-1][name]
                s['max_abs_residual']=max(abs(r['residual']) for r in rr);summary.append(s)
    assert len(rows)==120 and len(summary)==12
    save_csv(out/'loss_components.csv',rows);save_csv(out/'trajectory_summary.csv',summary)
    seconds=time.monotonic()-start;assert seconds<300
    audit=dict(status='PASS',rows=120,trajectories=12,max_abs_residual=max(abs(r['residual']) for r in rows),
               synthetic_formula=True,corrupt_record_rejected=True,CPU_process_seconds=seconds,
               neural_epochs=0,optimizer_steps=0,forward=0,new_predictions=0,GPU_seconds=0,
               image_reads=0,checkpoint_reads=0,feature_reads=0,sample_score_reads=0,test_reserved_reads=0)
    (out/'COMPLETE.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))
if __name__=='__main__':main()
