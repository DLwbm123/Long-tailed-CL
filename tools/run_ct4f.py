"""One frozen-Task1 reference, six orders, three stages; zero neural updates."""
import gc,os,sys,time,shutil
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from run_ct3p import Prefix,read,write_json,sha,network_hash,joint,append,ridge,npz
from run_ct1 import CTLearner
from ct2d_math import metrics,classify
from report_locked_holdout_r1 import csvwrite,bootstrap_weights,boot_unit


def report(root,cfg):
    pub=root/'output/public';private=root/'output/private';ref=Path(cfg['ct3p_root'])/'output'
    units=read(pub/'PREDICTIONS_LOCK.json')['units'];old=read(ref/'public/ONLINE_PREDICTIONS_LOCK.json')['units'];data={};ms=[];pcs=[]
    for e in units:
        with np.load(private/'sealed'/e['file']) as f:p={k:f[k].copy() for k in f.files}
        assert sha(private/'sealed'/e['file'])==e['sha256']
        meta={k:e[k] for k in ('dataset','seed','task','method')}
        m,pc,_=metrics(p['raw'],p['y'],p['order'],e['seen']-2,cfg['frequency_groups'][e['dataset']],meta,p['component']);m['prefix_terminal_BA']=m['balanced_accuracy'] if e['task']==3 else None
        ms.append(m);pcs+=pc;data[e['dataset'],e['seed'],e['task'],'F1']=p
        for method in ('P','C0','C2','C3'):
            q=next(v for v in old if all(v[k]==e[k] for k in ('dataset','seed','task')) and v['method']==method)
            path=ref/'private/sealed'/q['file'];assert sha(path)==q['sha256']
            with np.load(path) as f:v={k:f[k].copy() for k in f.files}
            for k in ('ids','order','y','component'):assert np.array_equal(p[k],v[k])
            data[e['dataset'],e['seed'],e['task'],method]=v
    assert len(ms)==18 and len(pcs)==72;csvwrite(pub/'metrics.csv',ms);csvwrite(pub/'per_class.csv',pcs)
    out=[]
    for ds in ('HK','ISIC'):
        identity={}
        for key,p in data.items():
            if key[0]!=ds:continue
            for sid,label,comp in zip(p['ids'],p['original'],p['component']):
                value=(int(label),str(comp));assert sid not in identity or identity[sid]==value;identity[sid]=value
        ids=sorted(identity);ix={x:i for i,x in enumerate(ids)}
        canon=dict(y=np.zeros(len(ids),int),original=np.array([identity[x][0] for x in ids]),component=np.array([identity[x][1] for x in ids]))
        w=bootstrap_weights(canon,2000,48001,sorted(set(canon['original'])))
        for task in (1,2,3):
            for a,b in [('C3','F1'),('C2','F1'),('F1','P'),('F1','C0')]:
                boot=[];point=[]
                for seed in (1993,1994,1995):
                    pa=data[ds,seed,task,a];pb=data[ds,seed,task,b];weights=w[:,[ix[x] for x in pa['ids']]]
                    ra=boot_unit(pa,weights,True);rb=boot_unit(pb,weights,True);d=(ra-rb).mean(1)
                    def ba(p):
                        pred=classify(p['raw'],p['order']);return np.mean([np.mean(pred[p['y']==c]==c)*100 for c in range(2*task)])
                    delta=float(ba(pa)-ba(pb));lo,hi=np.quantile(d,[.025,.975]);boot.append(d);point.append(delta)
                    out.append(dict(dataset=ds,seed=seed,task=task,contrast=a+'-'+b,difference_pp=delta,low=float(lo),high=float(hi)))
                lo,hi=np.quantile(np.mean(boot,axis=0),[.025,.975]);out.append(dict(dataset=ds,seed='fixed_three_mean',task=task,contrast=a+'-'+b,difference_pp=float(np.mean(point)),low=float(lo),high=float(hi)))
    csvwrite(pub/'paired_BA_intervals.csv',out)
    text=['# CT4-F冻结Task1参照','', '固定三模型/顺序；仅2→4→6类val。无神经训练，无test访问。','', '| 数据 | F1 prefix_terminal_BA |','|---|---:|']
    for ds in ('HK','ISIC'):text.append(f"| {ds} | {np.mean([m['balanced_accuracy'] for m in ms if m['dataset']==ds and m['task']==3]):.3f} |")
    for x in out:
        if x['seed']=='fixed_three_mean' and x['task']==3:text.append(f"- {x['dataset']} {x['contrast']}: {x['difference_pp']:+.3f}pp [{x['low']:+.3f},{x['high']:+.3f}]。")
    text+=['','条件性component区间不代表训练总体；singleton、val反复开发及不同seed类别集合限制保留。冻结Task1是单独控制，不能改写CT3-P定义。NEXT_DECISION=STOP。']
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(text))


def main():
    cfg=read(os.environ['P23_CONFIG']);root=Path(cfg['root']);pub=root/'output/public';private=root/'output/private';began=time.monotonic();cpu=0.;r=None
    lock=read(pub/'PROTOCOL_LOCK.json');assert sha(Path(__file__))==lock['worker_sha256'];assert not (pub/'STATE_W_LOCK.json').exists()
    def protect(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
            if flags and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):assert not p.is_relative_to(Path(cfg['ct3p_root']).resolve())
    sys.addaudithook(protect)
    def resources():
        size=sum(f.stat().st_size for f in root.rglob('*') if f.is_file() and not f.is_symlink());free=shutil.disk_usage(root).free
        assert size<2*1024**3 and free>=1024**3 and cpu<1800
        return dict(wall_seconds=time.monotonic()-began,CPU_analytic_report_seconds=cpu,disk_bytes=size,free_bytes=free,neural_epochs=0,optimizer_steps=0,test_reads=0,old_fit_reads=0,future_fit_reads=0,calls={} if r is None else r.access)
    try:
        r=Prefix(cfg);r.setup();states=[];audit=[]
        def no_training(*args,**kwargs):raise AssertionError('BLOCKED_NEURAL_TRAINING')
        CTLearner._init_train=no_training
        for name in ('HK','ISIC'):
            for seed in (1993,1994,1995):
                e=next(e for e in read(r.old/'output/public/MODEL_LINEAGE.json') if (e['dataset'],e['seed'],e['stream'],e['task'])==(name,seed,'U',1))
                path=private/'parents'/e['name'];assert sha(path)==e['sha256'];l=CTLearner(r.parent_view,name,seed,'U');p=l.restore(path);l.r=r
                assert p['task']==0 and p['epoch']==10 and p['seen']==2
                for v in l._network.parameters():v.requires_grad_(False)
                fingerprint=network_hash(l._network);state=p['stats'];del p
                for task in (1,2,3):
                    r.permit(name,seed,task)
                    if task>1:
                        probe_began=time.monotonic()
                        raw,_,_,y,_=r.extract(l,range(2*(task-1),2*task));state=append(state,joint(raw),y,range(2*(task-1),2*task),task);del raw
                        assert state['n'][-2:].sum()==l.tasks[task-1]['n_train']
                        if name=='HK' and seed==1993 and task==2:write_json(pub/'THROUGHPUT_GATE.json',dict(status='PASS',probe_seconds=time.monotonic()-probe_began,rows=l.tasks[task-1]['n_train'],resources=resources()))
                    t=time.monotonic();W,diag=ridge(state);cpu+=time.monotonic()-t
                    assert network_hash(l._network)==fingerprint
                    if task==1:
                        with np.load(r.old/'output/private/sealed'/f'{name}_{seed}_U_W_t01.npz') as f:np.testing.assert_allclose(W,f['CT-J-CB'],atol=1e-9,rtol=1e-9)
                    stem=f'{name}_{seed}_t{task:02d}';npz(private/'banks'/(stem+'.npz'),W=W)
                    states.append(dict(dataset=name,seed=seed,task=task,method='F1',seen=2*task,W_file=stem+'.npz',W_sha256=sha(private/'banks'/(stem+'.npz')),network_sha256=fingerprint,parent_sha256=e['sha256']))
                    audit.append(dict(dataset=name,seed=seed,task=task,counts=state['n'].tolist(),network_unchanged=True,**diag));write_json(pub/'FIT_AUDIT.json',audit);write_json(pub/'RESOURCES.json',resources());print('FIT',name,seed,task,flush=True)
                del l,state;gc.collect();torch.cuda.empty_cache()
        assert len(states)==18;write_json(pub/'STATE_W_LOCK.json',dict(status='LOCKED',units=states,protocol=lock))
        units=[];r.mode='evaluate';r.phase='evaluate'
        for name in ('HK','ISIC'):
            for seed in (1993,1994,1995):
                path=private/'parents'/f'{name}_{seed}_U_t01.pt';l=CTLearner(r.parent_view,name,seed,'U');p=l.restore(path);l.r=r;fingerprint=network_hash(l._network);del p
                for v in l._network.parameters():v.requires_grad_(False)
                for task in (1,2,3):
                    r.permit(name,seed,task);raw,_,_,y,rows=r.extract(l,range(2*task),'val');e=next(e for e in states if (e['dataset'],e['seed'],e['task'])==(name,seed,task))
                    wp=private/'banks'/e['W_file'];assert sha(wp)==e['W_sha256']
                    with np.load(wp) as f:score=joint(raw)@f['W']
                    order=np.array(l.order[:2*task]);ids=np.array([v['sample_id'] for v in rows]);comp=np.array([v['identity_component'] for v in rows])
                    if task==1:
                        with np.load(Path(cfg['ct3p_root'])/'output/private/sealed'/f'{name}_{seed}_C0_t01.npz') as f:
                            assert np.array_equal(ids,f['ids']);np.testing.assert_allclose(score,f['raw'],atol=1e-5,rtol=1e-5);assert np.array_equal(classify(score,order),classify(f['raw'],order))
                    assert network_hash(l._network)==fingerprint
                    filename=f'{name}_{seed}_F1_t{task:02d}.npz';npz(private/'sealed'/filename,raw=score,y=y,original=order[y],order=order,ids=ids,component=comp)
                    units.append(dict(e,file=filename,sha256=sha(private/'sealed'/filename)));del raw
                del l;gc.collect();torch.cuda.empty_cache()
        write_json(pub/'PREDICTIONS_LOCK.json',dict(status='LOCKED',units=units));t=time.monotonic();report(root,cfg);cpu+=time.monotonic()-t
        write_json(pub/'RESOURCES.json',resources());write_json(pub/'COMPLETE.json',dict(status='COMPLETE_CT4F',NEXT_DECISION='STOP',units=18,neural_epochs=0,optimizer_steps=0))
    except BaseException as e:write_json(pub/'FAILURE.json',dict(status='BLOCKED',error=type(e).__name__,reason=str(e),elapsed=time.monotonic()-began));raise
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
