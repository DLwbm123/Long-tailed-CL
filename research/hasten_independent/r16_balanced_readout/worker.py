"""One fixed HASTEN memory-branch intervention; reuse H0 learned scores read-only."""
import ast, copy, json, os, time, traceback, types
from pathlib import Path
import torch
import torch.nn.functional as F

def selected(source, names, namespace):
    tree = ast.parse(Path(source).read_text())
    tree.body = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in tree.body} == set(names)
    exec(compile(tree,'<pinned-components>','exec'),namespace)
    return namespace

def fusion(source):
    tree = ast.parse(Path(source).read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_ens']
    assert len(nodes)==1
    ns = dict(F=F)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-fusion>','exec'),ns)
    return lambda h,p: ns['get_ens'](types.SimpleNamespace(args={'stat':.5}),h,p)

def metrics(scores,y,n,tail):
    assert scores.shape==(len(y),n) and torch.isfinite(scores).all()
    pred = scores.argmax(1); counts = torch.bincount(y,minlength=n)
    correct = torch.bincount(y[pred==y],minlength=n)
    assert (counts>0).all()
    rec = correct.double()/counts
    return dict(BA=float(rec.mean()*100),tail_BA=float(rec[[c for c in tail if c<n]].mean()*100),counts=counts.tolist(),correct=correct.tolist())

def selfcheck():
    h=torch.tensor([[-4.,-2.,-1.],[-1.,-3.,-2.]],dtype=torch.double)
    p=torch.tensor([[.2,.5,.1],[.4,.1,.2]],dtype=torch.double)
    mix=(F.softmax(F.normalize(h,dim=1),1)+F.softmax(F.normalize(p,dim=1),1))*.5
    assert torch.isfinite(mix).all() and torch.allclose(mix.sum(1),torch.ones(2,dtype=torch.double))
    m=metrics(mix[:,:2],torch.tensor([1,0]),2,[0,1])
    assert m['counts']==[1,1] and m['correct']==[1,1]
    return dict(passed=True,synthetic_counts=True)

def run():
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def read(p):return json.loads(Path(p).read_text())
    def write(name,obj):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
        tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
    def status(**kw):write('STATUS.json',dict(time=time.time(),**kw))
    def finite(x):assert torch.isfinite(x).all()
    write('SYNTHETIC_ACCEPTANCE.json',selfcheck())
    d15=Path(cfg['d15']);control=read(d15/'CONTROL_EXIT.json');assert control['exit_code']==0
    old_results=read(d15/'RESULTS.json');results={}
    for ds in ['NIH','ISIC','HK']:
        status(dataset=ds,phase='PREPARE_SEALED_HEADS')
        spec=read(cfg['configs'][ds]);source=Path(cfg['h0_sources'][ds]);seal=read(source/'SPLIT_SEAL.json')
        assert seal==old_results[ds]['split'];order=seal['original_order'];stages=seal['stages']
        ns=dict(CFG=spec,ORDER=order,Path=Path,torch=torch)
        rows,tr,te=selected(spec['base_runner'],['read_rows'],ns)['read_rows']()
        y=torch.tensor([r['label'] for r in rows]);tail=[i for i,n in enumerate(seal['train_counts']) if n<250]
        assert torch.bincount(y[te]).tolist()==seal['eval_counts']
        heads=[torch.load(d15/ds/'heads'/f'Biomed_{s}.private.pt',map_location='cpu',weights_only=False) for s in range(len(stages))]
        for s,head in enumerate(heads):
            assert head['classes']==sum(map(len,stages[:s+1]))
            finite(head['ridge']);assert head['ridge'].shape==(513,head['classes'])
        write(ds+'/ALL_HEADS_SEALED.json',dict(time=time.time(),reused_D15_TRAIN_heads=True,stages=len(heads),validation_image_reads=0,no_fitting=True))
        ns=dict(CFG=spec,Path=Path,torch=torch,json=json,copy=copy,time=time,os=os,finite=finite,status=lambda **kw:status(dataset=ds,**kw),write=lambda p,v:write(ds+'/'+p,v))
        engine=selected(cfg['encoder_source'],['EncoderCache'],ns)['EncoderCache'](rows)
        cache=out/ds/'val_features.private.pt'
        if cache.exists():
            rec=torch.load(cache,map_location='cpu',weights_only=False);assert rec['ids']==te;x=rec['x']
        else:
            x=F.normalize(engine.encode(te,'ENCODE_VAL_AFTER_SEAL'),dim=1)
            torch.save(dict(x=x,ids=te),cache)
        engine.release();finite(x);assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6)
        ens=fusion(Path(spec['official'])/'models/hasten.py');agg=read(source/'aggregate.json')
        refs={(r['seed'],r['stage']):r for r in agg['results'] if r['arm']=='H-unit'}
        records=[]
        for seed in [74002,74003,74004]:
            for stage,head in enumerate(heads):
                n=head['classes'];keep=y[te]<n;ids=[i for i in te if int(y[i])<n];yy=y[ids];xx=x[keep].double()
                saved=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage}_responses.private.pt',map_location='cpu',weights_only=False)
                assert saved['ids']==ids and torch.equal(saved['labels'],yy)
                h,p,f=[saved['scores'][k] for k in ['hyper','prototype','final']]
                replay=ens(h,p);error=float((replay-f).abs().max())
                assert error<1e-7 and torch.equal(replay.argmax(1),f.argmax(1)),('H0 fusion replay',ds,seed,stage,error)
                pc=xx@head['prototype'].T;protoerr=float((pc-p).abs().max())
                assert protoerr<2e-5 and torch.equal(pc.argmax(1),p.argmax(1)),('feature replay',ds,seed,stage,protoerr)
                ridge=torch.cat([xx,torch.ones(len(xx),1,dtype=torch.double)],1)@head['ridge']
                rm=metrics(ridge,yy,n,tail);old=next(a for a in old_results[ds]['results'] if a['encoder']=='Biomed' and a['stage']==stage)['metrics']['ridge']
                assert rm['correct']==old['correct'] and rm['counts']==old['counts'],('D15 ridge replay',ds,stage)
                changed=ens(h,ridge);m=metrics(changed,yy,n,tail)
                base=refs[seed,stage]['metrics']['final'];assert m['counts']==base['counts']
                newok=changed.argmax(1)==yy;oldok=f.argmax(1)==yy
                rescue=torch.bincount(yy[newok&~oldok],minlength=n).tolist();harm=torch.bincount(yy[oldok&~newok],minlength=n).tolist()
                assert [c+r-h for c,r,h in zip(base['correct'],rescue,harm)]==m['correct']
                fixed=[]
                for c in range(n):
                    birth=next(s for s,cs in enumerate(stages) if c in cs);birthn=sum(map(len,stages[:birth+1]))
                    fixed.append(int((changed[yy==c,:birthn].argmax(1)==c).sum()))
                record=dict(seed=seed,stage=stage,seen=n,metrics=m,H0=base,ridge=rm,rescue=rescue,harm=harm,fixed_arrival_correct=fixed,H0_fusion_replay_max_error=error,H0_prototype_replay_max_error=protoerr)
                records.append(record)
                torch.save(dict(ids=ids,labels=yy,scores=changed),out/ds/f'{seed}_{stage}_responses.private.pt')
        results[ds]=dict(records=records,split=seal)
        write(ds+'/RESULTS.json',results[ds]);write('RESULTS.partial.json',results)
        status(dataset=ds,phase='COMPLETE_DATASET')
    write('RESULTS.json',results);status(phase='COMPLETE',optimizer_steps=0,new_fits=0)

if __name__=='__main__':
    try:run()
    except BaseException:
        out=Path(json.loads(Path(os.environ['RUN_CONFIG']).read_text())['output'])
        (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
