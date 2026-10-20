"""Sealed risk and input-gradient support; no encoder import/fit/update."""
import json, os, time, resource
from pathlib import Path
import torch

torch.set_num_threads(4)

def transform(g, mp):
    return (((g-mp['gn'][0])/mp['gn'][1])[:,mp['ids']]-mp['on'][0])/mp['on'][1]

def qstats(v):
    assert torch.isfinite(v).all()
    return dict(mean=float(v.mean()),quantiles=torch.quantile(v,torch.tensor([0.,.1,.5,.9,1.],dtype=v.dtype)).tolist())

def group_weights(groups):
    keys=sorted(set(groups)); ix=torch.tensor([keys.index(g) for g in groups]); n=torch.bincount(ix).double()
    return ix, 1/n[ix]/len(keys)

def gradient_norm(score,y,w):
    eta=score.softmax(1)-torch.nn.functional.one_hot(y,4)
    gram=w.T@w
    return ((eta@gram)*eta).sum(1).clamp_min(0).sqrt()

def synthetic():
    z=torch.tensor([[1.,-2.],[.2,.4]],dtype=torch.float64,requires_grad=True)
    w=torch.tensor([[.1,.2,-.3,.5],[.4,-.2,.1,.7]],dtype=z.dtype)
    y=torch.tensor([1,3]); score=z@w
    grad=torch.autograd.grad(torch.nn.functional.cross_entropy(score,y,reduction='sum'),z)[0]
    err=float((grad.norm(dim=1)-gradient_norm(score,y,w)).abs().max()); assert err<1e-12
    ix,q=group_weights(['a','a','b']); assert torch.allclose(q,torch.tensor([.25,.25,.5],dtype=q.dtype))
    assert torch.allclose(torch.zeros(2,4).softmax(1),torch.full((2,4),.25))
    assert torch.zeros(2,4).argmax(1).tolist()==[0,0]
    return dict(analytic_autograd_norm_error=err,group_weight_and_tie_checks=True)

def main(cfg):
    tests=synthetic(); old=Path(cfg['anchor']); new=Path(cfg['readout'])
    ocfg=json.loads((old/'fit.config.private.json').read_text())
    inputs=Path(ocfg['reference'])/'inputs/0.private.pt'
    paths=[inputs,old/'train/protection.private.pt',old/'fit/L.g.private.pt',old/'fit/L.state.private.pt']
    paths += [new/'fit'/(a+'.state.private.pt') for a in ['P','C','Q','R']]
    paths += [new/'fit'/(a+'.g.private.pt') for a in ['P','C']]
    allowed={str(p.resolve()) for p in paths}; original=torch.load
    def load(p,*args,**kw):
        assert str(Path(p).resolve()) in allowed,'READ_DENIED'
        return original(p,map_location='cpu',weights_only=False)
    torch.load=load
    obj=load(inputs); y=obj['y']; valid=obj['valid']; groups=obj['groups']
    assert [int((~valid).sum()),int(valid.sum())]==[2555,681]
    assert not {groups[i] for i in torch.where(valid)[0]} & {groups[i] for i in torch.where(~valid)[0]}
    teacher=load(old/'train/protection.private.pt')
    head={'F':(teacher['mapper'],teacher['teacher_W'])}
    features={'F':obj['g'].double(),'L':load(old/'fit/L.g.private.pt').double()}
    for a in ['P','C']: features[a]=load(new/'fit'/(a+'.g.private.pt')).double()
    for a in ['L','P','C','Q','R']:
        r=load((old if a=='L' else new)/'fit'/(a+'.state.private.pt'))
        head[a]=(r['mapper'],r['W'][:,:4])
    for base,arms in [(old,['L']),(new,['P','C'])]:
        for a in arms:
            seal=json.loads((base/'fit'/(a+'.FEATURE_SEAL.json')).read_text()); assert seal['rows']==3236 and seal['one_pass']
    expected=json.loads((new/'fit/aggregate.json').read_text())['summary']
    score={a:transform(features[{'Q':'P','R':'C'}.get(a,a)],mp)@w[:-1,:4]+w[-1,:4] for a,(mp,w) in head.items()}
    out={}; cells=0
    for a,s in score.items():
        assert torch.isfinite(s).all()
        pred=s.argmax(1); ce=torch.nn.functional.cross_entropy(s,y,reduction='none')
        rival=s.clone(); rival[torch.arange(len(y)),y]=-torch.inf; margin=s[torch.arange(len(y)),y]-rival.max(1).values
        norm=gradient_norm(s,y,head[a][1][:-1,:4]); out[a]={}
        for split,mask in [('train',~valid),('development',valid)]:
            rows=[]
            for c in range(4):
                ids=torch.where(mask & (y==c))[0]; iy=y[ids]; ok=pred[ids]==iy; fok=score['F'][ids].argmax(1)==iy
                ix,q=group_weights([groups[i] for i in ids]); ng=int(ix.max())+1
                mass=torch.zeros(ng,dtype=q.dtype).scatter_add_(0,ix,q*norm[ids]); total=mass.sum()
                ess=float(total.square()/mass.square().sum()) if total>0 else None
                r,h=int((ok & ~fok).sum()),int((~ok & fok).sum())
                assert r-h==int(ok.sum())-int(fok.sum())
                rows.append(dict(class_index=c,images=len(ids),groups=ng,correct=int(ok.sum()),recall=float(ok.double().mean()),
                    CE=qstats(ce[ids]),margin=qstats(margin[ids]),group_balanced_CE=float(q@ce[ids]),
                    group_balanced_recall=float(q@ok.double()),rescued_vs_F=r,harmed_vs_F=h,
                    CE_gradient_mapper_output_norm=qstats(norm[ids]),gradient_norm_group_ESS=ess,
                    gradient_norm_group_ESS_fraction=None if ess is None else ess/ng,
                    max_gradient_norm_group_share=float(mass.max()/total) if total>0 else None,
                    equal_group_weight_baseline_ESS=ng))
                cells+=1
            out[a][split]=dict(classes=rows,BA=sum(v['recall'] for v in rows)/4,
                group_balanced_BA=sum(v['group_balanced_recall'] for v in rows)/4,
                group_balanced_CE=sum(v['group_balanced_CE'] for v in rows)/4)
        assert [v['correct'] for v in out[a]['development']['classes']]==expected[a]['correct']
    flags={a:bool(out[a]['train']['classes'][2]['recall']>out['F']['train']['classes'][2]['recall'] and out[a]['development']['classes'][2]['recall']<out['F']['development']['classes'][2]['recall']) for a in ['P','C']}
    assert cells==48
    result=dict(endpoints=out,class_split_cells=cells,synthetic=tests,original_six_development_parity=True,
        TRAIN_GAIN_DEVELOPMENT_HARM=flags,COMMON_TRAIN_GAIN_DEVELOPMENT_HARM=all(flags.values()),
        gradient_support_location='mapper_output_z_not_encoder_parameters',direction_stability_measured=False,
        covariance_or_AGOP_not_computed=True,method_success=False,future_used=False,test_used=False,
        encoder_forwards=0,new_ridge_fits=0,parameter_updates=0,development_not_independent=True,
        source_commit=cfg['source_commit'])
    Path(cfg['output'],'aggregate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    start=time.monotonic(); cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text()); code=1
    try: main(cfg);code=0
    except BaseException as e:
        import traceback
        Path(cfg['output'],'FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__)))
        Path(cfg['output'],'FAILURE.private.txt').write_text(traceback.format_exc())
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF)
        Path(cfg['output'],'WORKER_RESOURCE_RECEIPT.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized())))
    raise SystemExit(code)
