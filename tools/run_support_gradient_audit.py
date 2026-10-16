"""Read-only old/new support gradient geometry at four frozen model states."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT,clone_model,scores,training_batch
from check_fullslot_retention_memory import centered
from run_real_query_node import load,save


def geometry(grads,mass):
    assert len(grads)==3 and all(bool(torch.isfinite(g).all()) for g in grads)
    gram=torch.stack(grads)@torch.stack(grads).T
    norms=gram.diag().clamp_min(0).sqrt()
    cosine=[[float(gram[i,j]/(norms[i]*norms[j])) if float(norms[i]*norms[j])>1e-24 else None for j in range(3)] for i in range(3)]
    combined=mass*grads[0]+(1-mass)*grads[1]+grads[2]
    return dict(norms=norms.tolist(),gram=gram.tolist(),cosine=cosine,
                original_full_objective_gradient_norm=float(combined.norm()),
                derivative_along_negative_full=[-float(g@combined) for g in grads],
                derivative_along_negative_new=[-float(g@grads[1]) for g in grads],
                old_new_conflict=cosine[0][1] is not None and cosine[0][1]<-1e-12)


def unit():
    g=[torch.tensor([1.,0.],dtype=DT),torch.tensor([-1.,1.],dtype=DT),torch.zeros(2,dtype=DT)]
    v=geometry(g,.5)
    assert abs(v['cosine'][0][1]+2**-.5)<1e-12 and v['cosine'][0][2] is None
    assert v['old_new_conflict'] and v['derivative_along_negative_new']==[1.,-2.,0.]
    assert v['derivative_along_negative_full']==[0.,-.5,0.]


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new'];prior=Path(cfg['prior_root'])
    allowed={prior/(name+'.teacher.private.pt'),prior/(name+'.S_new.features.private.pt')}
    model_paths={key:Path(path)/(name+'.HH.private.pt') for key,path in cfg['model_roots'].items()}
    allowed.update(model_paths.values());protected=[prior]+[Path(p) for p in cfg['model_roots'].values()]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if any(p.is_relative_to(r) for r in protected):assert p in allowed,'Support-only frozen-model boundary'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    state=load(prior/(name+'.teacher.private.pt'));bank=load(root/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    current={c:new['x'][new['y']==c] for c in range(old,seen)};x,y,p=training_batch(current,bank);rx,ry,rp=training_batch({},bank)
    assert all(abs(float(p[y==c].sum())-1/seen)<1e-12 for c in range(seen))
    mask=y<old;mass=float(p[mask].sum());assert abs(mass-old/seen)<1e-12
    with torch.no_grad():
        ts=scores(rx,state['teacher'],state['frame'])[:,:old];target=centered(ts);correct=ts.argmax(1)==ry
    results=[]
    for label in lock['states']:
        save(root/(job['id']+'.progress.json'),dict(phase='gradient_'+label,dataset=name))
        frozen=state['teacher'] if label=='teacher' else load(model_paths[label])['model'];params=clone_model(frozen,True)
        ss=scores(x,params,state['frame'])[:,:seen];ce=-ss.log_softmax(1)[torch.arange(len(y)),y]
        old_ce=(p[mask]*ce[mask]).sum()/mass;new_ce=(p[~mask]*ce[~mask]).sum()/(1-mass)
        retention=((centered(scores(rx,params,state['frame'])[:,:old])-target).square().mean(1)*correct*rp).sum()
        grads=[]
        for term in (old_ce,new_ce,retention):
            grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(term,params,retain_graph=term is old_ce)]))
        results.append(dict(state=label,old_CE=float(old_ce.detach()),new_CE=float(new_ce.detach()),retention=float(retention.detach()),old_class_mass=mass,**geometry(grads,mass)))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,states=results,
        primary_original_HH_conflict=next(v['old_new_conflict'] for v in results if v['state']=='original_HH'),
        gradient_terms=['old_group_normalized_full_CE','new_group_normalized_full_CE','original_H_retention'],
        diagnostic_adapter_gradient_evaluations=12,adapter_updates=0,engineering_gradient_evaluations=0,
        old_support_count=len(ry),new_support_count=len(new['y']),query_access=False,E_access=False,old_S_access=False,
        GPU_used=False,new_labels=0,model_updates=0,operation_seconds=time.monotonic()-start,
        scope='Local support CE geometry only; directional derivatives are not finite-step BA or query generalization evidence'))


if __name__=='__main__':main()
