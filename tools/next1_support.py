"""NEXT1 fixed schedules, state restoration and aggregate diagnostics."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from run_prototype_coherent import common_shift, fit_split
from run_prototype_single import save


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def block_steps(arm, task, n):
    if arm == 'F' and task > 1:
        return 0
    if arm in ('U', 'UCB') and task > 1:
        return [138, 33, 33, 34][task-1]
    return math.ceil(n/64)


def batches(loader, arm, task, block):
    target = block_steps(arm, task, len(loader.dataset))
    if not target:
        return
    cycle = 0; emitted = 0
    while emitted < target:
        loader.dataset.epoch = block+cycle*1000
        for batch in loader:
            yield batch
            emitted += 1
            if emitted == target:
                break
        cycle += 1


def fd_loss(distances, labels, counts, balanced):
    if not balanced:
        return distances.mean()
    n = sum(counts.values()); k = len(counts)
    weights = distances.new_tensor([n/(k*counts[int(c)]) for c in labels.tolist()])
    return (weights*distances).mean()


def fixed_split(rows, config):
    membership = json.loads(Path(config['split_file']).read_text())['membership']
    fit = [i for i, r in enumerate(rows) if membership[r['relative_path']] == 'fit']
    meta = [i for i, r in enumerate(rows) if membership[r['relative_path']] == 'meta']
    if not fit or not meta or len(fit)+len(meta) != len(rows):
        raise ValueError('Frozen split coverage mismatch')
    return fit, meta


def make_split(train, path):
    membership = {}; blocks = []
    order = [4,0,3,7,5,6,2,1]
    for task in range(1,5):
        rows = [r for r in train if r['label'] in order[(task-1)*2:task*2]]
        fit, meta = fit_split(rows, 74002+task*100003)
        for i in fit: membership[rows[i]['relative_path']] = 'fit'
        for i in meta: membership[rows[i]['relative_path']] = 'meta'
        blocks.append(dict(task=task, fit_n=len(fit), meta_n=len(meta)))
    assert sum(x['fit_n'] for x in blocks) == 15122 and sum(x['meta_n'] for x in blocks) == 3596
    value = dict(membership=membership, blocks=blocks)
    save(Path(path), value)
    return dict(blocks=blocks, sha256=digest_file(path))


def gradient_diagnostic(fit, fd, parameters):
    torch.cuda.synchronize(); started = time.monotonic()
    a = torch.autograd.grad(fit, parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(10.*fd, parameters, retain_graph=True, allow_unused=True)
    aa = fit.new_zeros(()); bb = aa.clone(); dot = aa.clone()
    for x,y in zip(a,b):
        if x is not None: aa += x.square().sum()
        if y is not None: bb += y.square().sum()
        if x is not None and y is not None: dot += (x*y).sum()
    torch.cuda.synchronize()
    result = dict(fit_gradient_norm=float(aa.sqrt()), weighted_fd_gradient_norm=float(bb.sqrt()),
        cosine=float(dot/(aa*bb).sqrt()) if float(aa*bb)>1e-24 else None)
    return result, time.monotonic()-started


def moment_diagnostic(before, after, y, fit_ids, meta_ids, old_head):
    torch.cuda.synchronize(); started=time.monotonic()
    delta = common_shift(before[fit_ids], after[fit_ids], y[fit_ids])
    change = after-before; residual = change-delta
    classes = sorted(y.unique().tolist()); result = {'classes':{}, 'cross_class':[]}
    basis = None
    if old_head is not None:
        u,s,_ = torch.linalg.svd(old_head.double(), full_matrices=False)
        basis = u[:, s>s.max()*max(old_head.shape)*torch.finfo(torch.float32).eps]
    for c in classes:
        rows=meta_ids[y[meta_ids]==c]; all_rows=y==c
        a,b=before[all_rows],after[all_rows]
        ca=a.T@a/len(a)-torch.outer(a.mean(0),a.mean(0))
        cb=b.T@b/len(b)-torch.outer(b.mean(0),b.mean(0))
        E0=float(change[rows].square().sum(1).mean()); Es=float(residual[rows].square().sum(1).mean())
        result['classes'][str(c)]=dict(meta_n=len(rows),E0=E0,Eshift=Es,
            covariance_change_frobenius=float((cb-ca).norm()),
            old_head_direction_residual=float((residual[rows]@basis).square().sum(1).mean()) if basis is not None else None)
        fit=fit_ids[y[fit_ids]==c]; dc=change[fit].mean(0)
        for other in classes:
            if other==c:continue
            test=meta_ids[y[meta_ids]==other]
            result['cross_class'].append(dict(source=c,target=other,
                E0=float(change[test].square().sum(1).mean()),Eshift=float((change[test]-dc).square().sum(1).mean())))
    E0=np.mean([v['E0'] for v in result['classes'].values()]);Es=np.mean([v['Eshift'] for v in result['classes'].values()])
    result.update(E0=float(E0),Eshift=float(Es),ratio=float(Es/E0) if E0>1e-12 else None)
    torch.cuda.synchronize();return result,time.monotonic()-started


def class_losses(x,y,W,fit_ids,meta_ids):
    loss=.5*(x@W-torch.nn.functional.one_hot(y,W.shape[1])).square().sum(1)
    result={}
    for c in sorted(y.unique().tolist()):
        fit=fit_ids[y[fit_ids]==c];meta=meta_ids[y[meta_ids]==c]
        a=float(loss[fit].mean());b=float(loss[meta].mean())
        result[str(c)]=dict(fit_n=len(fit),meta_n=len(meta),fit_square_loss=a,meta_square_loss=b,meta_minus_fit=b-a)
    return result


def summarize(scores, labels, seen):
    truth=np.array([seen.index(int(c)) for c in labels]);pred=scores.argmax(1);n=len(seen)
    confusion=np.zeros((n,n),dtype=int);np.add.at(confusion,(truth,pred),1)
    old=truth<n-2;new=~old;po=pred<n-2
    result=dict(class_order=seen,confusion=confusion.tolist(),validation_n=len(labels),
        old=dict(n=int(old.sum()),predicted_new=int((old&~po).sum()),wrong_other_old=int((old&po&(truth!=pred)).sum())),
        new=dict(n=int(new.sum()),predicted_old=int((new&po).sum()),wrong_other_new=int((new&~po&(truth!=pred)).sum())))
    if old.any():
        masked=scores[old,:n-2].argmax(1);result['old']['restricted_correct']=int((masked==truth[old]).sum())
    else:result['old']['restricted_correct']=None
    masked=scores[new,n-2:].argmax(1)+n-2;result['new']['restricted_correct']=int((masked==truth[new]).sum())
    result['per_class']={str(c):dict(n=int((truth==i).sum()),correct=int(((truth==i)&(pred==i)).sum()),
        recall=float((pred[truth==i]==i).mean())) for i,c in enumerate(seen)}
    return result


def screening(candidate, control):
    delta=dict(ba=candidate['final_balanced_accuracy']-control['final_balanced_accuracy'],
        tail=candidate['final_tail_recall']-control['final_tail_recall'],
        forgetting=candidate['forgetting']-control['forgetting'])
    return dict(**delta,passed=delta['ba']>=.01 and delta['tail']>=-.005 and delta['forgetting']<=.01)


def self_check():
    torch.set_num_threads(4)
    d=torch.full((5,),2.);y=torch.tensor([0,0,0,0,1]);counts={0:4,1:1}
    assert fd_loss(d,y,counts,False)==2 and fd_loss(d,y,counts,True)==2
    d=torch.tensor([1.,1.,1.,1.,3.]);assert fd_loss(d,y,counts,True)==2
    assert fd_loss(torch.ones(3),torch.zeros(3,dtype=torch.long),counts,True)==.625
    assert [2*block_steps('R',t,n) for t,n in enumerate([8785,1079,238,5020],1)]==[276,34,8,158]
    assert [2*block_steps('U',t,n) for t,n in enumerate([8785,1079,238,5020],1)]==[276,66,66,68]
    assert [2*block_steps('F',t,n) for t,n in enumerate([8785,1079,238,5020],1)]==[276,0,0,0]
    scores=np.array([[2.,0.,3.,0.],[0.,2.,0.,3.],[3.,0.,2.,0.],[0.,3.,0.,2.]])
    z=summarize(scores,np.arange(4),list(range(4)))
    assert z['old']['predicted_new']==2 and z['new']['predicted_old']==2
    assert z['old']['restricted_correct']==2 and z['new']['restricted_correct']==2
    class Dataset(torch.utils.data.Dataset):
        epoch=1
        def __len__(self):return 238
        def __getitem__(self,i):return i,self.epoch
    def loader():return torch.utils.data.DataLoader(Dataset(),batch_size=64,shuffle=True,
        generator=torch.Generator().manual_seed(12))
    native=list(loader());r=list(batches(loader(),'R',3,1))
    assert all(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) for a,b in zip(native,r))
    u=list(batches(loader(),'U',3,1));v=list(batches(loader(),'U',3,1))
    assert len(u)==33 and set(torch.cat([x[0] for x in u]).tolist())==set(range(238))
    assert len(set(torch.cat([x[1] for x in u]).tolist()))==9
    assert all(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) for a,b in zip(u,v))
    state=rng_state();expected=(random.random(),np.random.rand(),torch.rand(1))
    restore_rng(state);actual=(random.random(),np.random.rand(),torch.rand(1))
    assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])
    model=torch.nn.BatchNorm1d(3).eval()
    snapshot={k:v.clone() for k,v in model.state_dict().items()};fingerprint=state_digest(snapshot)
    model(torch.randn(2,3));assert state_digest(model.state_dict())==fingerprint
    restored=torch.nn.BatchNorm1d(3).eval();restored.load_state_dict(snapshot,strict=True)
    assert state_digest(restored.state_dict())==fingerprint
    p=torch.nn.Parameter(torch.ones(1));o=torch.optim.AdamW([p],lr=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(o,2,eta_min=1e-5)
    assert o.param_groups[0]['lr']==.0003
    o.step();scheduler.step();assert abs(o.param_groups[0]['lr']-.000155)<1e-12
    o.step();scheduler.step();assert scheduler.last_epoch==2
    print('PASS: FD scale, schedules, native first cycle, repeated cycle seeds, RNG/full state restoration, scheduler, masked diagnostics')
