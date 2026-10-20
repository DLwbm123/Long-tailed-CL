"""Independent Torch-autograd stationarity and saved-query audit of one CE head."""
import importlib.util,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run import write


def main(cfg):
    start=time.time();torch.set_num_threads(4);root=Path(cfg['output'])
    seal=json.loads((root/'FIT_SEAL.json').read_text());opt=json.loads((root/'OPTIMIZATION.json').read_text())
    assert seal['heads']==1 and seal['query_opened'] is False and seal['offline_final20'] is True
    assert json.loads((root/'QUALIFICATION.json').read_text())['state']=='PASS'
    src=Path(cfg['cache_run']);ix=np.load(src/'train_index.private.npz');cache=np.load(src/'train768.private.npy',mmap_mode='r');y=ix['labels'];mass=ix['weights']
    old=torch.load(Path(cfg['control_run'])/'heads_5.private.pt',weights_only=True);fac=old['factors'].numpy();alpha=opt['alpha'];w=np.load(root/'head.private.npy')
    assert w.shape==(1537,20) and w.dtype==np.float64 and np.isfinite(w).all() and opt['logit_multiplier']==20
    tw=torch.tensor(w,dtype=torch.float64,requires_grad=True);value=0.
    for lo in range(0,len(y),2048):
        assert time.time()<cfg['hard_deadline_epoch']-60
        hi=min(lo+2048,len(y));raw=np.column_stack([np.asarray(cache[lo:hi],dtype=np.float64),np.ones(hi-lo)])*fac
        loss=(F.cross_entropy(20*(torch.from_numpy(raw)@tw),torch.tensor(y[lo:hi],dtype=torch.long),reduction='none')*torch.tensor(mass[lo:hi])).sum()/20
        value+=float(loss.detach());loss.backward()
    penalty=alpha*.5*(tw*tw).sum();value+=float(penalty.detach());penalty.backward();grad=tw.grad.numpy()
    error=float(np.max(np.abs(grad-np.load(root/'terminal_gradient.private.npy'))));assert error<1e-8 and abs(value-opt['final_objective'])<1e-8
    gmax=float(np.max(np.abs(grad)));converged=gmax<=cfg['raw_gradient_tolerance'];assert converged==opt['converged']
    spec=importlib.util.spec_from_file_location('final20_data',Path(cfg['remote_base'])/'case_hierarchical1_20261018/data.py');data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)
    train,val=data.read_rows(cfg,'train'),data.read_rows(cfg,'val');assert len(train)==68058 and len(val)==300
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    train_rows=[]
    for c in cfg['order']:train_rows.extend(r for r in train if r['label']==c)
    names=sorted({r['group'] for r in train_rows});index={g:i for i,g in enumerate(names)}
    assert np.array_equal(np.array([index[r['group']] for r in train_rows]),ix['groups'])
    assert np.array_equal(np.array([cfg['order'].index(r['label']) for r in train_rows]),y)
    write(root/'QUERY_OPEN.private.json',dict(opened_at=time.time(),new_image_encodes=0));assert time.time()>seal['frozen_at']
    reference=torch.load(Path(cfg['control_run'])/'stage_5.private.pt',weights_only=True);labels=reference['labels'].numpy();assert labels.tolist()==[r['label'] for r in val]
    query=torch.load(Path(cfg['joint_run'])/'query768.private.pt',weights_only=True);pos={c:0 for c in cfg['order']};rows=[]
    for c in labels.tolist():
        j=pos[c];pos[c]+=1;rows.append(np.concatenate([query[c][0][j].numpy(),query[c][1][j].numpy(),[1.]]))
    x=np.stack(rows)*fac
    anchor_scores=x@old['df_matched_joint'].numpy();base_error=float(np.max(np.abs(anchor_scores-reference['scores']['df_matched_joint'].numpy())));assert base_error<1e-8
    scores={a:v.numpy() for a,v in reference['scores'].items()};scores['offline_CE']=x@w
    seen=np.array(cfg['order']);pred={a:seen[v.argmax(1)] for a,v in scores.items()};metrics={}
    support=json.loads((Path(cfg['prior_run'])/'support.json').read_text());ordered=sorted(cfg['order'],key=lambda c:(support[str(c)]['images'],c));bins=dict(tail7=ordered[:7],middle6=ordered[7:13],head7=ordered[13:],initial10=cfg['order'][:10],later10=cfg['order'][10:])
    for arm,p in pred.items():
        counts=[int((labels==c).sum()) for c in seen];assert counts==[15]*20
        correct=[int(((labels==c)&(p==c)).sum()) for c in seen]
        metrics[arm]=dict(BA=sum(correct)/300,correct=sum(correct),counts=counts,correct_by_class=correct,groups={k:sum(correct[cfg['order'].index(c)] for c in cs) for k,cs in bins.items()})
    a=metrics['offline_CE'];p=pred['offline_CE'];q=pred['df_matched_joint'];contrast=dict(rescue=int(((p==labels)&(q!=labels)).sum()),harm=int(((p!=labels)&(q==labels)).sum()),changed=int((p!=q).sum()))
    gates=dict(converged=converged,final_correct=a['correct']>=86,tail7=a['groups']['tail7']>=22,later10=a['groups']['later10']>=38,initial10=a['groups']['initial10']>=44)
    np.savez(root/'predictions.private.npz',labels=labels,scores=scores['offline_CE'])
    write(root/'RESULTS.json',dict(scope='offline_final20_only',seen=cfg['order'],metrics=metrics,contrast=contrast,frequency_bins=bins))
    write(root/'DECISION.json',dict(gates=gates,proceed=all(gates.values()),offline_reference_only=True,not_legal_same_memory_CIL=True))
    write(root/'VERIFICATION.json',dict(state='PASS',new_image_encodes=0,gradient_max_error=error,objective_error=abs(value-opt['final_objective']),terminal_raw_gradient_max=gmax,converged=converged,baseline_score_max_error=base_error,patient_overlap=0,training_gradient_row_evaluations=len(y),query_rows_scored=300,wall_seconds=time.time()-start))
    write(root/'status.private.json',dict(state='COMPLETE',updated=time.time(),proceed=all(gates.values()),delivery_pending=True,offline_reference_only=True))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        write(Path(cfg['output'])/'VERIFY_FAILURE.private.json',dict(updated=time.time(),error=traceback.format_exc()))
        raise
