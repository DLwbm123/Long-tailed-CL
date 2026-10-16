"""Fixed reverse-query-role diagnostic for node weighting and bias calibration."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT, clone_model, scores
from query_node_weights import rollout, query_loss, project_direction, select
from run_query_bias_control import fit_bias
from run_real_query_node import load, metrics, save


def main():
    root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());prior=Path(cfg['prior_root']);diagnostic=Path(cfg['diagnostic_root'])
    lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(2);torch.set_default_dtype(DT);sealed=False
    allowed={name+'.'+s for s in ('teacher.private.pt','baseline.private.pt','S_new.features.private.pt','Qgate.features.private.pt')}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in allowed or (sealed and p.name==name+'.Qfit.features.private.pt'),'Reverse-query role boundary'
        if p.is_relative_to(diagnostic):assert p.name==name+'_half.json','Only aggregate reference permitted'
    sys.addaudithook(audit)
    for suffix in ('E.features.private.pt','S_old.features.private.pt','Qfit.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Pre-seal role guard failed')
    begin=time.monotonic();save(root/(job['id']+'.progress.json'),dict(phase='reverse_query_hypergradient',dataset=name))
    state=load(prior/(name+'.teacher.private.pt'));base=load(prior/(name+'.baseline.private.pt'))
    new=load(prior/(name+'.S_new.features.private.pt'));fit=load(prior/(name+'.Qgate.features.private.pt'))
    current={c:new['x'][new['y']==c] for c in range(old,seen)};a=torch.zeros(old,16,dtype=DT,requires_grad=True)
    model=rollout(state['teacher'],current,state['memory'],state['frame'],a,graph=True)
    parity=max(float((v-w).detach().abs().max()) for v,w in zip(model,base['model']));assert parity<1e-10
    losses=[query_loss(model,state['frame'],fit['x'],fit['y'],cs,seen) for cs in (range(old),range(old,seen))]
    go=torch.autograd.grad(losses[0],a,retain_graph=True)[0];gn=torch.autograd.grad(losses[1],a)[0]
    d,info=project_direction(go,gn,old,seen);d=torch.from_numpy(d.reshape(old,16));del model,losses,go,gn
    save(root/(job['id']+'.progress.json'),dict(phase='reverse_query_candidates',dataset=name))
    node_model=clone_model(rollout(state['teacher'],current,state['memory'],state['frame'],d)) if bool(d.any()) else clone_model(base['model'])
    node=dict(model=node_model,frame=state['frame'])
    with torch.no_grad():logits=scores(fit['x'],base['model'],base['frame'])[:,:seen]
    b,trace=fit_bias(logits,fit['y'],lock['bias_steps'],lock['learning_rate']);bias=dict(base,bias=b)
    torch.save(node,root/(name+'.node.private.pt'));torch.save(bias,root/(name+'.bias.private.pt'))
    torch.save(dict(logit_change=d),root/(name+'.direction.private.pt'));sealed=True;sealed_at=time.time()
    ref=json.loads((diagnostic/(name+'_half.json')).read_text());rows=[];summary=[]
    for phase,data,source_role in [('fit',fit,'Qgate'),('gate',load(prior/(name+'.Qfit.features.private.pt')),'Qfit')]:
        reference=[dict(r,query=phase) for r in ref['class_rows'] if r['query']==source_role and r['model']=='baseline']
        rows.extend(reference)
        for mode,obj in [('node',node),('bias',bias)]:
            _,rr,_=metrics(obj,data,old,seen);rows.extend(dict(dataset=name,query=phase,model=mode,**r) for r in rr)
        for mode in ('baseline','node','bias'):
            for group in ('all','old','new'):
                rr=[r for r in rows if r['query']==phase and r['model']==mode and (group=='all' or r['group']==group)]
                summary.append(dict(dataset=name,query=phase,source_role=source_role,model=mode,group=group,
                    BA=sum(r['recall'] for r in rr)/len(rr),CE=sum(r['CE'] for r in rr)/len(rr)))
    zero_error=abs(trace[0]-next(r['CE'] for r in summary if r['query']=='fit' and r['model']=='baseline' and r['group']=='all'));assert zero_error<1e-10
    get=lambda mode:{r['group']:r['BA'] for r in summary if r['query']=='gate' and r['model']==mode}
    gates={mode:select(get('baseline'),get(mode)) for mode in ('node','bias')}
    result=dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,
        baseline_graph_replay_updates=48,node_adapter_updates=48 if bool(d.any()) else 0,bias_updates=lock['bias_steps'],
        baseline_replay_max_error=parity,zero_bias_CE_error=zero_error,direction=info,bias=b.tolist(),objective_trace=trace,
        hypergradient_backward_sweeps=2,E_access=False,old_S_access=False,new_labels=0,GPU_used=False,
        roles=dict(fit='previous Qgate',gate='previous Qfit'),query_exposure='Both already exposed; reverse-role development diagnostic only',
        gates=gates,forward_gates=lock['forward_gates'][name],
        both_directions_pass={m:gates[m]['selected']=='candidate' and lock['forward_gates'][name][m]['selected']=='candidate' for m in gates},
        class_rows=rows,summaries=summary,
        operation_seconds=time.monotonic()-begin)
    save(root/(job['id']+'.json'),result)


if __name__=='__main__':main()
