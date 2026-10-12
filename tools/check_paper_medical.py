"""Synthetic math, routing, replay and production checks, without medical images."""
import copy
import io
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from paper_medical_models import (DatEntry,HaoNet,dat_loss,dat_predict,hao_loss,herding_indices,
    balanced_indices,cluster_accuracy,adapted_features)
from run_paper_medical import make_hao,validate,gradient_norm
from run_external_medical import make_frozen


def check(config):
    began=time.monotonic();torch.set_num_threads(4);torch.manual_seed(74002)
    lock=json.loads(Path(config['protocol_lock']).read_text());synthetic_steps=0
    for ds,d in lock['data'].items():
        at=0;bank=0;dat=[];hao=[]
        for t in d['tasks']:
            size=t['train_n'];dat.append(((size+23)//24)*40);hao.append(((size+bank+31)//32)*100)
            bank+=sum(min(20,d['train_counts'][str(c)]) for c in t['classes']);at+=len(t['classes'])
        assert dat==d['DAT_paper']['updates_per_task'] and hao==d['Hao_paper']['updates_per_task']
        assert at==len(d['order']) and sum(d['train_counts'].values())==d['train_n'] and sum(d['val_counts'].values())==d['val_n']
    counts=torch.tensor([3.,5.,9.]);old=1;logits=torch.randn(6,3);targets=torch.tensor([0,1,2,2,1,0])
    alpha=.5;gamma=torch.tensor([alpha,1.,1.]);v=counts.log()[None,:]-counts[targets].log()[:,None]
    literal=-(logits[torch.arange(6),targets]+gamma[targets].log()-torch.logsumexp(logits+v+gamma.log()[None,:],dim=1)).mean()
    assert torch.allclose(literal,F.cross_entropy(logits+counts.log()+gamma.log(),targets),atol=1e-6)
    tiny=HaoNet(nn.Linear(5,512),2);teacher=copy.deepcopy(tiny).eval().requires_grad_(False);tiny.head.expand(3)
    before=[p.clone() for p in teacher.parameters()];x=torch.randn(6,5)
    loss,parts=hao_loss(tiny,x,targets,counts,2,counts,teacher)
    assert torch.isfinite(loss) and parts['distribution_margin']>=0 and parts['logit_l1_distillation']>=0
    loss.backward();assert gradient_norm(tiny.parameters())>0
    assert all(torch.equal(a,b) for a,b in zip(before,teacher.parameters()))
    loss2,parts2=hao_loss(tiny,x,torch.full((6,),2),counts,2,counts,teacher)
    assert parts2['distribution_margin']==0
    features=torch.tensor([[1.,0.],[1.,1.],[0.,1.]])
    selected=herding_indices(features,20);assert len(selected)==3 and len(set(selected))==3
    target=torch.tensor([0]*3+[1]*6+[2]*2)
    for indices in balanced_indices(target,4,24,74002):assert torch.bincount(target[indices]).tolist()==[8,8,8]
    target=torch.arange(13)
    for indices in balanced_indices(target,2,24,74002):assert len(indices)==13 and len(set(indices))==13
    assert cluster_accuracy(torch.tensor([[0.,0.],[.01,0.],[1.,1.],[1.,1.01]]),torch.tensor([0,0,1,1]),74002)==1
    core,_=make_frozen(config['weight'],torch.device('cpu'));image=torch.randn(2,3,224,224)
    entry=DatEntry([0,1]);key=nn.Parameter(torch.randn(768));original={k:v.clone() for k,v in core.state_dict().items()}
    with torch.no_grad():assert torch.allclose(adapted_features(core,entry.hyper,image),core(image),atol=2e-5,rtol=1e-5)
    y=torch.tensor([0,1]);parameters=list(entry.parameters())+[key];before=[p.clone() for p in parameters]
    loss,parts=dat_loss(core,entry,key,image,y,image.flip(0),y.flip(0),torch.tensor([3,9]))
    loss.backward();assert gradient_norm(parameters)>0 and all(torch.equal(a,b) for a,b in zip(before,parameters))
    assert all(torch.equal(v,original[k]) for k,v in core.state_dict().items())
    assert entry.hyper.up.grad.abs().sum()>0 and key.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in entry.balanced.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in entry.imbalanced.parameters())
    torch.optim.Adam(parameters,lr=.03).step();synthetic_steps+=1
    teacher=copy.deepcopy(entry).requires_grad_(False).eval();entry.expand([2]);y2=torch.tensor([2,2])
    loss,parts=dat_loss(core,entry,key,image,y2,image,y2,torch.tensor([3,9,8]),teacher=teacher)
    assert torch.isfinite(loss) and 'task_distillation' in parts
    loss.backward();assert gradient_norm(entry.parameters())>0 and entry.hyper.down.grad.abs().sum()>0
    entries=[entry,DatEntry([3])];keys=torch.randn(2,768);key_entries=torch.tensor([0,1])
    with torch.no_grad():
        batch,routes=dat_predict(core,entries,keys,key_entries,image,4)
        single=torch.cat([dat_predict(core,entries,keys,key_entries,image[i:i+1],4)[0] for i in range(2)])
    assert torch.allclose(batch,single,atol=2e-5,rtol=1e-5)
    state=entry.state_dict();buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0)
    restored=DatEntry(entry.classes);restored.load_state_dict(torch.load(buffer,weights_only=True))
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in state.items())
    hao=make_hao(config,2,torch.device('cpu'));hao.train();params=list(hao.parameters());before=[p.clone() for p in params]
    loss,_=hao_loss(hao,image,y,torch.tensor([3,9]),0,torch.tensor([3,9]));loss.backward()
    assert gradient_norm(params)>0 and all(torch.equal(a,b) for a,b in zip(before,params))
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in hao.core.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in hao.head.parameters())
    buffer=io.BytesIO();torch.save(hao.state_dict(),buffer);buffer.seek(0)
    restored_hao=make_hao(config,2,torch.device('cpu'));restored_hao.load_state_dict(torch.load(buffer,weights_only=True))
    assert all(torch.equal(v,restored_hao.state_dict()[k]) for k,v in hao.state_dict().items())
    lr_parameter=nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([lr_parameter],lr=.1,momentum=.9,weight_decay=.0005)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[60,80],gamma=.1);trace=[]
    for epoch in range(100):
        trace.append(opt.param_groups[0]['lr']);opt.zero_grad();lr_parameter.square().backward();opt.step();sched.step()
    assert all(abs(lr-(.1 if i<60 else .01 if i<80 else .001))<1e-12 for i,lr in enumerate(trace))
    synthetic_steps+=100
    import tempfile,os
    from run_external_medical import check_evaluation_barrier
    with tempfile.TemporaryDirectory() as tmp:
        gate=Path(tmp)/'gate.json';gate.write_text(json.dumps(dict(phase='TRAIN')))
        campaign=Path(tmp)/'config.json';campaign.write_text(json.dumps(dict(jobs=[])))
        old_env=os.environ.get('Q128_CONFIG');os.environ['Q128_CONFIG']=str(campaign)
        try:
            try:check_evaluation_barrier(dict(evaluation_gate=str(gate)))
            except ValueError:pass
            else:raise AssertionError('Early val barrier accepted')
        finally:
            if old_env is None:os.environ.pop('Q128_CONFIG',None)
            else:os.environ['Q128_CONFIG']=old_env
    c=dict(config,dataset='ISIC',method='DAT_paper',order=lock['data']['ISIC']['order'],task_sizes=lock['data']['ISIC']['task_sizes'],
        seed=74002,expected_steps=sum(lock['data']['ISIC']['DAT_paper']['updates_per_task']),expected_analytic_batches=0,
        test_accessed=False,evaluation_split='development_validation',nonofficial_paper_reimplementation=True,
        epochs=40,batch_size=24,version=lock['version'])
    validate(c)
    for change in [dict(test_accessed=True),dict(seed=5),dict(expected_steps=0),dict(method='AIR'),dict(epochs=20)]:
        try:validate(dict(c,**change))
        except (ValueError,KeyError):pass
        else:raise AssertionError('Unfrozen access accepted')
    from summarize_paper_medical import report,NEW
    import tempfile
    fixture=dict(program=dict(status='COMPLETE',active={},failures=[],costs=[dict(gpu=i%2,returncode=0,residence_seconds=1.) for i in range(12)],elapsed_seconds=1.),
        lock=lock,runs=copy.deepcopy(config['published_controls']),preflights={},closed_round_cost_ledger={},round_name='synthetic_fixture',source_commit='synthetic_fixture',cpu_checks={})
    fixture['resource_comparison']=json.loads((Path(__file__).resolve().parents[1]/'docs/dat_hao_isic_hk2_20261012/REPORT_COMPARISON_SPEC.json').read_text())
    fixture['cost_sources']={m:[] for m in fixture['resource_comparison']['project_methods']+fixture['resource_comparison']['external_methods']}
    for ds,d in lock['data'].items():
        for method in NEW:
            v=copy.deepcopy(fixture['runs'][ds+'_base']);v['status'].update(status='COMPLETE',steps=d[method]['total_updates'],stages=len(d['task_sizes']),policy_updates=0)
            v['metrics'].update(module=method,nonofficial_paper_reimplementation=True,evaluation_split='development_validation')
            v['scalar_diagnostics']=[dict(loss_mean=1.) for _ in range(lock[method]['epochs']*len(d['task_sizes']))]
            fixture['runs'][ds+'_'+method]=v
            fixture['preflights'][ds+'_'+method]=dict(status='PASS',optimizer_updates=0,test_accessed=False,parameters_unchanged=True,gradient_norm=1.)
    with tempfile.TemporaryDirectory() as tmp:
        audit=report(fixture,Path(tmp)/'complete')
        assert audit['formal_complete']==4 and len(json.loads((Path(tmp)/'complete/CONTRIBUTIONS.json').read_text()))==40
        fixture['runs']['ISIC_DAT_paper']['metrics']['stages'][-1]['per_class_n']['0']+=1
        try:report(fixture,Path(tmp)/'bad')
        except ValueError:pass
        else:raise AssertionError('Changed denominator accepted')
    return dict(status='PASS',cpu_pipeline_seconds=time.monotonic()-began,gpu_processes=0,formal_updates=0,
        synthetic_optimizer_steps=synthetic_steps,medical_images_opened=0,test_accessed=False,
        CBC_literal_equation_equivalence=True,actual_unique_replay_cap_and_herding=True,
        balanced_batches_all_current_labels=True,DAT_identity_initial_adapter_matches_core=True,
        DAT_frozen_core_exact=True,DAT_real_12_layer_synthetic_backward=True,DAT_reuse_distillation_and_checkpoint=True,
        DAT_per_sample_route_batch_independent=True,Hao_pretrained_ResNet18_synthetic_backward_zero_update=True,
        old_teacher_frozen_and_no_old_anchor_margin_zero=True,Hao_checkpoint_roundtrip_and_100epoch_schedule=True,four_trained_gate_rejects_early_validation=True,labels_support_budgets_and_sealed_scope=True,
        complete_eight_method_report_and_bad_denominator_rejection=True,model_effectiveness_not_tested=True,real_GPU_preflights_passed=0)


if __name__=='__main__':
    import sys
    print(json.dumps(check(json.load(sys.stdin)),indent=2))
