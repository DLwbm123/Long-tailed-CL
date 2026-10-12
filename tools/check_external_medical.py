"""CPU production plumbing: exact author call-chain parity, checkpoint and sealed reports."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import types

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from run_external_medical import (Observation, PreflightDone, apart_learner, author, PINNED,
    medical_transform, stage_report, aggregate, task_blocks, checkpoint_metadata,
    check_evaluation_barrier, read_rows, validate)
from run_external_medical_cycle import validate_campaign
from external_medical_compare import air_checkpoint, restore_air, extend_frequency_embedding
from summarize_external_methods import report


def check(config):
    started=time.monotonic();assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(1)
    module=author('fixture_pinned_apart','apart_vit.py')
    class Tiny(nn.Module):
        def __init__(self, classes):
            super().__init__();self.backbone=nn.Module();b=self.backbone
            b.pool=nn.Linear(6,768);b.pool_few=nn.Linear(6,768)
            b.head=nn.Linear(768,classes);b.head_few=nn.Linear(768,classes)
            b.assigner=module.PoolAssigner();extend_frequency_embedding(b.assigner,10529,74002)
            self.original_backbone=nn.Linear(6,768).requires_grad_(False)
        def forward(self,x,**kw):
            b=self.backbone;z=b.pool(x);q=b.pool_few(x)
            with torch.no_grad():cls=self.original_backbone(x)
            return dict(logits=b.head(z),logits_few=b.head_few(q),pool_id=b.assigner(cls,kw['weight'].long()),
                        reduce_sim=z.mean(),reduce_sim_few=q.mean())
    # Oracle executes the pristine _train including its optimizer/scheduler binding behavior.
    source=ast.parse((PINNED/'apart_learner.py').read_text());cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Learner')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    class Progress:
        def __init__(self, values):self.values=values
        def __iter__(self):return iter(self.values)
        def set_description(self,*args):pass
    ns=dict(torch=torch,nn=nn,optim=optim,F=F,np=np,tqdm=Progress,
            logging=types.SimpleNamespace(info=lambda *a:None),tensor2numpy=lambda t:t.detach().numpy())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[methods[k] for k in ('_train','get_optimizer','get_scheduler','_init_prompt','_init_train')],type_ignores=[])),'<pristine_author_oracle>','exec'),ns)
    formal_steps=0
    for ds,data in config['lock']['data'].items():
        torch.manual_seed(74002);a=Tiny(len(data['order']));b=copy.deepcopy(a)
        frequencies=[data['train_counts'][str(c)] for c in data['order']]
        obs=Observation(lambda:None,lambda *a:None,False)
        new=apart_learner(a,frequencies,data['task_sizes'][0],torch.device('cpu'),obs)
        old=types.SimpleNamespace(_network=b,_device=torch.device('cpu'),init_lr=.003,weight_decay=0.,
            min_lr=1e-5,theta=100,args=copy.deepcopy(new.args),_compute_accuracy=lambda *a:(0.,0.,0.))
        for name in methods:
            if name in ns:setattr(old,name,types.MethodType(ns[name],old))
        known=0
        for task,k in enumerate(data['task_sizes']):
            stream=[(torch.arange(2),torch.randn(2,6),torch.tensor([known,known+k-1]))]
            for v in (new,old):v._cur_task=task;v._known_classes=known;v._total_classes=known+k;v._train(stream,None)
            assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
            formal_steps+=10;known+=k
        assert obs.steps==10*len(data['task_sizes'])
        assert all(d['active_lrs']==[.003*.1,.003] for d in obs.records[10:])
        saved=copy.deepcopy(a.state_dict());c=Tiny(len(data['order']));c.load_state_dict(saved,strict=True)
        assert all(torch.equal(c.state_dict()[k],saved[k]) for k in saved)
    torch.manual_seed(74002);model=Tiny(8);before=copy.deepcopy(model.state_dict())
    obs=Observation(lambda:None,lambda *a:None,True);v=apart_learner(model,[458,27,256,3263,2835,44,10529,1306],4,torch.device('cpu'),obs)
    v._cur_task=0;v._known_classes=0;v._total_classes=4
    try:v._train([(torch.arange(2),torch.randn(2,6),torch.tensor([0,3]))],None)
    except PreflightDone:pass
    else:raise AssertionError('Preflight ran optimizer')
    assert obs.steps==0 and all(torch.equal(before[k],model.state_dict()[k]) for k in before)
    assert obs.current['gradient_norm_max']>0
    arm=author('fixture_arm','air_analytic.py').GeneralizedARM(12,gamma=2000,dtype=torch.float64)
    projection=author('fixture_projection','air_buffer.py').RandomBuffer(5,12,dtype=torch.float64)
    z=projection(torch.randn(8,5,dtype=torch.float64));y=torch.tensor([0,0,1,1,2,2,3,3])
    arm.fit(z,y);arm.update();restored=type(arm)(12,gamma=2000,dtype=torch.float64)
    restore_air(restored,air_checkpoint(arm));restored.update();assert torch.equal(arm.weight,restored.weight)
    alpha=arm.cnt.double().reciprocal();alpha*=len(alpha)/alpha.sum()
    expected=torch.linalg.solve(z.T@(z*alpha[y,None])+2000*torch.eye(12),z.T@(F.one_hot(y,4).double()*alpha[y,None]))
    torch.testing.assert_close(arm.weight,expected,rtol=1e-12,atol=1e-12)
    campaign=config['campaign'];validate_campaign(campaign)
    for key,value in [('seed',3),('method','base'),('test_accessed',True),('expected_steps',1)]:
        broken=copy.deepcopy(campaign['jobs'][0]);broken[key]=value
        try:validate(broken)
        except (ValueError,KeyError):pass
        else:raise AssertionError('Unauthorized variant accepted')
    with tempfile.TemporaryDirectory(dir=config['temporary_root']) as tmp:
        root=Path(tmp);p=root/'campaign.json';p.write_text(json.dumps(campaign));old_env=os.environ.get('Q128_CONFIG')
        os.environ['Q128_CONFIG']=str(p);gate=root/'PROGRAM.json';gate.write_text(json.dumps(dict(phase='TRAIN',active={})))
        job=dict(campaign['jobs'][0],evaluation_gate=str(gate))
        try:read_rows(job,'val')
        except ValueError:pass
        else:raise AssertionError('Validation opened before training barrier')
        # Synthetic report uses already-public controls; no new medical evaluation.
        runs=copy.deepcopy(config['controls']['runs']);preflights={}
        for j in campaign['jobs']:
            ds=j['dataset'];method=j['method'];v=copy.deepcopy(runs[ds+'_base']);v['metrics'].update(dataset=ds,module=method)
            d=config['lock']['data'][ds];v['status'].update(status='COMPLETE',steps=j['expected_steps'],
                analytic_batches=j['expected_analytic_batches'],stages=len(d['task_sizes']),policy_updates=0)
            v['model_cost']=dict(max_solve_residual=1e-12)
            v['scalar_diagnostics']=[dict(epoch=e+1) for t in d['task_sizes'] for e in range(10)] if method=='APART_full' else []
            for s in v['metrics']['stages']:s['macro_F1']=s['balanced_accuracy'] # synthetic field, never reported as observed F1
            v['metrics']['final_macro_F1']=v['metrics']['final_balanced_accuracy'];runs[ds+'_'+method]=v
            preflights[ds+'_'+method]=dict(status='PASS',optimizer_updates=0,policy_updates=0,test_accessed=False,
                projection_dim=8192,max_solve_residual=1e-12,parameters_unchanged=True,dual_pool_router_head_gradients=True,gradient_norm=1.)
        costs=[dict(returncode=0,gpu=i%2,residence_seconds=1.) for i in range(12)]
        s=dict(program=dict(status='COMPLETE',active={},failures=[],costs=costs,elapsed_seconds=6.),lock=config['lock'],runs=runs,
            preflights=preflights,round_name='synthetic_fixture',source_commit='fixture',closed_round_cost_ledger={},cpu_checks={})
        result=report(s,root/'report');assert result['status']=='PASS'
        assert len(json.loads((root/'report/RESULTS.json').read_text())['rows'])==12
        bad=copy.deepcopy(s);bad['runs']['HK_AIR']['metrics']['stages'][-1]['per_class_n']['5']=2
        try:report(bad,root/'invalid')
        except ValueError:pass
        else:raise AssertionError('Changed denominator accepted')
        if old_env is None:os.environ.pop('Q128_CONFIG')
        else:os.environ['Q128_CONFIG']=old_env
    # Current augmentation's exact operations, not the preparation's earlier incorrect description.
    assert [type(t).__name__ for t in medical_transform(True).transforms]==['Resize','RandomCrop','RandomHorizontalFlip','ToTensor','Normalize']
    assert medical_transform(False).transforms[-1].mean==(.5,.5,.5)
    return dict(status='PASS',cpu_pipeline_seconds=time.monotonic()-started,gpu_processes=0,formal_updates=0,
        synthetic_optimizer_steps_per_arm=90,medical_images_opened=0,test_accessed=False,
        native_full_train_parameter_equality=True,first_task_cosine_later_constant_exact=True,
        instrumented_native_preflight_zero_update=True,AIR_checkpoint_counts_and_joint_ridge=True,
        APART_checkpoint_roundtrip=True,unknown_method_seed_budget_test_rejected=True,
        four_trained_validation_barrier_rejected_early_access=True,all_six_methods_complete_report_fixture=True,
        changed_class_denominator_rejected=True,medical_transform_exact=True,real_GPU_preflights_passed=0)


if __name__=='__main__':
    config=json.loads(Path(os.environ['Q155_CPU_CONFIG']).read_text());v=check(config)
    Path(config['receipt']).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
