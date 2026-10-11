"""CPU compatibility of pinned author heads, routing, sealed training and cached ViT weights."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from safetensors.torch import load_file
import timm
from tqdm import tqdm

from external_medical_compare import (air_checkpoint, restore_air, label_map,
    validate_access, extend_frequency_embedding, load_apart_pretrained, sealed_apart_train)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(config):
    started = time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    torch.manual_seed(74002)
    cache = Path(config['author_cache'])
    arm_module = load_module('author_arm', cache/'AIR/analytic/AnalyticLinear.py')
    buffer_module = load_module('author_buffer', cache/'AIR/analytic/Buffer.py')
    arm = arm_module.GeneralizedARM(12, gamma=2000, device='cpu', dtype=torch.float64)
    projection = buffer_module.RandomBuffer(5, 12, device='cpu', dtype=torch.float64)
    x = torch.randn(19, 5, dtype=torch.float64)
    y = torch.tensor([0]*7+[1]*3+[2]*2+[3]*4+[0]*2+[2])
    phi = projection(x)
    for end in (10, 12, 19):
        begin = 0 if end == 10 else (10 if end == 12 else 12)
        arm.fit(phi[begin:end], y[begin:end]);arm.update()
        counts = torch.bincount(y[:end], minlength=arm.out_features).double()
        alpha = counts.reciprocal();alpha *= len(counts)/alpha.sum()
        row_weights = alpha[y[:end]]
        onehot = F.one_hot(y[:end], arm.out_features).double()
        system = phi[:end].T @ (phi[:end]*row_weights[:, None]) + 2000*torch.eye(12)
        rhs = phi[:end].T @ (onehot*row_weights[:, None])
        expected = torch.linalg.solve(system, rhs)
        torch.testing.assert_close(arm.weight, expected, rtol=1e-12, atol=1e-12)
        assert torch.equal(arm.cnt.long(), counts.long())
    checkpoint = air_checkpoint(arm)
    restored = arm_module.GeneralizedARM(12, gamma=2000, dtype=torch.float64)
    restore_air(restored, checkpoint)
    restored.update();torch.testing.assert_close(restored.weight, arm.weight, rtol=0, atol=0)
    assert not list(projection.parameters()) and not list(arm.parameters())
    assert 'cnt' not in arm.state_dict()
    relative_residual = float((system@arm.weight-rhs).norm()/rhs.norm())

    apart = load_module('author_apart_vit', cache/'APART/backbone/vision_transformer_adapter_pool_a.py')
    source = ast.parse((cache/'APART/models/apart.py').read_text())
    author_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'Learner')
    methods = {n.name:n for n in author_class.body if isinstance(n, ast.FunctionDef)}
    namespace = dict(torch=torch,np=np,nn=nn,optim=optim,F=F,tqdm=lambda x:tqdm(x,disable=True),
                     tensor2numpy=lambda t:t.detach().numpy(),logging=types.SimpleNamespace(info=lambda *a:None))
    selected = [methods[k] for k in ('get_optimizer','get_scheduler','_init_prompt','_init_train')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])), '<author_apart_methods>', 'exec'), namespace)
    author_train = namespace['_init_train']
    train = sealed_apart_train(methods['_init_train'], namespace)

    class TinyNetwork(nn.Module):
        def __init__(self, classes, max_count):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.pool = nn.Linear(6, 768)
            self.backbone.pool_few = nn.Linear(6, 768)
            self.backbone.head = nn.Linear(768, classes)
            self.backbone.head_few = nn.Linear(768, classes)
            self.backbone.assigner = apart.PoolAssigner()
            extend_frequency_embedding(self.backbone.assigner, max_count, 74002)
            self.original_backbone = nn.Linear(6,768).requires_grad_(False)
        def forward(self, x, **kwargs):
            b=self.backbone;z=b.pool(x);zf=b.pool_few(x)
            return dict(logits=b.head(z),logits_few=b.head_few(zf),
                pool_id=b.assigner(self.original_backbone(x),kwargs['weight'].long()),
                reduce_sim=z.mean(),reduce_sim_few=zf.mean())

    synthetic_steps = 0
    for name, audit in config['data_audit'].items():
        order=audit['order'];mapping=label_map(order)
        assert [mapping[c] for c in order] == list(range(len(order)))
        frequencies=[audit['train_counts'][str(c)] for c in order]
        net=TinyNetwork(len(order),max(frequencies))
        learner=types.SimpleNamespace(_network=net,init_lr=.003,weight_decay=0.,min_lr=1e-5,
            theta=100,_device=torch.device('cpu'),args=dict(tuned_epoch=10,lt_list=frequencies,
            optimizer='adamw',init_cls=audit['task_sizes'][0],increment=2,prompt_pool=True,
            shared_prompt_pool=False,shared_prompt_key=False,pull_constraint=True,pull_constraint_coeff=.1))
        learner._compute_accuracy=lambda *args: (_ for _ in ()).throw(AssertionError('Val remains sealed'))
        known=0
        for task,k in enumerate(audit['task_sizes']):
            learner._cur_task=task;learner._known_classes=known;learner._total_classes=known+k
            data=[(torch.arange(2),torch.randn(2,6),torch.tensor([known,known+k-1]))]
            optimizer=namespace['get_optimizer'](learner)
            assert all(abs(g['lr']-target)<1e-15 for g,target in zip(optimizer.param_groups,[.0003,.003]))
            assert all(g['weight_decay']==.01 for g in optimizer.param_groups)
            scheduler=namespace['get_scheduler'](learner,optimizer)
            assert isinstance(scheduler,torch.optim.lr_scheduler.CosineAnnealingLR)
            train(learner,data,None,optimizer,scheduler)
            assert all(torch.isfinite(p).all() for p in net.parameters())
            synthetic_steps+=10;known+=k
        # The exact author training loop and sealed adaptation yield equal tensors.
        torch.manual_seed(71);a=TinyNetwork(len(order),max(frequencies))
        b=TinyNetwork(len(order),max(frequencies));b.load_state_dict(a.state_dict())
        learner._cur_task=0;learner._known_classes=0;learner._total_classes=audit['task_sizes'][0]
        learner._compute_accuracy=lambda *args:(0.,0.,0.) # Synthetic author-loop fixture only.
        data=[(torch.arange(2),torch.randn(2,6),torch.tensor([0,audit['task_sizes'][0]-1]))]
        for model,loop in [(a,author_train),(b,train)]:
            learner._network=model
            opt=namespace['get_optimizer'](learner);sched=namespace['get_scheduler'](learner,opt)
            loop(learner,data,None,opt,sched);synthetic_steps+=10
        assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())

    state=load_file(config['weight'],device='cpu')
    frozen=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    core={k:v for k,v in state.items() if not k.startswith('head.')}
    frozen.load_state_dict(core,strict=True);frozen.requires_grad_(False).eval()
    class Tuning(types.SimpleNamespace):
        def __getitem__(self,key):return getattr(self,key)
    tune=Tuning(ffn_adapt=True,ffn_option='parallel',ffn_adapter_layernorm_option='none',
        ffn_adapter_init_option='lora',ffn_adapter_scalar='0.1',ffn_num=64,d_model=768,vpt_on=False,vpt_num=0,
        batchwise_prompt=True)
    model=apart.VisionTransformer(num_classes=23,patch_size=16,embed_dim=768,depth=12,num_heads=12,
        mlp_ratio=4,qkv_bias=True,tuning_config=tune,pool_size=5,bs=48)
    core_tensors=load_apart_pretrained(model,state)
    old_embedding=model.assigner.cls_emb.weight.detach().clone()
    extend_frequency_embedding(model.assigner,10529,74002)
    assert torch.equal(model.assigner.cls_emb.weight[:501],old_embedding)
    image=torch.randn(2,3,224,224)
    with torch.no_grad():cls=frozen(image)
    output=model(image,cls_features=cls,train=True,weight=torch.tensor([10529,3]))
    loss=F.cross_entropy(output['logits'][:,:13],torch.tensor([0,12]))
    loss+=F.cross_entropy(output['logits_few'][:,:13],torch.tensor([0,12]))
    loss+=output['pool_id'].sum()-.1*(output['reduce_sim']+output['reduce_sim_few'])
    loss.backward()
    for prefix in ['pool.','pool_few.','assigner.','head.','head_few.']:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for key,p in model.named_parameters() if key.startswith(prefix))
    assert all(p.grad is None for key,p in model.named_parameters() if key in core or key.startswith('blocks.'))
    with torch.no_grad():
        model.eval();out=model(image,cls_features=cls)
        assert out['logits'].shape==out['logits_few'].shape==(2,23)
        assert torch.isfinite(out['logits']+out['logits_few']).all()

    for name in ('ISIC','HK'):
        cfg=dict(dataset=name,test_accessed=False,all_four_trained=False)
        validate_access(cfg,'train')
        for split in ('test','reserved','development_validation'):
            try:validate_access(cfg,split)
            except ValueError:pass
            else:raise AssertionError('Sealed split accepted')
        receipts={r:({'status':'DELIVERED','nas_saved':True,'anonymous_http_status':200,'commit':'fixture'},)*2
                  for r in ('module_drift10_20261011','rfmid_contrast4_20261011')}
        validate_access(cfg,'train',receipts)
        receipts['rfmid_contrast4_20261011']=({'status':'RUNNING'},)*2
        try:validate_access(cfg,'train',receipts)
        except ValueError:pass
        else:raise AssertionError('Protected round barrier bypassed')
    return dict(status='PASS',cpu_pipeline_seconds=time.monotonic()-started,formal_adapter_updates=0,
        gpu_processes=0,test_accessed=False,medical_images_opened=0,synthetic_optimizer_steps=synthetic_steps,
        AIR_author_weighted_joint_equivalence=True,AIR_count_checkpoint_roundtrip=True,
        AIR_fixed_random_projection=True,AIR_fixture_projection_dim=12,AIR_formal_projection_dim=8192,
        AIR_max_relative_residual=relative_residual,APART_exact_author_loss_optimizer_scheduler=True,
        APART_sealed_author_loop_parameter_equality=True,APART_author_full_ViT_CPU_forward_backward=True,
        APART_both_pools_assigner_heads_nonzero_finite_gradients=True,APART_pretrained_core_tensors=core_tensors,
        APART_pretrained_core_exact=True,APART_frequency_capacity=10530,
        APART_old_embedding_rows_exact=True,label_mapping_and_budgets=True,
        sealed_test_validation_and_protected_delivery_rejection=True,
        real_GPU_preflights_passed=0,actual_8192_ARM_memory_and_GPU_solve_not_checked=True)


if __name__=='__main__':
    config=json.loads(Path(os.environ['Q154_CONFIG']).read_text())
    result=check(config)
    Path(config['cpu_receipt']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
