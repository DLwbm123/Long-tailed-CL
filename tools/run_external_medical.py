"""Pinned AIR/full APART, fresh training outputs and a sealed four-trajectory evaluation."""
import ast
import copy
import csv
import gc
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import timm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from external_medical_compare import (air_checkpoint, extend_frequency_embedding,
    label_map, load_apart_pretrained, sealed_apart_train, validate_access)
from run_module_cycle import save

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT/'third_party/air_apart_pinned'
VERSION = 'air_apart_medical_native_v1'


def author(name, filename):
    spec = importlib.util.spec_from_file_location(name, PINNED/filename)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def medical_transform(training):
    # Same actual medical transform as the controls, including their .5 normalization.
    ops = [T.Resize((256,256) if training else (224,224),
                    interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
    if training: ops += [T.RandomCrop(224), T.RandomHorizontalFlip(.5)]
    return T.Compose(ops+[T.ToTensor(), T.Normalize((.5,)*3,(.5,)*3)])


def validate(config):
    lock = json.loads(Path(config['protocol_lock']).read_text())
    validate_access(config, 'train')
    ds = lock['data'][config['dataset']];method = config['method']
    if (config.get('authorized_external_comparison') is not True or
            method not in ('AIR','APART_full') or config['version'] != VERSION or
            config['seed'] != lock['seed'] or config['order'] != ds['order'] or
            config['task_sizes'] != ds['task_sizes'] or config['test_accessed'] is not False or
            config['evaluation_split'] != 'development_validation' or
            config['expected_steps'] != (0 if method=='AIR' else ds['APART_total_updates']) or
            config['expected_analytic_batches'] != (ds['AIR_total_analytic_batches'] if method=='AIR' else 0)):
        raise ValueError('External frozen method/data/budget differs')
    out = Path(config['output']).resolve();root = Path(config['campaign_root']).resolve()
    if not out.is_relative_to(root) or out.parent.name not in ('runs','preflight'):
        raise ValueError('Fresh campaign output required')
    return lock, ds


def read_rows(config, split):
    if split not in ('train','val'): raise ValueError('Sealed split')
    if split == 'val': check_evaluation_barrier(config)
    with (Path(config['manifest'])/(split+'.csv')).open() as f: rows=list(csv.DictReader(f))
    mapping=label_map(config['order']);counts={c:0 for c in config['order']}
    for r in rows:
        p=Path(r['relative_path']);original=int(r['original_label'])
        if (r['split']!=split or p.is_absolute() or '..' in p.parts or
                not r['identity_component'] or original not in mapping):
            raise ValueError('Manifest scope/path/label differs')
        r['target']=mapping[original];counts[original]+=1
    expected=json.loads(Path(config['protocol_lock']).read_text())['data'][config['dataset']]
    if counts!={int(c):n for c,n in expected[split+'_counts'].items()}:
        raise ValueError('Frozen per-class support differs')
    if len({r['relative_path'] for r in rows})!=len(rows): raise ValueError('Duplicate images')
    return rows


class MedicalImages(Dataset):
    def __init__(self, rows, root, training, seed):
        self.rows=rows;self.root=Path(root).resolve();self.training=training
        self.transform=medical_transform(training);self.seed=seed;self.epoch=0
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path=(self.root/self.rows[i]['relative_path']).resolve()
        if not path.is_relative_to(self.root): raise ValueError('Image escapes root')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed+self.epoch*1000003+i)
            with Image.open(path) as im: x=self.transform(im.convert('RGB'))
        return i,x,self.rows[i]['target']


def loader(config, rows, task, training, preflight=False):
    ds=MedicalImages(rows,config['images'],training,config['seed']+task*100003)
    workers=0 if preflight or not training else 4
    options=dict(multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=False) if workers else {}
    return DataLoader(ds,batch_size=32 if config['method']=='AIR' else 48,
        shuffle=training,num_workers=workers,drop_last=False,
        generator=torch.Generator().manual_seed(config['seed']+task*2003),**options)


def make_frozen(weight, device):
    state=load_file(weight,device='cpu');core={k:v for k,v in state.items() if not k.startswith('head.')}
    model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    model.load_state_dict(core,strict=True);model.requires_grad_(False).eval()
    return model.to(device),state


class ApartNetwork(nn.Module):
    def __init__(self, backbone, original):
        super().__init__();self.backbone=backbone;self.original_backbone=original
    def forward(self, x, **kwargs):
        with torch.no_grad(): cls=self.original_backbone(x)
        return self.backbone(x,cls_features=cls,**kwargs)


class PreflightDone(Exception): pass


def apart_learner(network, frequencies, init_classes, device, observation):
    source=ast.parse((PINNED/'apart_learner.py').read_text())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Learner')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    class CheckedAdamW(optim.AdamW):
        def step(self, *a, **kw):
            observation.before_step(self)
            result=super().step(*a,**kw);observation.steps+=1;return result
    ns=dict(torch=torch,np=np,nn=nn,F=F,tqdm=tqdm,logging=logging,
            optim=types.SimpleNamespace(AdamW=CheckedAdamW,lr_scheduler=optim.lr_scheduler),
            tensor2numpy=lambda t:t.detach().cpu().numpy(),observe_loss=observation.loss)
    selected=[methods[k] for k in ('get_optimizer','get_scheduler','_init_prompt','_train')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),
                 '<pinned_author_train>', 'exec'),ns)
    train=copy.deepcopy(methods['_init_train'])
    for node in ast.walk(train):
        for field, value in ast.iter_fields(node):
            if not isinstance(value,list): continue
            replaced=[]
            for item in value:
                if (isinstance(item,ast.Expr) and isinstance(item.value,ast.Call) and
                        isinstance(item.value.func,ast.Attribute) and item.value.func.attr=='zero_grad'):
                    replaced.append(ast.Expr(value=ast.Call(func=ast.Name(id='observe_loss',ctx=ast.Load()),
                        args=[ast.Name(id='loss',ctx=ast.Load()),ast.Name(id='epoch',ctx=ast.Load())],keywords=[])))
                replaced.append(item)
            setattr(node,field,replaced)
    ns['_init_train']=sealed_apart_train(train,ns)
    v=types.SimpleNamespace(_network=network,_device=device,init_lr=.003,weight_decay=0.,
        min_lr=1e-5,theta=100,args=dict(tuned_epoch=10,lt_list=frequencies,optimizer='adamw',
        init_cls=init_classes,increment=2,imbalance=True,reinit_optimizer=True,prompt_pool=True,
        shared_prompt_pool=False,shared_prompt_key=False,pull_constraint=True,pull_constraint_coeff=.1))
    for name in ('get_optimizer','get_scheduler','_init_prompt','_init_train','_train'):
        setattr(v,name,types.MethodType(ns[name],v))
    return v


class Observation:
    def __init__(self, budget, state, preflight):
        self.budget=budget;self.state=state;self.preflight=preflight;self.steps=0;self.records=[];self.current={}
    def loss(self, loss, epoch):
        self.budget()
        if not torch.isfinite(loss): raise ValueError('Nonfinite native APART loss')
        if not self.current or self.current['epoch']!=epoch+1:
            self.current=dict(epoch=epoch+1,loss_sum=0.,batches=0,gradient_norm_max=0.,steps=0)
            self.records.append(self.current)
        self.current['loss_sum']+=float(loss.detach());self.current['batches']+=1
    def before_step(self, optimizer):
        self.budget();norm=0.
        for g in optimizer.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise ValueError('Nonfinite native gradient')
                    norm+=float(p.grad.detach().double().square().sum())
        norm=norm**.5
        if not np.isfinite(norm) or norm<=0: raise ValueError('Missing finite backward gradient')
        self.current['gradient_norm_max']=max(self.current['gradient_norm_max'],norm)
        self.current['active_lrs']=[float(g['lr']) for g in optimizer.param_groups]
        if self.preflight: raise PreflightDone()
        self.current['steps']+=1
        if self.steps%20==0: self.state(self.steps+1,self.current['epoch'])


class EpochLoader:
    def __init__(self, data): self.data=data;self.epoch=0
    def __iter__(self):
        self.data.dataset.epoch=self.epoch;self.epoch+=1;return iter(self.data)
    def __len__(self): return len(self.data)


def make_apart(config, device, observation):
    frozen,state=make_frozen(config['weight'],device)
    module=author('pinned_apart_vit','apart_vit.py')
    class Tuning(types.SimpleNamespace):
        def __getitem__(self,key):return getattr(self,key)
    tune=Tuning(ffn_adapt=True,ffn_option='parallel',ffn_adapter_layernorm_option='none',
        ffn_adapter_init_option='lora',ffn_adapter_scalar='0.1',ffn_num=64,d_model=768,
        vpt_on=False,vpt_num=0,batchwise_prompt=True)
    model=module.VisionTransformer(num_classes=len(config['order']),patch_size=16,embed_dim=768,
        depth=12,num_heads=12,mlp_ratio=4,qkv_bias=True,tuning_config=tune,pool_size=5,bs=48)
    load_apart_pretrained(model,state)
    counts=config['data']['train_counts'];frequencies=[counts[str(c)] for c in config['order']]
    extend_frequency_embedding(model.assigner,max(frequencies),config['seed'])
    network=ApartNetwork(model.to(device),frozen)
    return apart_learner(network,frequencies,config['task_sizes'][0],device,observation)


def air_residual(arm):
    alpha=arm.cnt.to(arm.dtype).reciprocal();alpha[torch.isinf(alpha)]=0
    alpha*=len(alpha)/alpha.sum()
    # Same weighted sum as author, without an extra K*D*D audit temporary.
    system=torch.zeros((arm.in_features,arm.in_features),device=arm.A.device,dtype=arm.dtype)
    for c in range(len(alpha)):system.add_(arm.A[c],alpha=float(alpha[c]))
    system.diagonal().add_(arm.gamma);rhs=arm.C*alpha[None,:]
    residual=float((system@arm.weight-rhs).norm()/rhs.norm().clamp_min(1e-30))
    if not np.isfinite(residual) or residual>1e-8 or not torch.isfinite(arm.weight).all():
        raise ValueError('AIR solve residual exceeds tolerance')
    return residual


def checkpoint_metadata(config, task, steps, batches):
    return dict(version=VERSION,method=config['method'],dataset=config['dataset'],order=config['order'],
        seed=config['seed'],source_commit=config['source_commit'],task=task,steps=steps,analytic_batches=batches)


def check_evaluation_barrier(config):
    campaign=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    gate=json.loads(Path(config['evaluation_gate']).read_text())
    if gate['phase']!='EVALUATE' or not gate.get('training_sealed_at'):
        raise ValueError('All-four training barrier not met')
    for job in campaign['jobs']:
        s=json.loads((Path(job['output'])/'STATUS.json').read_text())
        if (s['status'] not in ('TRAINED','RUNNING_EVALUATION','COMPLETE') or
                s['steps']!=job['expected_steps'] or s['analytic_batches']!=job['expected_analytic_batches'] or
                s['stages']!=len(job['task_sizes']) or s['policy_updates']!=0):
            raise ValueError('Missing exact four-trajectory trained budget')


def stage_report(task, seen, targets, predictions, scores, order, counts):
    n={str(c):int((targets==i).sum()) for i,c in enumerate(order) if c in seen}
    recall={str(c):float((predictions[targets==i]==i).mean()) for i,c in enumerate(order) if c in seen}
    if any(v<=0 for v in n.values()):raise ValueError('Missing validation support')
    ranked=sorted(order,key=lambda c:(-counts[str(c)],c));tail=ranked[len(order)//2:]
    f1=[]
    for i,c in enumerate(order):
        if c not in seen:continue
        tp=int(((predictions==i)&(targets==i)).sum());den=int((predictions==i).sum()+(targets==i).sum())
        f1.append(2*tp/den if den else 0.)
    return dict(task=task,seen=seen,validation_n=len(targets),accuracy=float((targets==predictions).mean()),
        balanced_accuracy=float(np.mean(list(recall.values()))),per_class_n=n,per_class_recall=recall,
        tail_recall=float(np.mean([recall[str(c)] for c in tail if c in seen])) if any(c in seen for c in tail) else None,
        macro_F1=float(np.mean(f1)))


def task_blocks(order,sizes):
    blocks=[];at=0
    for n in sizes:blocks.append(order[at:at+n]);at+=n
    if at!=len(order):raise ValueError('Task sizes differ')
    return blocks


def aggregate(reports, tasks, counts):
    final=reports[-1];old=sum(tasks[:-1],[]);new=tasks[-1]
    forgetting=[max(s['per_class_recall'][str(c)] for s in reports[:-1] if c in s['seen'])-
                final['per_class_recall'][str(c)] for c in old]
    return dict(stages=reports,average_incremental_accuracy=float(np.mean([s['accuracy'] for s in reports])),
        final_accuracy=final['accuracy'],average_incremental_balanced_accuracy=float(np.mean([s['balanced_accuracy'] for s in reports])),
        final_balanced_accuracy=final['balanced_accuracy'],final_tail_recall=final['tail_recall'],
        forgetting=float(np.mean(forgetting)),forgetting_classes=len(old),
        final_old_recall=float(np.mean([final['per_class_recall'][str(c)] for c in old])),
        final_new_recall=float(np.mean([final['per_class_recall'][str(c)] for c in new])),
        final_macro_F1=final['macro_F1'],average_incremental_macro_F1=float(np.mean([s['macro_F1'] for s in reports])),
        class_frequency=counts,test_accessed=False,evaluation_split='development_validation',independent_confirmation=False)


def run(config):
    lock,data=validate(config);config=dict(config,data=data);output=Path(config['output'])
    operation=config['operation'];preflight=config.get('preflight',False)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available():
        raise ValueError('GPU01 visible single GPU required')
    torch.set_num_threads(4);random.seed(config['seed']);np.random.seed(config['seed'])
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);device=torch.device('cuda:0')
    def budget():
        if time.time()>=config['original_deadline']:raise TimeoutError('Original round deadline')
    steps=0;batches=0;stages=0;diagnostics=[];max_residual=0.
    def state(status,phase,**kw):
        save(output/'STATUS.json',dict(status=status,phase=phase,steps=steps,analytic_batches=batches,
            stages=stages,policy_updates=0,test_accessed=False,**kw))
    if operation=='train':
        if output.exists():raise ValueError('No overwrite or automatic retry')
        output.mkdir(parents=True);state('RUNNING','initialize')
    elif operation=='evaluate':
        if preflight:raise ValueError('Preflight evaluation forbidden')
        check_evaluation_barrier(config)
        prior=json.loads((output/'STATUS.json').read_text());steps=prior['steps'];batches=prior['analytic_batches'];stages=prior['stages']
        state('RUNNING_EVALUATION','EVALUATE')
    else:raise ValueError('Unknown operation')
    try:
        if operation=='train' and not preflight:
            campaign=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
            gate=json.loads(Path(config['evaluation_gate']).read_text())
            if gate['phase']!='TRAIN':raise ValueError('Four preflights sealed training')
            for job in campaign['preflights']:
                p=json.loads((Path(job['output'])/'PREFLIGHT.json').read_text())
                if p['status']!='PASS' or p['optimizer_updates']!=0 or p['policy_updates']!=0:
                    raise ValueError('Missing four zero-update preflights')
        tasks=task_blocks(config['order'],config['task_sizes'])
        if config['method']=='AIR':
            frozen,_=make_frozen(config['weight'],device)
            arm_class=author('pinned_air_arm','air_analytic.py').GeneralizedARM
            projection=author('pinned_air_buffer','air_buffer.py').RandomBuffer(768,8192,device=device,dtype=torch.float64)
            arm=arm_class(8192,gamma=2000,device=device,dtype=torch.float64)
            model=None
        else:
            def callback(n,epoch):state('RUNNING','TRAIN',task=stages+1,epoch=epoch,live_steps=n)
            observation=Observation(budget,callback,preflight)
            learner=make_apart(config,device,observation);model=learner._network
        if operation=='train':
            train=read_rows(config,'train')
            for task,classes in enumerate(tasks,1):
                budget();known=sum(config['task_sizes'][:task-1]);total=known+len(classes)
                rows=[r for r in train if known<=r['target']<total]
                stream=loader(config,rows,task,config['method']=='APART_full',preflight)
                state('RUNNING','PREFLIGHT' if preflight else 'TRAIN',task=task)
                if config['method']=='AIR':
                    with torch.no_grad():
                        for _,x,y in stream:
                            budget();phi=projection(frozen(x.to(device)));arm.fit(phi,y.to(device));batches+=1
                            if preflight:break
                            if batches%20==0:state('RUNNING','ANALYTIC_FIT',task=task)
                        if preflight:
                            # Exercise the final class-bank allocation and author update temporary peak.
                            k=len(config['order']);missing=k-arm.out_features
                            if missing:
                                arm.A=torch.cat([arm.A,torch.zeros((missing,8192,8192),device=device,dtype=torch.float64)])
                                arm.C=torch.cat([arm.C,torch.zeros((8192,missing),device=device,dtype=torch.float64)],1)
                                arm.cnt=torch.cat([arm.cnt,torch.zeros(missing,device=device,dtype=arm.cnt.dtype)])
                        arm.update();max_residual=max(max_residual,air_residual(arm))
                    if preflight:
                        receipt=dict(status='PASS',optimizer_updates=0,policy_updates=0,formal_analytic_batches=0,
                            shadow_analytic_batches=1,projection_dim=8192,final_class_bank=len(config['order']),
                            max_solve_residual=max_residual,peak_gpu_bytes=torch.cuda.max_memory_allocated(),test_accessed=False)
                        save(output/'PREFLIGHT.json',receipt);steps=0;batches=0;state('COMPLETE','PREFLIGHT');return
                    if arm.cnt.tolist()!=[data['train_counts'][str(c)] for c in config['order'][:total]]:
                        raise ValueError('AIR cumulative author counts differ')
                    checkpoint=dict(metadata=checkpoint_metadata(config,task,0,batches),
                        projection={k:v.cpu() for k,v in projection.state_dict().items()},head=arm.weight.cpu(),cnt=arm.cnt.cpu())
                    diagnostics.append(dict(task=task,analytic_batches=batches,optimizer_updates=0,
                        max_solve_residual=max_residual,observed_train_n=int(arm.cnt.sum()),seen=total))
                else:
                    learner._cur_task=task-1;learner._known_classes=known;learner._total_classes=total
                    before={k:p.detach().cpu().clone() for k,p in model.named_parameters()} if preflight else None
                    observation.records=[];observation.current={}
                    try:learner._train(EpochLoader(stream),None)
                    except PreflightDone:
                        if not preflight:raise
                    if preflight:
                        if any(not torch.equal(before[k],p.detach().cpu()) for k,p in model.named_parameters()):
                            raise ValueError('Zero-update parameters changed')
                        for prefix in ('pool.','pool_few.','assigner.','head.','head_few.'):
                            if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                                       for k,p in model.backbone.named_parameters() if k.startswith(prefix)):
                                raise ValueError('Missing dual-pool/router/head gradient')
                        if any(p.grad is not None for p in model.original_backbone.parameters()):
                            raise ValueError('Frozen backbone received gradient')
                        save(output/'PREFLIGHT.json',dict(status='PASS',optimizer_updates=0,policy_updates=0,
                            parameters_unchanged=True,finite_loss=True,gradient_norm=observation.current['gradient_norm_max'],
                            dual_pool_router_head_gradients=True,peak_gpu_bytes=torch.cuda.max_memory_allocated(),test_accessed=False))
                        state('COMPLETE','PREFLIGHT');return
                    steps=observation.steps
                    expected=sum(t['APART_updates_10epoch'] for t in data['tasks'][:task])
                    if steps!=expected or len(observation.records)!=10:raise ValueError('APART native update budget differs')
                    for d in observation.records:
                        d=dict(d,task=task);d['mean_loss']=d.pop('loss_sum')/d['batches'];diagnostics.append(d)
                    checkpoint=dict(metadata=checkpoint_metadata(config,task,steps,0),
                        model={k:v.detach().cpu() for k,v in model.state_dict().items()})
                torch.save(checkpoint,output/f'stage_{task}.pt');del checkpoint;stages=task
                save(output/'diagnostics.json',diagnostics);state('RUNNING','TASK_TRAINED',task=task)
            if steps!=config['expected_steps'] or batches!=config['expected_analytic_batches']:
                raise ValueError('Final exact budget differs')
            if config['method']=='AIR':torch.save(air_checkpoint(arm),output/'analytic_state.pt')
            save(output/'MODEL_COST.json',dict(trainable_parameters=0 if model is None else sum(p.numel() for p in model.parameters() if p.requires_grad),
                checkpoint_bytes=sum(p.stat().st_size for p in output.glob('*.pt')),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                max_solve_residual=max_residual,formal_updates=steps,analytic_batches=batches))
            state('TRAINED','SEALED',max_solve_residual=max_residual);return
        val=read_rows(config,'val');reports=[]
        for task in range(1,len(tasks)+1):
            budget();checkpoint=torch.load(output/f'stage_{task}.pt',map_location='cpu',weights_only=False)
            meta=checkpoint['metadata'];expected_steps=0 if config['method']=='AIR' else sum(t['APART_updates_10epoch'] for t in data['tasks'][:task])
            expected_batches=sum(t['AIR_batches32'] for t in data['tasks'][:task]) if config['method']=='AIR' else 0
            if meta!=checkpoint_metadata(config,task,expected_steps,expected_batches):raise ValueError('Checkpoint provenance/method/budget differs')
            total=sum(config['task_sizes'][:task]);seen=config['order'][:total]
            if config['method']=='AIR':projection.load_state_dict(checkpoint['projection']);head=checkpoint['head'].to(device)
            else:model.load_state_dict(checkpoint['model'],strict=True);model.eval()
            targets=[];predictions=[]
            with torch.no_grad():
                for _,x,y in loader(config,[r for r in val if r['target']<total],task,False):
                    budget();x=x.to(device)
                    if config['method']=='AIR':score=projection(frozen(x))@head
                    else:
                        o=model(x,task_id=task-1);score=(o['logits']+o['logits_few'])[:,:total]
                    if not torch.isfinite(score).all() or score.shape[1]!=total:raise ValueError('Nonfinite/incorrect seen scores')
                    targets.append(y.numpy());predictions.append(score.argmax(1).cpu().numpy())
            reports.append(stage_report(task,seen,np.concatenate(targets),np.concatenate(predictions),None,config['order'],data['train_counts']))
            del checkpoint;gc.collect()
        metrics=aggregate(reports,tasks,data['train_counts']);metrics.update(dataset=config['dataset'],module=config['method'],version=VERSION)
        save(output/'metrics.json',metrics);state('COMPLETE','EVALUATED')
    except BaseException as exc:
        state('INCOMPLETE',operation,error=str(exc));raise


if __name__=='__main__': run(json.load(sys.stdin))
