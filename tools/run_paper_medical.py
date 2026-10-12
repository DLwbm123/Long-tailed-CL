"""Four sealed, nonofficial paper reconstructions, with unchanged prior controls."""
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from paper_medical_models import (VERSION,METHODS,DatEntry,HaoNet,balanced_indices,cluster_accuracy,
    adapted_features,dat_loss,dat_predict,hao_loss,herding_indices)
from run_external_medical import (MedicalImages,make_frozen,read_rows,task_blocks,stage_report,aggregate,
    check_evaluation_barrier)
from run_module_cycle import save


def validate(c):
    lock=json.loads(Path(c['protocol_lock']).read_text());d=lock['data'][c['dataset']]
    if (c['version']!=VERSION or c['method'] not in METHODS or c['seed']!=lock['seed'] or
        c['order']!=d['order'] or c['task_sizes']!=d['task_sizes'] or c['test_accessed'] is not False or
        c['evaluation_split']!='development_validation' or not c.get('nonofficial_paper_reimplementation') or
        c['expected_steps']!=sum(d[c['method']]['updates_per_task']) or c['expected_analytic_batches']!=0 or
        c['epochs']!=lock[c['method']]['epochs'] or c['batch_size']!=lock[c['method']]['batch_size']):
        raise ValueError('Frozen paper reconstruction scope/budget differs')
    if c['method']=='Hao_paper' and not lock['replay_authorized']:raise ValueError('Replay not authorized')
    out=Path(c['output']).resolve();root=Path(c['campaign_root']).resolve()
    if not out.is_relative_to(root) or out.parent.name not in ('runs','preflight'):raise ValueError('Fresh output scope')
    return lock,d


def stream(c,rows,task,training,preflight=False,batch_sampler=None):
    ds=MedicalImages(rows,c['images'],training,c['seed']+task*100003)
    workers=0 if preflight or not training else 4
    extras=dict(multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=False) if workers else {}
    if batch_sampler is not None:return DataLoader(ds,batch_sampler=batch_sampler,num_workers=workers,**extras)
    return DataLoader(ds,batch_size=c['batch_size'],shuffle=training,drop_last=False,num_workers=workers,
        generator=torch.Generator().manual_seed(c['seed']+task*2003),**extras)


class BalancedBatches:
    def __init__(self,targets,batches,size,seed):self.targets=targets;self.batches=batches;self.size=size;self.seed=seed;self.epoch=0
    def __len__(self):return self.batches
    def __iter__(self):return balanced_indices(self.targets,self.batches,self.size,self.seed+self.epoch*2003)


def make_hao(c,classes,device):
    core=resnet18(weights=None);core.load_state_dict(torch.load(c['resnet_weight'],map_location='cpu',weights_only=True),strict=True)
    core.fc=nn.Identity();return HaoNet(core,classes).to(device)


def metadata(c,task,steps):
    return dict(version=VERSION,method=c['method'],dataset=c['dataset'],order=c['order'],seed=c['seed'],
        source_commit=c['source_commit'],task=task,steps=steps,nonofficial_paper_reimplementation=True)


def gradient_norm(parameters):
    total=0.
    for p in parameters:
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite gradient')
            total+=float(p.grad.detach().double().square().sum())
    if not math.isfinite(total) or total<=0:raise ValueError('Missing finite nonzero backward')
    return total**.5


def verify_preflights(c):
    p=json.loads(Path(os.environ['Q128_CONFIG']).read_text());gate=json.loads(Path(c['evaluation_gate']).read_text())
    if gate['phase']!='TRAIN':raise ValueError('Four preflights not sealed')
    for j in p['preflights']:
        r=json.loads((Path(j['output'])/'PREFLIGHT.json').read_text())
        if r['status']!='PASS' or r['optimizer_updates'] or r['test_accessed'] or not r['parameters_unchanged'] or not r['gradient_norm']>0:
            raise ValueError('Missing real zero-update preflight')


def run(c):
    lock,d=validate(c);out=Path(c['output']);device=torch.device('cuda:0');pre=c.get('preflight',False)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available():raise ValueError('Only GPU01')
    torch.set_num_threads(4);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    tasks=task_blocks(c['order'],c['task_sizes']);steps=0;stages=0;records=[];model=None;core=None;entries=[];keys=[];key_entries=[];memory={}
    trainable_parameters=0
    def budget():
        if time.time()>=c['original_deadline']:raise TimeoutError('Original 24h deadline')
    def state(status,phase,**kw):
        save(out/'STATUS.json',dict(status=status,phase=phase,steps=steps,analytic_batches=0,stages=stages,
            policy_updates=0,test_accessed=False,nonofficial_paper_reimplementation=True,**kw))
    if c['operation']=='train':
        if out.exists():raise ValueError('No overwrite/retry')
        out.mkdir(parents=True);state('RUNNING','INITIALIZE')
    elif c['operation']=='evaluate':
        if pre or (out/'metrics.json').exists():raise ValueError('No preflight/repeated evaluation')
        check_evaluation_barrier(c);prior=json.loads((out/'STATUS.json').read_text());steps=prior['steps'];stages=prior['stages']
        state('RUNNING_EVALUATION','EVALUATE')
    else:raise ValueError('Unknown operation')
    try:
        if c['operation']=='train' and not pre:verify_preflights(c)
        if c['method']=='DAT_paper':core,_=make_frozen(c['weight'],device)
        if c['operation']=='evaluate':
            val=read_rows(c,'val');reports=[]
            for task,classes in enumerate(tasks,1):
                budget();seen=sum(c['task_sizes'][:task]);f=out/f'stage_{task}.pt'
                saved=torch.load(f,map_location=device,weights_only=True)
                expected=metadata(c,task,sum(d[c['method']]['updates_per_task'][:task]))
                if saved['metadata']!=expected:raise ValueError('Checkpoint identity/budget differs')
                if c['method']=='Hao_paper':
                    model=make_hao(c,seen,device);model.load_state_dict(saved['model'],strict=True);model.eval()
                else:
                    entries=[]
                    for v in saved['entries']:
                        entry=DatEntry(v['classes']).to(device);entry.load_state_dict(v['state'],strict=True);entry.eval();entries.append(entry)
                    assigned=[v for entry in entries for v in entry.classes]
                    if sorted(assigned)!=list(range(seen)):raise ValueError('Cache class coverage differs')
                    k=saved['keys'];ki=saved['key_entries']
                    if len(k)!=task or len(ki)!=task or int(ki.max())>=len(entries):raise ValueError('Cache key/checkpoint route differs')
                rows=[r for r in val if r['target']<seen];targets=[];predictions=[];route_counts={}
                with torch.no_grad():
                    for _,x,y in stream(c,rows,task,False):
                        budget();x=x.to(device)
                        if model is not None:z,_=model(x)
                        else:
                            z,route=dat_predict(core,entries,k,ki,x,seen)
                            for i in route.tolist():route_counts[str(i)]=route_counts.get(str(i),0)+1
                        if not torch.isfinite(z).all():raise ValueError('Nonfinite predictions')
                        targets.extend(y.tolist());predictions.extend(z.argmax(1).cpu().tolist())
                r=stage_report(task,c['order'][:seen],np.asarray(targets),np.asarray(predictions),None,c['order'],d['train_counts'])
                r['cache_route_counts']=route_counts;reports.append(r)
                state('RUNNING_EVALUATION','EVALUATE',evaluated_stage=task)
                del saved;model=None;entries=[];gc.collect();torch.cuda.empty_cache()
            m=aggregate(reports,tasks,d['train_counts']);m.update(module=c['method'],name=c['dataset'],
                version=VERSION,steps=steps,source_commit=c['source_commit'],nonofficial_paper_reimplementation=True)
            save(out/'metrics.json',m);state('COMPLETE','EVALUATED');return
        train=read_rows(c,'train')
        for task,classes in enumerate(tasks,1):
            budget();old=sum(c['task_sizes'][:task-1]);seen=old+len(classes)
            current=[i for i,r in enumerate(train) if old<=r['target']<seen];current_rows=[train[i] for i in current]
            teacher=None;selected=-1;best_acc=0.
            if c['method']=='DAT_paper':
                for i,entry in enumerate(entries):
                    entry.eval();features=[];targets=[]
                    with torch.no_grad():
                        for _,x,y in stream(c,current_rows,task,False):
                            budget();features.append(adapted_features(core,entry.hyper,x.to(device)).cpu());targets.append(y)
                    acc=cluster_accuracy(torch.cat(features),torch.cat(targets),c['seed']+task)
                    if acc>best_acc:best_acc=acc;selected=i
                    del features,targets
                reused=selected>=0 and best_acc>lock['DAT_paper']['tau']
                if reused:
                    teacher=copy.deepcopy(entries[selected]).eval().requires_grad_(False)
                    entry=copy.deepcopy(entries[selected]).requires_grad_(True);entry.expand(list(range(old,seen)))
                else:
                    selected=len(entries);entry=DatEntry(list(range(old,seen))).to(device)
                entry.train();key=nn.Parameter(torch.randn(768,device=device)*.02)
                parameters=list(entry.parameters())+[key]
                optimizer=torch.optim.Adam(parameters,lr=lock['DAT_paper']['lr'],weight_decay=0.)
                rows=current_rows;data=stream(c,rows,task,True,pre)
                sampler=BalancedBatches(torch.tensor([r['target'] for r in rows]),len(data),c['batch_size'],c['seed']+task*107)
                balanced=stream(c,rows,task,True,pre,sampler)
                counts=torch.tensor([d['train_counts'][str(c['order'][i])] for i in entry.classes],device=device)
            else:
                if model is None:model=make_hao(c,seen,device)
                else:
                    teacher=copy.deepcopy(model).eval().requires_grad_(False);model.head.expand(seen)
                model.train();parameters=list(model.parameters())
                optimizer=torch.optim.SGD(parameters,lr=lock['Hao_paper']['lr'],momentum=.9,weight_decay=.0005)
                scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60,80],gamma=.1)
                replay=[i for indices in memory.values() for i in indices]
                if len(set(replay))!=len(replay) or any(train[i]['target']>=old for i in replay):raise ValueError('Old-only unique replay')
                rows=current_rows+[train[i] for i in replay];data=stream(c,rows,task,True,pre)
                counts=torch.bincount(torch.tensor([r['target'] for r in rows]),minlength=seen).to(device)
                expected=torch.tensor([min(20,d['train_counts'][str(c['order'][i])]) if i<old else d['train_counts'][str(c['order'][i])] for i in range(seen)],device=device)
                if not torch.equal(counts,expected):raise ValueError('Frozen actual old/new counts differ')
                inherent=torch.tensor([d['train_counts'][str(i)] for i in c['order'][:seen]],device=device)
            if len(data)*c['epochs']!=d[c['method']]['updates_per_task'][task-1]:raise ValueError('Actual loader budget differs')
            before={str(i):p.detach().clone() for i,p in enumerate(parameters)} if pre else None
            if c['method']=='DAT_paper':
                frozen_before={k:v.detach().clone() for k,v in core.state_dict().items()} if pre else None
            trainable_parameters=max(trainable_parameters,sum(p.numel() for p in parameters))
            for epoch in range(c['epochs']):
                budget();data.dataset.epoch=epoch
                if c['method']=='DAT_paper':sampler.epoch=epoch;balanced.dataset.epoch=epoch;balanced_iterator=iter(balanced)
                scalars={};loss_sum=0.;norm_max=0.;nb=0
                state('RUNNING','PREFLIGHT' if pre else 'TRAIN',task=task,epoch=epoch+1)
                for _,x,y in data:
                    budget();x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
                    if c['method']=='DAT_paper':
                        _,xb,yb=next(balanced_iterator)
                        loss,pieces=dat_loss(core,entry,key,x,y,xb.to(device),yb.to(device),counts,teacher=teacher)
                    else:loss,pieces=hao_loss(model,x,y,counts,old,inherent,teacher=teacher)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();norm=gradient_norm(parameters)
                    if pre:
                        unchanged=all(torch.equal(p,before[str(i)]) for i,p in enumerate(parameters))
                        frozen=core is None or all(torch.equal(v,frozen_before[k]) for k,v in core.state_dict().items())
                        if not unchanged or not frozen:raise ValueError('Zero-update/core invariance failed')
                        if c['method']=='DAT_paper':
                            groups=dict(hyper_up=[entry.hyper.up],key=[key],balanced_head=list(entry.balanced.parameters()),imbalanced_head=list(entry.imbalanced.parameters()))
                        else:groups=dict(backbone=list(model.core.parameters()),cosine_head=list(model.head.parameters()))
                        group_gradients={name:any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in group) for name,group in groups.items()}
                        if not all(group_gradients.values()):raise ValueError('Missing head/key/hyper/backbone gradient')
                        receipt=dict(status='PASS',method=c['method'],optimizer_updates=0,policy_updates=0,test_accessed=False,
                            parameters_unchanged=unchanged,pretrained_core_frozen=core is not None,pretrained_core_unchanged=frozen if core is not None else None,gradient_norm=norm,
                            loss=float(loss.detach()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                            required_gradient_groups_nonzero=all(group_gradients.values()),gradient_groups=group_gradients,shadow_batch_only=True,bn_buffers_shadow_only=c['method']=='Hao_paper')
                        save(out/'PREFLIGHT.json',receipt);state('COMPLETE','ZERO_UPDATE_PREFLIGHT');return
                    optimizer.step();steps+=1;nb+=1;loss_sum+=float(loss.detach());norm_max=max(norm_max,norm)
                    for name,value in pieces.items():scalars[name]=scalars.get(name,0.)+float(value.detach())
                    if steps%20==0:state('RUNNING','TRAIN',task=task,epoch=epoch+1,live_steps=steps)
                record=dict(task=task,epoch=epoch+1,steps=steps,batches=nb,loss_mean=loss_sum/nb,gradient_norm_max=norm_max,
                    learning_rate=optimizer.param_groups[0]['lr'],**{k:v/nb for k,v in scalars.items()})
                if c['method']=='DAT_paper':record.update(cache_reused=reused,cluster_accuracy=best_acc,cache_entries_after_task=selected+1 if not reused else len(entries),balanced_batch_size=len(classes)*(c['batch_size']//len(classes)))
                else:record.update(replay_images=len(replay),memory_classes=len(memory),classifier_scale=float(model.head.scale.detach()))
                records.append(record);save(out/'diagnostics.json',records)
                if c['method']=='Hao_paper':scheduler.step()
            budget()
            if c['method']=='DAT_paper':
                entry.eval()
                if selected==len(entries):entries.append(entry)
                else:entries[selected]=entry
                keys.append(key.detach());key_entries.append(selected)
                payload=dict(entries=[dict(classes=v.classes,state=v.state_dict()) for v in entries],
                    keys=torch.stack(keys),key_entries=torch.tensor(key_entries,device=device))
            else:
                model.eval()
                for target in range(old,seen):
                    indices=[i for i in current if train[i]['target']==target];features=[]
                    with torch.no_grad():
                        for _,x,_ in stream(c,[train[i] for i in indices],task,False):
                            budget();_,z=model(x.to(device));features.append(z.cpu())
                    memory[target]=[indices[i] for i in herding_indices(torch.cat(features),20)]
                expected_memory={i:min(20,d['train_counts'][str(c['order'][i])]) for i in range(seen)}
                if {k:len(v) for k,v in memory.items()}!=expected_memory:raise ValueError('Unique herding bank differs')
                payload=dict(model=model.state_dict(),memory=memory)
            components=[model] if model is not None else [core,*entries]
            if any(not torch.isfinite(v).all() for component in components for v in component.state_dict().values() if v.is_floating_point()) or (keys and not torch.isfinite(torch.stack(keys)).all()):raise ValueError('Nonfinite checkpoint state')
            stages=task;payload['metadata']=metadata(c,task,steps)
            torch.save(payload,out/f'stage_{task}.pt');state('RUNNING','TASK_TRAINED',task=task)
            del data,optimizer,parameters,payload,teacher
            if c['method']=='DAT_paper':del balanced,balanced_iterator
            gc.collect();torch.cuda.empty_cache()
        if steps!=c['expected_steps'] or stages!=len(tasks):raise ValueError('Formal budget incomplete')
        final_parameters=(sum(p.numel() for p in model.parameters()) if model is not None else sum(p.numel() for component in [core,*entries] for p in component.parameters())+len(keys)*768)
        save(out/'MODEL_COST.json',dict(model_parameters_final=final_parameters,trainable_parameters_peak=trainable_parameters,
            checkpoint_bytes=sum(f.stat().st_size for f in out.glob('stage_*.pt')),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            training_steps=steps,replay_image_bank_max=sum(len(v) for v in memory.values()),nonofficial_paper_reimplementation=True))
        state('TRAINED','TRAINED')
    except BaseException as exc:state('FAILED','FAILED',error=str(exc));raise


if __name__=='__main__':run(json.load(sys.stdin))
