"""V2 gated training and sealed evaluation. Configuration is supplied by a neutral launcher."""
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/APART'))
from models.apart import Learner
from utils.medical_v2 import extend_embedding,effective_optimizer,effective_scheduler,headnorm_alpha,WEIGHT_SHA

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def network_hash(network):
    h=hashlib.sha256()
    for k,v in sorted(network.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def transform(train):
    ops=[T.Resize((256,256) if train else (224,224),interpolation=T.InterpolationMode.BICUBIC,antialias=True)]
    if train:ops += [T.RandomCrop(224),T.RandomHorizontalFlip(.5)]
    return T.Compose(ops+[T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])

class Images(Dataset):
    def __init__(self,manifest,image_root,order,classes,train,smoke=False):
        with open(manifest,newline='') as f:source=list(csv.DictReader(f))
        mapping={int(c):i for i,c in enumerate(order)}
        self.rows=[dict(r,target=mapping[int(r['original_label'])]) for r in source if mapping[int(r['original_label'])] in classes]
        assert all(r['split']=='train' for r in self.rows) if train else True
        if smoke:
            # P1 only: retain at least two records per current class, then fill up to two batches.
            selected=[]
            for c in classes:selected += [r for r in self.rows if r['target']==c][:2]
            ids={r['sample_id'] for r in selected}
            selected += [r for r in self.rows if r['sample_id'] not in ids][:96-len(selected)]
            self.rows=selected
        self.root=Path(image_root);self.transform=transform(train)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.root/r['relative_path']) as im:x=self.transform(im.convert('RGB'))
        return i,x,r['target']

def args_for(config,seed,branch,smoke):
    a=json.loads((ROOT/'third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1993_gpu0.json').read_text())
    summary=json.loads((Path(config['protocol'])/'V2_DATASET_SUMMARY.json').read_text())
    order=list(config['class_orders'][str(seed)]) if 'class_orders' in config else np.random.default_rng(seed).permutation(np.arange(8)).tolist()
    counts={r['original_label']:r['n_train_images'] for r in summary['class_counts']}
    a.update(nb_classes=8,nb_tasks=3,init_cls=4,increment=2,seed=seed,device=[torch.device('cuda:0')],
             medical_v2=True,locked_weight_path=config['weight'],tuned_epoch=1 if smoke else 10,
             concm_stage1=branch=='C',concm_stage1_eval_calibration=False,calibration_rule='none',
             lt_list=[counts[c] for c in order],dataset=summary['protocol_id'],weight_decay=.01,
             optimizer_profile='legacy_effective_v1',scheduler='S0_cosine_S1_S2_none',save_task_checkpoints=False,
             class_order=order,split_id='split1',actual_training_imbalance=summary['train_image_imbalance_ratio'],
             data_loader_seed=seed,embedding_seed=seed+1000003,synthesis_seed=seed+2000003,
             memory_statistics_seed_rule='train_seed * 1000 + session')
    for key in ('longtail','order','task_checkpoint_dir'):a.pop(key,None)
    return a,order

class MedicalLearner(Learner):
    def __init__(self,args,config,order,output):
        super().__init__(args)
        extend_embedding(self._network.backbone.assigner,args['seed'])
        self.config=config;self.order=order;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        summary=json.loads((Path(config['protocol'])/'V2_DATASET_SUMMARY.json').read_text())
        self.manifest_hashes={s:sha(Path(config['protocol'])/(s+'.csv')) for s in ('train','val','test')}
        assert self.manifest_hashes==summary['manifest_sha256'],'BLOCKED_MANIFEST_DRIFT'
        self.protocol_hash=config.get('protocol_sha256',sha(Path(config['protocol'])/'V2_DATASET_SUMMARY.json'))
        self.loader_generator=torch.Generator().manual_seed(args['data_loader_seed'])
        # Fork only the device synthesis stream; preserve reference sampling and distributions.
        self.synth_rng=torch.Generator(device=self._device).manual_seed(args['synthesis_seed']).get_state()
        self.records=[];self.epoch_start=time.monotonic();self.gradient_rows=set();self.input_hashes=[];self.pool_grad=False
    def _v2_input_hook(self,epoch,batch,inputs,targets):
        if self.args['tuned_epoch']==1:
            h=hashlib.sha256(inputs.cpu().numpy().tobytes()+targets.cpu().numpy().tobytes()).hexdigest()
            self.input_hashes.append(dict(session=self._cur_task,epoch=epoch,batch=batch,sha256=h))
    def _concm_stage1_sample_memory(self,*args,**kwargs):
        with torch.random.fork_rng(devices=[0]):
            torch.cuda.set_rng_state(self.synth_rng,0)
            result=super()._concm_stage1_sample_memory(*args,**kwargs)
            self.synth_rng=torch.cuda.get_rng_state(0)
        return result
    def _v2_gradient_hook(self,epoch,batch,loss):
        assert torch.isfinite(loss), 'BLOCKED_NONFINITE_LOSS'
        for p in self._network.backbone.parameters():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),'BLOCKED_NONFINITE_GRADIENT'
        g=self._network.backbone.assigner.cls_emb.weight.grad
        if g is not None:self.gradient_rows.update(torch.where(g.abs().sum(1)>0)[0].tolist())
        self.pool_grad |= any(p.grad is not None and p.grad.abs().sum().item()>0 for name,p in self._network.backbone.named_parameters() if name.startswith(('pool.','pool_few.')))
    def checkpoint(self,path,optimizer,scheduler,epoch,phase):
        state=dict(network=self._network.state_dict(),task=self._cur_task,known=self._known_classes,total=self._total_classes,
                   memory=self.concm_stage1_memory,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
                   rng=self._capture_rng_state(),loader_rng=self.loader_generator.get_state(),synth_rng=self.synth_rng,
                   epoch=epoch,phase=phase,order=self.order,config=self.config,weight_sha256=WEIGHT_SHA,
                   train_seed=self.args['seed'],train_branch='C' if self.concm_stage1_enabled else 'B',
                   training_args=dict(self.args,device=[str(d) for d in self.args['device']]),
                   manifest_sha256=self.manifest_hashes,protocol_sha256=self.protocol_hash,code_sha256=self.config['code_sha256'])
        tmp=Path(str(path)+'.part');torch.save(state,tmp);tmp.replace(path)
    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['order']==self.order and state['code_sha256']==self.config['code_sha256'] and state['weight_sha256']==WEIGHT_SHA
        assert state['manifest_sha256']==self.manifest_hashes and state['protocol_sha256']==self.protocol_hash,'BLOCKED_CHECKPOINT_PROTOCOL'
        assert state['train_seed']==self.args['seed'] and state['train_branch']==('C' if self.concm_stage1_enabled else 'B'),'BLOCKED_CHECKPOINT_BRANCH'
        assert state['training_args']==dict(self.args,device=[str(d) for d in self.args['device']]),'BLOCKED_CHECKPOINT_CONFIG'
        self._network.load_state_dict(state['network'],strict=True);self._network.to(self._device)
        self._cur_task=state['task'];self._known_classes=state['known'];self._total_classes=state['total'];self.concm_stage1_memory=state['memory']
        self.optimizer=effective_optimizer(self._network.backbone);self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler=effective_scheduler(self.optimizer,self._cur_task)
        if self.scheduler:self.scheduler.load_state_dict(state['scheduler'])
        self.loader_generator.set_state(state['loader_rng']);self.synth_rng=state['synth_rng'];self._restore_rng_state(state['rng'])
        return state
    def _v2_epoch_hook(self,epoch,optimizer,scheduler,stats):
        torch.cuda.synchronize();now=time.monotonic()
        row=dict(session=self._cur_task,epoch=epoch+1,seconds=now-self.epoch_start,lr_used=self.epoch_lr,lr_next=[g['lr'] for g in optimizer.param_groups],**stats)
        self.epoch_lr=row['lr_next']
        self.records.append(row)
        with (self.output/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.checkpoint(self.output/'resume.pt',optimizer,scheduler,epoch+1,'epoch_complete')
        self.epoch_start=time.monotonic()
    def session(self,task,smoke=False,resume=None):
        self._cur_task=task;self._known_classes=[0,4,6][task];self._total_classes=[4,6,8][task]
        self.train_dataset=Images(Path(self.config['protocol'])/'train.csv',self.config['images'],self.order,range(self._known_classes,self._total_classes),True,smoke)
        self.train_loader=DataLoader(self.train_dataset,batch_size=48,shuffle=True,num_workers=8,generator=self.loader_generator)
        self._network.to(self._device);self.optimizer=effective_optimizer(self._network.backbone);self.scheduler=effective_scheduler(self.optimizer,task)
        self._v2_start_epoch=0
        if resume:
            state=self.restore(resume);self._v2_start_epoch=state['epoch']
        self.epoch_lr=[g['lr'] for g in self.optimizer.param_groups]
        assert {id(p) for g in self.optimizer.param_groups for p in g['params']}=={id(p) for p in self._network.backbone.parameters() if p.requires_grad}
        self.epoch_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        self._init_train(self.train_loader,None,self.optimizer,self.scheduler)
        t=time.monotonic();before=self._capture_rng_state()
        if self.concm_stage1_enabled:self._update_concm_stage1_memory()
        after=self._capture_rng_state();assert rng_equal(before,after),'BLOCKED_MEMORY_RNG'
        if self.concm_stage1_enabled:
            for c in range(self._known_classes,self._total_classes):
                assert self.concm_stage1_memory[c]['count']==sum(r['target']==c for r in self.train_dataset.rows)
                assert self.concm_stage1_memory[c]['task']==task
        memory_seconds=time.monotonic()-t;t=time.monotonic()
        self.checkpoint(self.output/f'session{task}.pt',self.optimizer,self.scheduler,self.args['tuned_epoch'],'session_complete')
        return dict(session=task,train_images=len(self.train_dataset),train_batches=len(self.train_loader),
                    epoch_seconds=self.records[-1]['seconds'] if self.records else 0.,memory_seconds=memory_seconds,checkpoint_seconds=time.monotonic()-t,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),gradient_rows=sorted(self.gradient_rows),
                    network_sha256=network_hash(self._network),pool_gradient=self.pool_grad)

def rng_equal(a,b):
    return a['python']==b['python'] and all(np.array_equal(x,y) for x,y in zip(a['numpy'],b['numpy'])) and torch.equal(a['torch'],b['torch']) and all(torch.equal(x,y) for x,y in zip(a.get('cuda',[]),b.get('cuda',[])))

@torch.no_grad()
def logits(model,x,seen,known,calibrate=False):
    out=model(x,train=False)
    raw=(out['logits']+out['logits_few'])[:,:seen]
    assert raw.shape[1]==seen and torch.isfinite(raw).all()
    alpha=1.
    if calibrate and known:
        alpha=headnorm_alpha(model.backbone,seen,known);raw=raw.clone();raw[:,:known]*=alpha
    return raw,raw.topk(min(5,seen),dim=1).indices,alpha

def p1(config):
    start=time.monotonic();report={'status':'RUNNING','test_predictions':0,'branches':{},'budget_seconds':1800}
    out=Path(config['output'])/'p1';out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'tests'));from test_medical_v2 import check_cleaning,check_components
    check_cleaning();check_components();report['cpu_regression']='PASS'
    summary=json.loads((Path(config['protocol'])/'V2_DATASET_SUMMARY.json').read_text())
    assert summary['status']=='V2_DATA_SUPPORT_PASS'
    # Remote readiness: open every retained training image once; no test model call.
    rows=list(csv.DictReader((Path(config['protocol'])/'train.csv').open()))
    for row in rows:
        with Image.open(Path(config['images'])/row['relative_path']) as im:im.load()
    report['remote_train_readable']=len(rows)
    initial={};s0={};input_hashes={}
    for branch in ('B','C'):
        seed_all(1993);args,order=args_for(config,1993,branch,True)
        learner=MedicalLearner(args,config,order,out/branch)
        initial[branch]=network_hash(learner._network)
        write_json(out/'weight_loading.json',learner._network.backbone.weight_load_audit)
        assert len(order)==len(set(order))==8
        learner._network.to('cuda').eval();report['branches'][branch]=[]
        ds=Images(Path(config['protocol'])/'train.csv',config['images'],order,range(8),False)
        x=torch.stack([ds[i][1] for i in range(2)]).cuda()
        before=learner._capture_rng_state();h=network_hash(learner._network)
        with torch.no_grad():
            a=learner._network(x,weight=torch.tensor([10529,27],device='cuda'))
            b=learner._network(x,weight=torch.tensor([27,10529],device='cuda'))
            c=learner._network(x)
            for key in ('logits','logits_few'):
                assert torch.equal(a[key],b[key]) and torch.equal(a[key],c[key]),'BLOCKED_LABEL_DEPENDENCE'
        assert network_hash(learner._network)==h and rng_equal(before,learner._capture_rng_state())
        for task in range(3):
            assert time.monotonic()-start<1800,'BLOCKED_P1_BUDGET'
            stats=learner.session(task,True);report['branches'][branch].append(stats)
            if task==0:s0[branch]=stats['network_sha256']
            learner._network.eval();before=learner._capture_rng_state();h=network_hash(learner._network)
            original=logits(learner._network,x,[4,6,8][task],[0,4,6][task])[0]
            calibrated=logits(learner._network,x,[4,6,8][task],[0,4,6][task],True)[0]
            if task==0:assert torch.equal(original,calibrated)
            assert rng_equal(before,learner._capture_rng_state()) and h==network_hash(learner._network)
            learner.restore(out/branch/f'session{task}.pt');learner._network.eval()
            assert torch.equal(original,logits(learner._network,x,[4,6,8][task],[0,4,6][task])[0]),'BLOCKED_RELOAD_LOGITS'
            # Same saved optimizer/RNG produces the same next update twice; discard both.
            saved=out/branch/f'session{task}.pt';next_hash=[]
            for _ in range(2):
                learner.restore(saved);learner._network.train();learner._network.original_backbone.eval()
                learner.optimizer.zero_grad();o=learner._network(x,train=True,weight=torch.tensor([10529,27],device='cuda'))
                (o['logits'].square().mean()+o['logits_few'].square().mean()+o['pool_id'].sum()).backward();learner.optimizer.step()
                next_hash.append(network_hash(learner._network))
            assert next_hash[0]==next_hash[1],'BLOCKED_RESUME_UPDATE'
            learner.restore(saved)
            assert learner.pool_grad and any(i>500 for i in learner.gradient_rows)
            if branch=='C' and task>0:
                before=learner._capture_rng_state();sampled=learner._concm_stage1_sample_memory()
                assert sampled is not None and rng_equal(before,learner._capture_rng_state())
                assert sampled[2].numel()==4*learner._known_classes
            write_json(out/'technical.json',report)
        input_hashes[branch]=learner.input_hashes
        del learner,a,b,c,x;gc.collect();torch.cuda.empty_cache()
    assert initial['B']==initial['C'],'BLOCKED_PAIRED_INITIALIZATION'
    assert s0['B']==s0['C'],'BLOCKED_S0_EQUIVALENCE'
    assert input_hashes['B']==input_hashes['C'],'BLOCKED_PAIRED_REAL_DATA_RNG'
    report.update(status='P1_PASS',initial_hashes=initial,s0_hashes=s0,elapsed_seconds=time.monotonic()-start,
                  label_blind='PASS',evaluation_purity='PASS',checkpoint_reload_and_update='PASS',paired_input_hashes=input_hashes)
    # Whole-matrix upper estimate using observed per-image time plus a 25% reserve.
    rates=[r['epoch_seconds']/r['train_images'] for rows in report['branches'].values() for r in rows]
    memory_rates=[r['memory_seconds']/r['train_images'] for r in report['branches']['C']]
    checkpoint_seconds=max(r['checkpoint_seconds'] for rs in report['branches'].values() for r in rs)
    estimate=(6*10*len(rows)*max(rates)+3*len(rows)*max(memory_rates)+198*checkpoint_seconds)*1.25
    report['budget_formula']='1.25 * (6*10*n_train*max_training_seconds_per_image + 3*n_train*max_memory_seconds_per_image + 198*max_checkpoint_seconds)'
    report['formal_estimate_seconds']=estimate;report['formal_budget_seconds']=86400
    if estimate>86400:report['status']='BLOCKED_FORMAL_BUDGET'
    write_json(out/'technical.json',report)
    return report
