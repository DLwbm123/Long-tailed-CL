"""NB-RL-A1 fixed experiment. Paths and role are supplied by a neutral launcher."""
import ast
import copy
import csv
import gc
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(os.environ['N78_ROOT'])
CFG = json.loads((ROOT / 'private/INPUT.json').read_text())
sys.path[:0] = [str(ROOT / 'source'), str(Path(CFG['legacy']) / 'tools'), CFG['deps']]
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from threadpoolctl import threadpool_limits
from run_medical_v2 import Learner, Images, seed_all, sha, write_json, network_hash, rng_equal
from preflight_ct1 import ISIC_ORDERS
from ct3p_core import features, translation, math_check
from ct1_statistics import empty, append, transport, ridge
from nb_rl_a1_core import objective, selfcheck

PUB = ROOT / 'public'
PRIVATE = ROOT / 'private'
WORKER = os.environ.get('N78_WORKER')
OUT = PUB if WORKER is None else PRIVATE / 'workers' / WORKER
OUT.mkdir(parents=True, exist_ok=True)
METHODS = tuple(CFG.get('methods', ('S', 'H', 'K', 'G', 'R', 'E')))
FROZEN_REFERENCES = CFG.get('frozen_references', [('F_S', 'S'), ('F_R', 'R')])
if CFG.get('experiment') in ('NB-RL-A2', 'NB-RL-A3'):
    from nb_rl_a2_core import objective
elif CFG.get('experiment') == 'NB-RL-A5':
    from nb_rl_a5_core import objective
elif CFG.get('experiment') == 'NB-RL-A4':
    from nb_rl_a4_core import objective


RUN_SPECS = {int(k): v for k, v in CFG.get('run_specs', {}).items()}
RUN_ORDERS = {k: ISIC_ORDERS[v['order_seed']] for k, v in RUN_SPECS.items()} if RUN_SPECS else ISIC_ORDERS
RUN_IDS = tuple(RUN_ORDERS)


def training_seed(run_id):
    return RUN_SPECS.get(run_id, {}).get('training_seed', run_id)


def seed_fields(run_id):
    return dict(run_id=run_id, order_seed=RUN_SPECS.get(run_id, {}).get('order_seed', run_id),
                training_seed=training_seed(run_id))


def save(name, value):
    if isinstance(value, dict) and 'seed' in value: value = dict(value, **seed_fields(value['seed']))
    write_json(OUT / name, value)


def record(name, value):
    if 'seed' in value: value = dict(value, **seed_fields(value['seed']))
    with (OUT / name).open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def dump(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.part')
    opener = (lambda p, mode: gzip.open(p, mode, compresslevel=1)) if path.suffix == '.gz' else open
    with opener(temp, 'wb') as f:
        torch.save(value, f, pickle_protocol=4)
    temp.replace(path)


def load(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rb') as f:
        return torch.load(f, map_location='cpu', weights_only=False)


def tensors_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tensors_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tensors_equal(x, y) for x, y in zip(a, b))
    return a == b


class BatchDataset(Dataset):
    """Deterministic per-sample augmentation allows exact batch-cursor recovery."""
    def __init__(self, owner, seed, task, epoch=0, split='train', train=False, bounded=False):
        spec = CFG['dataset']; classes = range(2 * (task - 1), 2 * task) if split == 'train' else range(2 * task)
        assert split == 'train' or owner.phase == 'evaluate'
        self.base = Images(Path(spec['manifests']) / (split + '.csv'), spec['images'], RUN_ORDERS[seed], classes, train)
        self.base.rows.sort(key=lambda x: x['sample_id'])
        if bounded:
            self.base.rows = sum(([r for r in self.base.rows if r['target'] == c][:48] for c in classes), [])
        self.rows = self.base.rows; self.seed = seed; self.task = task; self.epoch = epoch
        self.train = train
        owner.allowed = {str((Path(spec['images']) / r['relative_path']).resolve()) for r in self.rows}
        owner.scope = dict(seed=seed,task=task,split=split,classes=list(classes))
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        # Augmentations are identical across methods and independent of action draws.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(training_seed(self.seed) * 10000019 + self.task * 100003 + self.epoch * 2003 + index)
            result=self.base[index]
        return result


class Model:
    def __init__(self, seed):
        seed_all(training_seed(seed))
        args = json.loads((Path(CFG['legacy']) / 'third_party/APART/exps/apart_cifar_shuffle.json').read_text())
        args.update(nb_classes=8,nb_tasks=4,init_cls=2,increment=2,seed=training_seed(seed),device=[torch.device('cuda:0')],
                    medical_v2=True,locked_weight_path=CFG['weight'],concm_stage1=False,
                    concm_stage1_eval_calibration=False,calibration_rule='none',save_task_checkpoints=False,
                    batchwise_prompt=False,shared_prompt_pool=False,shared_prompt_key=False)
        # No global class counts or labels are supplied to feature construction.
        args.pop('longtail', None)
        self.learner = Learner(args)
        self.net = self.learner._network.cuda().eval()
        self.nonshared = {'backbone.' + k for k in self.net.backbone.weight_load_audit['allowed_missing']}
        self.names = []
        for name, p in self.net.named_parameters():
            trainable = '.pool.pool.' in name or '.pool_few.pool.' in name
            p.requires_grad_(trainable)
            if trainable: self.names.append(name)
        self.parameters = [p for p in self.net.parameters() if p.requires_grad]
        assert self.parameters and all(n in self.nonshared for n in self.names)
        for pool in (self.net.backbone.pool, self.net.backbone.pool_few): pool.batchwise_prompt = False
        self.seed = seed; self.sigma = .5; self.policy = torch.Generator(device='cuda').manual_seed(training_seed(seed) + 71000003)
        self.bank = empty(1536); self.opt = None; self.scheduler = None
        self.task = 0; self.epoch = 0; self.batch = 0; self.steps = 0; self.method = None
    def z(self, x, net=None):
        return features(self.net if net is None else net, x, self.learner)
    def delta(self):
        return {k: v.detach().cpu().clone() for k,v in self.net.state_dict().items() if k in self.nonshared}
    def restore_delta(self, delta):
        assert set(delta) == self.nonshared
        state = self.net.state_dict(); state.update(delta); self.net.load_state_dict(state, strict=True)
    def optimizer(self, epochs):
        lr = CFG.get('conditions', {}).get(self.method, {}).get('lr', .0003) if self.task > 1 else .0003
        self.opt = torch.optim.AdamW(self.parameters,lr=lr,weight_decay=.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,T_max=epochs,eta_min=1e-5)
    def snapshot(self):
        return dict(delta=self.delta(),optimizer=self.opt.state_dict(),scheduler=self.scheduler.state_dict(),
                    sigma=self.sigma,policy_rng=self.policy.get_state(),global_rng=self.learner._capture_rng_state(),
                    seed=self.seed,**seed_fields(self.seed),task=self.task,epoch=self.epoch,batch=self.batch,steps=self.steps,bank=self.bank,
                    loader_rule='stateless per sample; epoch permutation seeded by seed/task/epoch',
                    loader_rng=loader_generator(self.seed,self.task,self.epoch).get_state())
    def restore(self, state):
        self.restore_delta(state['delta']);self.opt.load_state_dict(state['optimizer']);self.scheduler.load_state_dict(state['scheduler'])
        for k in ('sigma','task','epoch','batch','steps','bank'):setattr(self,k,state[k])
        self.policy.set_state(state['policy_rng']);self.learner._restore_rng_state(state['global_rng'])


def loader_generator(seed, task, epoch):
    return torch.Generator().manual_seed(training_seed(seed) * 100003 + task * 2003 + epoch)


def loader(ds, shuffle=False):
    return DataLoader(ds,batch_size=48,shuffle=shuffle,num_workers=4,drop_last=False,
                      generator=loader_generator(ds.seed,ds.task,ds.epoch))


class Run:
    def __init__(self):
        self.started = float(Path('/proc/self/stat').read_text().split()[21])/os.sysconf('SC_CLK_TCK')
        self.phase = 'engineering' if os.environ.get('N78_ROLE') in ('engineering','verify','parallel_probe') else 'initialization'
        self.allowed = set(); self.scope = None
        self.prior = json.loads((OUT/'RESOURCE_LEDGER.json').read_text()) if (OUT/'RESOURCE_LEDGER.json').exists() else {}
        self.gpu_prior = self.prior.get('GPU_process_residence_seconds',0)
        self.counts = self.prior.get('counts',{}).copy()
        self.forbidden = 0
        self.image_reads=mp.Array('q',[0,0])
        def guard(event,args):
            if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
            p=Path(os.fsdecode(args[0])).resolve()
            if p.suffix.lower() in ('.jpg','.jpeg','.png') and str(p) not in self.allowed:
                self.forbidden += 1; raise AssertionError('BLOCKED_OLD_FUTURE_OR_VAL_IMAGE')
            if p.suffix.lower() in ('.jpg','.jpeg','.png'):
                with self.image_reads.get_lock():self.image_reads[1 if self.phase=='evaluate' else 0]+=1
            if any(v.lower() in ('test','reserved','oracle') for v in p.parts):
                raise AssertionError('BLOCKED_TEST_RESERVED')
        sys.addaudithook(guard)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
        self.resources()
    def count(self,k,n=1):
        self.counts[k]=self.counts.get(k,0)+n
    def resources(self,enforce=True):
        elapsed=time.time()-CFG['wall_T0_unix'];gpu=self.gpu_prior+time.monotonic()-self.started
        primary=0
        for path in ROOT.rglob('*'):
            try:
                if path.is_file() and not path.is_symlink():primary+=path.stat().st_size
            except FileNotFoundError:pass  # Another worker atomically replaced its rolling state.
        backups=0
        if (PRIVATE/'transfer_acks').exists():
            entries={}
            for p in (PRIVATE/'transfer_acks').glob('*.json'):
                ack=json.loads(p.read_text())
                for item in ack.get('items',[]):entries[item['file']]=item['bytes']
            backups=sum(entries.values())
        x=dict(phase=self.phase,wall_T0_unix=CFG['wall_T0_unix'],wall_seconds=elapsed,GPU_process_residence_seconds=gpu,
               counts=self.counts,primary_bytes=primary,independent_backup_bytes=backups,persistent_including_backup_bytes=primary+backups,
               free_bytes=shutil.disk_usage(ROOT).free,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
               budget_wall_seconds=57600,budget_GPU_seconds=57600,budget_persistent_bytes=16*1024**3,
               forbidden_attempts=self.forbidden,old_fit_reads=0,future_fit_reads=0,test_reserved_reads=0,
               current_scope=self.scope,unix=time.time())
        x['image_opens_this_process']=dict(fit=int(self.image_reads[0]),val=int(self.image_reads[1]))
        x['P0_image_count_note']='P0 measured consumed rows; loader prefetch stayed in legal current Task1 fit.'
        save('RESOURCE_LEDGER.json',x)
        if enforce:
            assert elapsed<57600 and gpu<57600,'INCOMPLETE_BUDGET_16H'
            assert primary+backups<16*1024**3 and x['free_bytes']>=1024**3,'BLOCKED_STORAGE'
            if self.phase=='engineering':assert elapsed<9000 and self.counts.get('engineering_steps',0)<=180,'BLOCKED_P0_BUDGET'
        return x
    def before_forward(self,training=False):
        elapsed=time.time()-CFG['wall_T0_unix']
        assert elapsed<52200,'INCOMPLETE_BUDGET_FORWARD_14_5H'
        if training:assert elapsed<45000,'INCOMPLETE_BUDGET_TRAIN_12_5H'
    def extract(self,model,seed,task,net=None,split='train',bounded=False):
        self.before_forward();ds=BatchDataset(self,seed,task,split=split,bounded=bounded)
        zs=[];ys=[];started=time.monotonic()
        with torch.no_grad():
            for _,x,y in loader(ds):
                self.before_forward();zs.append(model.z(x.cuda(),net).cpu().numpy());ys.append(y.numpy())
                self.count(self.phase+'_'+split+'_rows',len(y))
        return np.concatenate(zs).astype(np.float64),np.concatenate(ys),ds.rows,time.monotonic()-started


def gradients(terms,coefficients,model):
    grads={};out={}
    for name,term in terms.items():
        g=torch.autograd.grad(term,model.parameters,retain_graph=True,allow_unused=True)
        grads[name]=g;norm=math.sqrt(sum(float(v.square().sum()) for v in g if v is not None))
        out[name]=dict(norm=norm,weighted_norm=norm*coefficients[name])
    pairs={}
    for a in grads:
        for b in grads:
            if a>=b:continue
            denom=out[a]['norm']*out[b]['norm']
            dot=sum(float((x*y).sum()) for x,y in zip(grads[a],grads[b]) if x is not None and y is not None)
            pairs[a+'_'+b]=None if denom<=1e-30 else dot/denom
    return dict(terms=out,cosines=pairs)


def step(run,model,anchor,x,y,W,weights,method,task,diagnose=False):
    run.before_forward(training=True);model.opt.zero_grad(set_to_none=True)
    with torch.no_grad():target=model.z(x,anchor)
    z=model.z(x)
    total,terms,coef,next_sigma,diag,actions=objective(z,target,W,y,weights,method,task,model.sigma,model.policy)
    assert torch.isfinite(total) and all(torch.isfinite(v) for v in terms.values())
    gd=gradients(terms,coef,model) if diagnose else None
    total.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters,1.,error_if_nonfinite=True)
    assert all(p.grad is None for p in anchor.parameters())
    model.opt.step();model.sigma=next_sigma;model.steps+=1
    run.count(run.phase+'_steps');run.count(run.phase+'_train_rows',len(y))
    diag.update(losses={k:float(v.detach()) for k,v in terms.items()},weighted_losses={k:float(v.detach())*coef[k] for k,v in terms.items()},
                total_loss=float(total.detach()),total_gradient_norm=float(norm),clipped=bool(norm>1),sigma_after=model.sigma)
    return diag,gd,actions


def metadata():
    spec=CFG['dataset'];rows={}
    for split in ('train','val'):
        p=Path(spec['manifests'])/(split+'.csv');assert sha(p)==spec['manifest_sha256'][split]
        rows[split]=list(csv.DictReader(p.open()));assert len(rows[split])==spec['n_'+split]
    tasks={}
    for seed,order in RUN_ORDERS.items():
        tasks[seed]=[]
        for t in range(4):
            nt=sum(int(x['original_label']) in order[2*t:2*t+2] for x in rows['train'])
            nv=sum(int(x['original_label']) in order[:2*t+2] for x in rows['val'])
            tasks[seed].append(dict(task=t+1,n_train=nt,n_val=nv,batches=math.ceil(nt/48),val_batches=math.ceil(nv/48)))
    return tasks


def engineering(run):
    save('MATH_CHECK.json',dict(policy=selfcheck(),translation=math_check()))
    tasks=metadata();save('METHOD_MATRIX.json',dict(methods=list(METHODS)+['F_S','F_R'],orders=ISIC_ORDERS,tasks=tasks,
             tiers={str(e):sum(v['batches'] for ts in tasks.values() for v in ts)*6*e for e in (5,3)},no_base=True))
    model=Model(1993);initial=model.delta();initial_hash=network_hash(model.net)
    save('INITIALIZATION_AUDIT.json',dict(source='EXTERNAL_AUGREG_PLUS_UNTRAINED_FEATURE_MODULES',initial_network_sha256=initial_hash,
          trainable_names=model.names,trainable_numel=sum(p.numel() for p in model.parameters),feature_dim=1536,
          frozen_neural_heads=True,frozen_discrete_prompt_keys=True,frozen_unused_label_assigner=True,task_setup_called=False))
    z,y,rows,extract_seconds=run.extract(model,1993,1)
    start_bank=append(empty(1536),z,y,range(2),1)
    began=time.monotonic();W,solve_audit=ridge(start_bank);solve_seconds=time.monotonic()-began
    counts=np.bincount(y,minlength=8);weights=np.zeros(8);weights[:2]=len(y)/(2*counts[:2]);weights=torch.tensor(weights,device='cuda',dtype=torch.float32)
    ds=BatchDataset(run,1993,1,epoch=1,train=True);batches=[]
    for _,x,b_y in loader(ds,True):
        batches.append((x.cuda(),b_y.cuda()))
        if len(batches)==3:break
    assert all(len(b_y)==48 for _,b_y in batches)
    rng=np.random.default_rng(78001);W8=np.column_stack([W,rng.normal(size=(1536,6))*.01]).astype(np.float32)
    W8=torch.tensor(W8,device='cuda');W2=torch.tensor(W,device='cuda',dtype=torch.float32)
    timings=[];t1=[];peak=0
    for method in METHODS:
        model.restore_delta(initial);model.sigma=.5;model.policy.manual_seed(1993+71000003);model.optimizer(5)
        anchor=copy.deepcopy(model.net).requires_grad_(False).eval()
        times=[];diagnostics=[]
        for index in range(6):
            x,b_y=batches[index%3];torch.cuda.synchronize();began=time.monotonic()
            diag,gd,_=step(run,model,anchor,x,b_y,W8,weights,method,2,diagnose=index==1)
            torch.cuda.synchronize();times.append(time.monotonic()-began)
            if gd:diagnostics.append(gd)
        # Two genuine Task1 updates from the same untrained initialization.
        model.restore_delta(initial);model.optimizer(5);model.sigma=.5;model.policy.manual_seed(1993+71000003)
        firstdiag=None
        for x,b_y in batches[:2]:firstdiag,_,_=step(run,model,anchor,x,b_y,W2,weights,method,1)
        t1.append(dict(method=method,delta={k:v.clone() for k,v in model.delta().items()},diag=firstdiag))
        assert diagnostics[0]['terms']['FD']['norm']>0,'BLOCKED_FD_FEATURE_GRADIENT'
        branch='PG' if method in ('G','R') else 'Exact' if method=='E' else None
        if branch:assert diagnostics[0]['terms'][branch]['norm']>0,'BLOCKED_RL_FEATURE_GRADIENT'
        assert all(not v.requires_grad for v in anchor.parameters())
        timings.append(dict(method=method,step_seconds=float(np.median(times[2:])),diagnostic_step_seconds=times[1],
                            measured_steps=8,gradient=diagnostics[0],task1_terms=firstdiag['losses']))
        run.resources();del anchor;gc.collect();torch.cuda.empty_cache()
    assert tensors_equal(t1[0]['delta'],t1[1]['delta']) and tensors_equal(t1[0]['delta'],t1[2]['delta'])
    assert tensors_equal(t1[3]['delta'],t1[4]['delta'])
    # Exact update recovery, including optimizer, scheduler, local/global RNG and bank.
    model.restore_delta(initial);model.optimizer(5);model.bank=start_bank;model.task=1;model.epoch=1;model.batch=0;model.sigma=.5
    anchor=copy.deepcopy(model.net).requires_grad_(False).eval();x,b_y=batches[0]
    step(run,model,anchor,x,b_y,W2,weights,'R',1)
    snapshot=copy.deepcopy(model.snapshot());dump(PRIVATE/'p0_resume.pt.gz',snapshot)
    state=load(PRIVATE/'p0_resume.pt.gz');assert tensors_equal(snapshot,state)
    d1,_,a1=step(run,model,anchor,x,b_y,W2,weights,'R',1);after=copy.deepcopy(model.snapshot())
    model.restore(state);d2,_,a2=step(run,model,anchor,x,b_y,W2,weights,'R',1)
    assert torch.equal(a1,a2) and tensors_equal(after,model.snapshot()) and d1==d2,'BLOCKED_NEXT_UPDATE_RESTORE'
    # Worst-width no-label synthetic evaluation timing, without reading validation images.
    inference=[]
    with torch.no_grad():
        for _ in range(3):
            torch.cuda.synchronize();began=time.monotonic();model.z(torch.zeros_like(x));torch.cuda.synchronize();inference.append(time.monotonic()-began)
    compact=dict(delta=model.delta(),bank=start_bank,W_start=W,W_final=W,seed=1993,task=1,method='P0_ONLY')
    began=time.monotonic();dump(PRIVATE/'p0_stage.pt.gz',compact);save_seconds=time.monotonic()-began
    report=dict(status='MEASURED_PENDING_BACKUP_AND_ADMISSION',timings=timings,engineering_steps=run.counts['engineering_steps'],
                current_T1_rows=len(y),current_T1_extract_seconds=extract_seconds,extraction_seconds_per_batch=extract_seconds/math.ceil(len(y)/48),
                synthetic_evaluation_seconds_per_batch=float(np.median(inference)),ridge_seconds=solve_seconds,ridge=solve_audit,
                compact_stage_bytes=(PRIVATE/'p0_stage.pt.gz').stat().st_size,rolling_bytes=(PRIVATE/'p0_resume.pt.gz').stat().st_size,
                stage_save_seconds=save_seconds,task1_S_H_K_equal=True,task1_G_R_equal=True,next_real_update_restore_exact=True,
                initialization=initial_hash,no_validation_images=True,logical_prefix_reuse=False)
    save('P0_MEASUREMENTS.json',report);run.resources()
    print('P0_MEASUREMENTS_READY',flush=True)


def enqueue(paths,label,wait=False):
    key=str(time.time_ns());items=[]
    for p in paths:
        p=Path(p)
        if p.parent==PUB:
            dest=PRIVATE/'metadata'/key/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest);p=dest
        items.append(dict(file=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    write_json(PRIVATE/'transfer_requests'/(key+'.json'),dict(id=key,label=label,items=items,unix=time.time()))
    if wait:
        started=time.monotonic()
        while not (PRIVATE/'transfer_acks'/(key+'.json')).exists():
            assert time.monotonic()-started<900 and time.time()-CFG['wall_T0_unix']<57600,'BLOCKED_BACKUP_TIMEOUT'
            time.sleep(2)
        ack=json.loads((PRIVATE/'transfer_acks'/(key+'.json')).read_text())
        assert ack['status']=='PASS' and ack['items']==items
    return key


def verify_engineering(run):
    model=Model(1993);model.optimizer(3);model.task=1;model.epoch=1
    anchor=copy.deepcopy(model.net).requires_grad_(False).eval()
    stage=load(PRIVATE/'p0_stage.pt.gz');W=torch.tensor(stage['W_start'],device='cuda',dtype=torch.float32)
    ds=BatchDataset(run,1993,1,epoch=1,train=True)
    counts=np.bincount([r['target'] for r in ds.rows],minlength=8)
    weights=np.zeros(8);weights[:2]=len(ds)/(2*counts[:2]);weights=torch.tensor(weights,device='cuda',dtype=torch.float32)
    iterator=iter(loader(ds,True));_,x0,y0=next(iterator);_,x1,y1=next(iterator)
    step(run,model,anchor,x0.cuda(),y0.cuda(),W,weights,'R',1);model.batch=1
    state=copy.deepcopy(model.snapshot());cursor_path=OUT/'cursor_resume.pt' if WORKER else PRIVATE/'p0_cursor_resume.pt.gz';dump(cursor_path,state)
    d1,_,a1=step(run,model,anchor,x1.cuda(),y1.cuda(),W,weights,'R',1);model.batch=2
    after=copy.deepcopy(model.snapshot());model.restore(load(cursor_path))
    ds2=BatchDataset(run,1993,model.task,epoch=model.epoch,train=True)
    restored_loader=iter(loader(ds2,True))
    for _ in range(model.batch):next(restored_loader)
    _,rx,ry=next(restored_loader)
    assert torch.equal(x1,rx) and torch.equal(y1,ry),'BLOCKED_LOADER_CURSOR'
    d2,_,a2=step(run,model,anchor,rx.cuda(),ry.cuda(),W,weights,'R',1);model.batch+=1
    assert torch.equal(a1,a2) and d1==d2 and tensors_equal(after,model.snapshot())
    # Task transitions modify only bookkeeping/optimizer; they never change feature modules.
    delta=model.delta();before=model.z(rx.cuda()).detach();model.task=2;model.optimizer(3)
    torch.testing.assert_close(before,model.z(rx.cuda()).detach(),atol=0,rtol=0)
    assert tensors_equal(delta,model.delta())
    save('CURSOR_AND_TRANSITION_CHECK.json',dict(status='PASS',next_distinct_real_batch_exact=True,
          restored_policy_actions_exact=True,optimizer_scheduler_sigma_global_rng_exact=True,
          task_transition_feature_mapping_unchanged=True,engineering_steps=run.counts['engineering_steps']))
    print('P0_CURSOR_PASS',flush=True)


def check_backlog(wait=False):
    started=time.monotonic()
    while True:
        pending=[]
        for p in (PRIVATE/'transfer_requests').glob('*.json'):
            ack=PRIVATE/'transfer_acks'/p.name;request=json.loads(p.read_text())
            if ack.exists():
                a=json.loads(ack.read_text());assert a['status']=='PASS' and a['items']==request['items']
            else:pending.append(request)
        if not pending:return
        assert max(time.time()-r['unix'] for r in pending)<900,'BLOCKED_BACKUP_BACKLOG'
        if not wait:return
        assert time.monotonic()-started<900 and time.time()-CFG['wall_T0_unix']<57600,'BLOCKED_BACKUP_TIMEOUT'
        time.sleep(2)


def delta_hash(delta):
    h=hashlib.sha256()
    for k,v in sorted(delta.items()):h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


def stage_path(method,seed,task):
    return PRIVATE/'stages'/f'{method}_{seed}_t{task}.pt.gz'


def save_stage(run,model,method,task,W_start,W_final,extra):
    state=dict(seed=model.seed,method=method,task=task,bank=model.bank,W_start=W_start,W_final=W_final,
               sigma=model.sigma,policy_rng=model.policy.get_state(),global_rng=model.learner._capture_rng_state(),
               total_steps=model.steps,protocol_sha256=sha(PUB/'PROTOCOL_LOCK.json'),**extra)
    path=stage_path(method,model.seed,task);assert not path.exists(),'BLOCKED_STAGE_OVERWRITE'
    dump(path,state)
    entry=dict(method=method,seed=model.seed,**seed_fields(model.seed),task=task,file=path.name,sha256=sha(path),bytes=path.stat().st_size,
               new_steps=extra.get('task_steps',0),network_delta_sha256=extra['network_delta_sha256'])
    record('STAGE_RECEIPTS.jsonl',entry);enqueue([path],'stage_'+path.stem);run.resources();return entry


def train_matrix(run,epochs,jobs=None):
    run.phase='formal';entries=[];formal_start=time.time()
    for seed in RUN_IDS:
        initial_hash=None
        for method in METHODS:
            if jobs is not None and (seed,method) not in jobs:continue
            model=Model(seed);model.method=method
            fresh_hash=delta_hash(model.delta())
            if initial_hash is None:initial_hash=fresh_hash
            assert fresh_hash==initial_hash and model.sigma==.5
            record('INITIALIZATION_BY_TRAJECTORY.jsonl',dict(seed=seed,method=method,delta_sha256=fresh_hash,sigma=model.sigma))
            for task in range(1,5):
                run.before_forward(training=True);check_backlog()
                model.task=task;model.epoch=0;model.batch=0;model.optimizer(epochs)
                anchor=copy.deepcopy(model.net).requires_grad_(False).eval();prior=copy.deepcopy(model.bank)
                before,y,rows,_=run.extract(model,seed,task,anchor)
                current=range(2*(task-1),2*task);seen=2*task
                bootstrap=append(prior,before,y,current,task);W_start,diag=ridge(bootstrap)
                assert tensors_equal(model.bank,prior) and len(bootstrap['n'])==seen
                counts=np.bincount(y,minlength=8);weights=np.zeros(8)
                weights[list(current)]=len(y)/(2*counts[list(current)])
                weights=torch.tensor(weights,dtype=torch.float32,device='cuda');W=torch.tensor(W_start,dtype=torch.float32,device='cuda')
                ids=[r['sample_id'] for r in rows];first_steps=model.steps;anchor_delta=model.delta()
                for epoch in range(1,epochs+1):
                    model.epoch=epoch;ds=BatchDataset(run,seed,task,epoch=epoch,train=True)
                    path=PRIVATE/'batch_checks'/f'{method}_{seed}_t{task}_e{epoch}.jsonl'
                    assert not path.exists(),'BLOCKED_BATCH_RECEIPT_OVERWRITE'
                    n=0;agg={};class_exposure={str(c):0 for c in current};nb=0;clipped_updates=0
                    for batch,(_,x,b_y) in enumerate(loader(ds,True)):
                        model.batch=batch
                        h=hashlib.sha256(x.numpy().tobytes()+b_y.numpy().tobytes()).hexdigest()
                        item=dict(batch=batch,sha256=h,n=len(b_y))
                        with path.open('a') as f:f.write(json.dumps(item)+'\n')
                        x=x.cuda();b_y=b_y.cuda();previous=[p.detach().clone() for p in model.parameters] if batch==0 else None
                        values,gd,actions=step(run,model,anchor,x,b_y,W,weights,method,task,diagnose=batch==0)
                        if gd:
                            update=math.sqrt(sum(float((p.detach()-q).square().sum()) for p,q in zip(model.parameters,previous)))
                            record('gradient_diagnostics.jsonl',dict(method=method,seed=seed,task=task,epoch=epoch,
                                   sampling='first real minibatch of epoch',actual_update_norm=update,**gd))
                        b=len(b_y);n+=b;nb+=1;clipped_updates+=int(values['clipped'])
                        for c in current:class_exposure[str(c)]+=int((b_y==c).sum())
                        flattened={k:v for k,v in values.items() if isinstance(v,(int,float,bool))}
                        flattened.update({'loss_'+k:v for k,v in values['losses'].items()})
                        flattened.update({'weighted_'+k:v for k,v in values['weighted_losses'].items()})
                        for k,v in flattened.items():agg[k]=agg.get(k,0.)+float(v)*b
                        model.batch=batch+1
                        if not (OUT/'FIRST_FORMAL_UPDATE.json').exists():save('FIRST_FORMAL_UPDATE.json',dict(unix=time.time(),seed=seed,method=method,task=task,pid=os.getpid()))
                        if model.steps%50==0:run.resources();check_backlog()
                    assert n==len(ds) and nb==math.ceil(len(ds)/48)
                    assert all(class_exposure[str(c)]==int(counts[c]) for c in current)
                    model.scheduler.step()
                    record('reward_diagnostics.jsonl',dict(method=method,seed=seed,task=task,epoch=epoch,rows=n,class_exposure=class_exposure,
                           means={k:v/n for k,v in agg.items()},clipped_updates=clipped_updates,updates=nb,clip_frequency=clipped_updates/nb,
                           sigma_end=model.sigma,total_steps=model.steps))
                    # One epoch boundary snapshot contains the complete task recovery state.
                    snapshot=model.snapshot();snapshot.update(method=method,W_start=W_start,before=before,labels=y,ids=ids,
                           anchor_delta=anchor_delta,prior_bank=prior,protocol_sha256=sha(PUB/'PROTOCOL_LOCK.json'))
                    dump((OUT if WORKER else PRIVATE)/'rolling.pt',snapshot);del snapshot
                    run.resources();print('EPOCH',method,seed,task,epoch,model.steps,flush=True)
                after,ys,rs,_=run.extract(model,seed,task)
                assert ids==[r['sample_id'] for r in rs] and np.array_equal(y,ys)
                a,b,residual,audit=translation(before,after,y)
                model.bank=append(transport(prior,a,b,residual,task),after,y,current,task)
                assert len(model.bank['n'])==seen and model.bank['arrival']==sum(([t,t] for t in range(1,task+1)),[])
                W_final,ridge_audit=ridge(model.bank);delta=model.delta();dh=delta_hash(delta)
                entries.append(save_stage(run,model,method,task,W_start,W_final,dict(delta=delta,network_delta_sha256=dh,
                       task_steps=model.steps-first_steps,translation=audit,ridge=ridge_audit,initialization='no_base_external_pretraining')))
                ((OUT if WORKER else PRIVATE)/'rolling.pt').unlink();del anchor,before,after,prior,bootstrap,delta,anchor_delta,x,b_y
                gc.collect();torch.cuda.empty_cache()
            del model;gc.collect();torch.cuda.empty_cache()
    matrix=json.loads((PUB/'METHOD_MATRIX.json').read_text())
    expected_stages=4*len(METHODS)*len(RUN_IDS) if jobs is None else 4*len(jobs)
    expected_steps=matrix['tiers'][str(epochs)] if jobs is None else sum(sum(t['batches'] for t in matrix['tasks'][str(seed)])*epochs for seed,method in jobs)
    assert len(entries)==expected_stages and run.counts['formal_steps']==expected_steps
    save('TRAINED_MATRIX_LOCK.json',dict(status='LOCKED',entries=entries,new_optimizer_steps=expected_steps,epochs=epochs,
                                       total_task_epochs=expected_stages*epochs,unix=time.time(),formal_started= formal_start))
    return entries


def frozen_matrix(run,entries,pairs=None):
    initial_entries=len(entries)
    run.phase='frozen_statistics'
    for seed in RUN_IDS:
        for method,owner in FROZEN_REFERENCES:
            if pairs is not None and (seed,method,owner) not in pairs:continue
            source=load(stage_path(owner,seed,1));model=Model(seed);model.restore_delta(source['delta']);model.bank=source['bank']
            model.sigma=source['sigma'];model.steps=source['total_steps'];model.policy.set_state(source['policy_rng'])
            ref=next(x for x in entries if x['method']==owner and x['seed']==seed and x['task']==1)
            entries.append(dict(ref,method=method,alias_of=owner))
            for task in range(2,5):
                z,y,rows,_=run.extract(model,seed,task)
                model.bank=append(model.bank,z,y,range(2*(task-1),2*task),task);W,diag=ridge(model.bank)
                entries.append(save_stage(run,model,method,task,W,W,dict(network_source=ref['file'],network_delta_sha256=ref['network_delta_sha256'],
                       task_steps=0,ridge=diag,initialization=owner+'_Task1')))
                del z;check_backlog()
            assert delta_hash(model.delta())==ref['network_delta_sha256']
            del model;gc.collect();torch.cuda.empty_cache()
    assert len(entries)==initial_entries+4*(len(RUN_IDS)*len(FROZEN_REFERENCES) if pairs is None else len(pairs))
    save('ALL_STATES_LOCK.json',dict(status='LOCKED',entries=entries,unix=time.time(),validation_images_read=0))
    return entries


def evaluate_matrix(run,entries):
    assert (PUB/'ALL_STATES_LOCK.json').exists();run.phase='evaluate';out=[]
    for entry in entries:
        path=PRIVATE/'stages'/entry['file'];assert sha(path)==entry['sha256']
        state=load(path);model=Model(entry['seed'])
        delta=state.get('delta')
        if delta is None:delta=load(PRIVATE/'stages'/state['network_source'])['delta']
        model.restore_delta(delta);assert delta_hash(model.delta())==entry['network_delta_sha256']
        z,y,rows,_=run.extract(model,entry['seed'],entry['task'],split='val')
        payload=dict(raw=z@state['W_final'],start_raw=z@state['W_start'],y=y,order=np.array(RUN_ORDERS[entry['seed']][:2*entry['task']]),
                     ids=np.array([r['sample_id'] for r in rows]),component=np.array([r['identity_component'] for r in rows]),
                     original=np.array([int(r['original_label']) for r in rows]))
        p=PRIVATE/'sealed'/f"{entry['method']}_{entry['seed']}_t{entry['task']}.npz"
        assert not p.exists();np.savez_compressed(p,**payload)
        e=dict(entry,prediction_file=p.name,prediction_sha256=sha(p));out.append(e)
        enqueue([p],'sealed_'+p.stem);run.resources();check_backlog()
        del model,state,z,delta;gc.collect();torch.cuda.empty_cache()
    save('PREDICTIONS_LOCK.json',dict(status='LOCKED',entries=out,unix=time.time()))
    return out


def qualification():
    gate=json.loads((PUB/'P0_ENGINEERING_REPORT.json').read_text());assert gate['status']=='PASS'
    protocol=json.loads((PUB/'PROTOCOL_LOCK.json').read_text())
    for name,expected in protocol['source_sha256'].items():assert sha(ROOT/'source'/name)==expected
    for name,expected in CFG['qualified_sha256'].items():assert sha(Path(CFG['legacy'])/name)==expected
    return protocol


def parallel_probe(run):
    model=Model(1993);model.optimizer(3);model.task=1;model.epoch=1
    anchor=copy.deepcopy(model.net).requires_grad_(False).eval();stage=load(PRIVATE/'p0_stage.pt.gz')
    rng=np.random.default_rng(78001)
    W=torch.tensor(np.column_stack([stage['W_start'],rng.normal(size=(1536,6))*.01]),device='cuda',dtype=torch.float32)
    ds=BatchDataset(run,1993,1,epoch=1,train=True);counts=np.bincount([r['target'] for r in ds.rows],minlength=8)
    weights=np.zeros(8);weights[:2]=len(ds)/(2*counts[:2]);weights=torch.tensor(weights,device='cuda',dtype=torch.float32)
    iterator=iter(loader(ds,True));times=[];cores=[];extra=[];diagnostic=0
    for index in range(10):
        began=time.monotonic();_,x,y=next(iterator)
        hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
        x=x.cuda();y=y.cuda();torch.cuda.synchronize();start=time.monotonic()
        step(run,model,anchor,x,y,W,weights,'R',2,diagnose=index==1)
        torch.cuda.synchronize();core=time.monotonic()-start;whole=time.monotonic()-began
        if index==1:diagnostic=core
        elif index>1:cores.append(core);times.append(whole);extra.append(whole-core)
    save('PARALLEL_PROBE.json',dict(status='PASS',GPU=os.environ['CUDA_VISIBLE_DEVICES'],steps=10,
          core_seconds=float(np.median(cores)),end_to_end_seconds=float(np.median(times)),
          extra_seconds=float(np.median(extra)),diagnostic_seconds=diagnostic,validation_images=0))


def main():
    for d in ('stages','sealed','transfer_requests','transfer_acks','batch_checks'):(PRIVATE/d).mkdir(exist_ok=True)
    run=Run()
    try:
        role=os.environ.get('N78_ROLE')
        if role=='engineering':engineering(run)
        elif role=='verify':verify_engineering(run)
        elif role=='parallel_probe':parallel_probe(run)
        elif role=='next_probe':
            if CFG.get('experiment') in ('NB-RL-A3','NB-RL-A4'):
                from preflight_nb_rl_a3 import check
            else:
                from preflight_nb_rl_a2 import check
            check(run)
        elif role=='analyze':
            qualification();run.phase='report'
            if CFG.get('experiment') == 'NB-RL-A5':
                from report_nb_rl_a5 import report
            elif CFG.get('experiment') == 'NB-RL-A4':
                from report_nb_rl_a4 import report
            elif CFG.get('experiment') == 'NB-RL-A3':
                from report_nb_rl_a3 import report
            elif CFG.get('experiment') == 'NB-RL-A2':
                from report_nb_rl_a2 import report
            else:
                from report_nb_rl_a1 import report
            report(json.loads((PUB/'PREDICTIONS_LOCK.json').read_text())['entries'],ROOT)
            save('WORKER_COMPLETE.json',dict(status='COMPLETE',role=role,counts=run.counts,unix=time.time()))
        elif role in ('train','frozen','evaluate'):
            protocol=qualification();job=json.loads(os.environ['N78_JOB'])
            assert WORKER and not (OUT/'WORKER_COMPLETE.json').exists()
            if role=='train':
                entries=train_matrix(run,protocol['epochs'],jobs=[(job['seed'],job['method'])])
            elif role=='frozen':
                entries=json.loads((PUB/'TRAINED_MATRIX_LOCK.json').read_text())['entries']
                old=len(entries);entries=frozen_matrix(run,entries,pairs=[(job['seed'],job['method'],job['owner'])])[old:]
            else:
                all_entries=json.loads((PUB/'ALL_STATES_LOCK.json').read_text())['entries']
                entries=evaluate_matrix(run,[all_entries[i] for i in job['indices']])
            save('WORKER_COMPLETE.json',dict(status='COMPLETE',role=role,entries=entries,counts=run.counts,unix=time.time()))
        else:
            assert role=='formal'
            protocol=qualification()
            assert run.counts.get('formal_steps',0)==0,'BLOCKED_NEEDS_EXPLICIT_RECOVERY_AUDIT'
            entries=train_matrix(run,protocol['epochs']);entries=frozen_matrix(run,entries)
            predictions=evaluate_matrix(run,entries);run.phase='report'
            from report_nb_rl_a1 import report
            report(predictions,ROOT);check_backlog(wait=True);run.resources()
            acks=[json.loads(p.read_text()) for p in (PRIVATE/'transfer_acks').glob('*.json')]
            save('BACKUP_REPORT.json',dict(status='PASS',requests=len(acks),all_independent_SHA_reread=all(a['independent_SHA_reread'] for a in acks)))
            save('ACCESS_LEDGER.json',run.resources())
            save('COMPLETE.json',dict(execution_status='COMPLETE',matrix_status='COMPLETE',stage_rows=96,class_rows=480,
                 new_steps=run.counts['formal_steps'],epochs=protocol['epochs'],NEXT_DECISION='STOP',publication_status='AWAITING_AUTHORIZATION'))
            enqueue(list(PUB.iterdir()),'completion_receipt',wait=True)
            save('FINAL_BACKUP_ACK.json',dict(status='PASS',all_requests_acknowledged=True,unix=time.time()))
    except BaseException as error:
        write_json(PRIVATE/('failure_'+str(time.time_ns())+'.json'),dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        save('RUN_STATUS.json',dict(status='INCOMPLETE_BUDGET' if 'BUDGET' in str(error) else 'BLOCKED',reason=str(error),phase=run.phase,unix=time.time()))
        raise
    finally:run.resources(False)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
