"""Isolated CT2-D forensic worker. No CT1 Run object, training loop or optimizer."""
import copy,csv,gc,json,math,multiprocessing as mp,os,resource,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from run_medical_v2 import ROOT,Learner,Images,seed_all,extend_embedding,sha,network_hash,WEIGHT_SHA
from run_hyperkvasir_hk1_m1 import tensor_digest
from check_isic_a1_linear_native import install_batched_linear
from report_locked_holdout_r1 import read,write,csvwrite
from ct1_statistics import joint,ridge,risk,fit_map,transport
from ct2d_math import metrics,classify,moments,factorial,independent_checks

SOURCE='8e63785896d17126114a436027d1cb0890b4658f'
DELIVERY='345bd7714662a8b7404c4609770df4bba4808451'
def jl(path,x):
    with Path(path).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def group(k):
    if 'prompt_key' in k:return 'few_key' if '.pool_few.' in k else 'main_key'
    for prefix,label in [('.pool_few.','few_adapter'),('.pool.','main_adapter'),('.assigner.','assigner'),('.head_few.','few_head'),('.head.','main_head')]:
        if prefix in k:return label
    return 'core'


class CountedImages(Images):
    def __getitem__(self,i):
        ans=super().__getitem__(i)
        with self.counts.get_lock():
            self.counts[0 if self.split=='train' else 2]+=1
            if self.split=='train' and ans[2]<self.final_known:self.counts[1]+=1
        return ans


class Diagnosis:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['root']);self.pub=self.root/'output/public';self.private=self.root/'output/private'
        self.old=Path(cfg['ct1_root']);self.op=self.old/'output/public';self.os=self.old/'output/private/sealed'
        self.lock=read(self.op/'DATA_AND_TASK_LOCK.json');self.lineage=read(self.op/'MODEL_LINEAGE.json')
        if cfg['mode']=='formal':
            assert cfg.get('ct2d_code_sha256'),'BLOCKED_UNBOUND_DIAGNOSTIC_SOURCE'
            assert all(sha(ROOT/f)==h for f,h in cfg['ct2d_code_sha256'].items()),'BLOCKED_DIAGNOSTIC_SOURCE_DRIFT'
        self.counts=mp.Array('q',[0,0,0]);self.started=time.monotonic();self.gpu_started=None
        self.prior=read(self.pub/'RESOURCE_AND_ACCESS_LEDGER.json') if (self.pub/'RESOURCE_AND_ACCESS_LEDGER.json').exists() else {}
        self.calls=dict(self.prior.get('calls',{}));self.maxactive=self.prior.get('active_peak_bytes',0)
        self.minfree=shutil.disk_usage(self.root).free;self.allowed=set();self.restore_rows=[]
        for spec in cfg['datasets'].values():
            for split in ('train','val'):
                with (Path(spec['manifests'])/(split+'.csv')).open() as f:rs=list(csv.DictReader(f))
                self.allowed.update(str((Path(spec['images'])/r['relative_path']).resolve()) for r in rs)
        oldroot=self.old.resolve()
        def guard(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0]));rp=p.resolve()
            if p.suffix.lower() in ('.jpg','.jpeg','.png'):assert str(rp) in self.allowed,'BLOCKED_DIAGNOSTIC_IMAGE_BOUNDARY'
            if p.suffix in ('.csv','.npz','.npy','.pt'):
                assert not any(x.lower().startswith(('test','reserved')) for x in p.parts),'BLOCKED_TEST_ACCESS'
            flags=args[2] if len(args)>2 else 0
            if flags and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
                assert not rp.is_relative_to(oldroot),'BLOCKED_ORIGINAL_WRITE'
        sys.addaudithook(guard)
        def no_step(*a,**kw):raise AssertionError('BLOCKED_OPTIMIZER_STEP')
        torch.optim.Optimizer.step=no_step;torch.optim.AdamW.step=no_step;torch.optim.SGD.step=no_step

    def count(self,k,n=1):self.calls[k]=self.calls.get(k,0)+n
    def forward(self,net,x,**kw):
        self.count('wrapper_forward_calls');self.count('encoder_forward_calls',3);self.count('encoder_image_rows',3*len(x))
        return net(x,**kw)
    def ledger(self,enforce=True):
        files=[p for p in self.root.rglob('*') if p.is_file() and not p.is_symlink()]
        size=sum(p.stat().st_size for p in files);persistent=sum(p.stat().st_size for p in files if '/private/tmp/' not in str(p) and p.name not in ('parent.pt','parent.pt.part'))
        self.maxactive=max(self.maxactive,size);self.minfree=min(self.minfree,shutil.disk_usage(self.root).free)
        gpu=self.prior.get('GPU_process_residence_seconds',0)+(time.monotonic()-self.gpu_started if self.gpu_started else 0)
        cpu_analytic=self.calls.get('CPU_analytic_milliseconds',0)/1000
        x=dict(status='RUNNING',new_neural_training_epochs=0,optimizer_steps=0,calls=self.calls,
          backward_probe_calls=self.calls.get('backward_probe_calls',0),new_analytic_fits=self.calls.get('new_analytic_fits',0),
          encoder_forward_calls=self.calls.get('encoder_forward_calls',0),
          diagnostic_train_image_reads=self.prior.get('diagnostic_train_image_reads',0)+self.counts[0],
          diagnostic_old_train_image_reads=self.prior.get('diagnostic_old_train_image_reads',0)+self.counts[1],
          diagnostic_val_image_reads=self.prior.get('diagnostic_val_image_reads',0)+self.counts[2],
          new_test_image_reads=0,new_test_feature_reads=0,new_test_prediction_reads=0,new_test_model_forwards=0,new_test_predictions=0,
          original_ct1_artifacts_modified=False,oracle_diagnostics_are_valid_online_CIL_results=False,further_training_started=False,
          monitoring_tasks_created=0,GPU_process_residence_seconds=gpu,active_bytes=size,persistent_bytes=persistent,
          CPU_analytic_seconds=cpu_analytic,CPU_other_engineering_reserved_seconds=600,
          active_peak_bytes=self.maxactive,min_free_bytes=self.minfree,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if self.gpu_started else 0)
        write(self.pub/'RESOURCE_AND_ACCESS_LEDGER.json',x)
        if enforce:assert gpu<10800 and cpu_analytic+600<7200 and size<=1024**3 and persistent<=512*1024**2 and self.minfree>=1024**3,'BLOCKED_RESOURCE_BUDGET'
        return x
    def dataset(self,name,seed,split,train=False,subset=None):
        order=self.lock[name]['orders'][str(seed)];spec=self.cfg['datasets'][name]
        ds=CountedImages(Path(spec['manifests'])/(split+'.csv'),spec['images'],order,range(len(order)),train)
        ds.rows.sort(key=lambda r:r['sample_id'])
        if subset is not None:ds.rows=[ds.rows[i] for i in subset]
        ds.counts=self.counts;ds.split=split;ds.final_known=len(order)-(3 if name=='HK' else 2)
        return ds
    def acquire(self,name,seed,stream,task):
        e=next(e for e in self.lineage if (e['dataset'],e['seed'],e['stream'],e['task'])==(name,seed,stream,task))
        request_id=f'{self.cfg["mode"]}_{name}_{seed}_{stream}_{task}_{time.time_ns()}'
        write(self.private/'PARENT_REQUEST.json',dict(request_id=request_id,name=e['name'],sha256=e['sha256']))
        end=time.monotonic()+200
        while time.monotonic()<end:
            a=self.private/'PARENT_ACK.json'
            if a.exists() and read(a)['request_id']==request_id:break
            time.sleep(1)
        else:raise TimeoutError('BLOCKED_PARENT_TRANSFER')
        p=self.private/'parent.pt.part';assert sha(p)==e['sha256'],'BLOCKED_PARENT_SHA'
        p.replace(self.private/'parent.pt');self.ledger()
        return torch.load(self.private/'parent.pt',map_location='cpu',weights_only=False),e
    def restore(self,p,e):
        if self.gpu_started is None:
            self.gpu_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        assert p['code_sha256']==read(self.op/'SOURCE_COMMIT_BINDING.json')['qualified_code_sha256']
        assert p['weight_sha256']==WEIGHT_SHA and p['network_sha256']==e['network_sha256']
        assert p['order']==self.lock[p['dataset']]['orders'][str(p['seed'])]
        assert p['manifest_sha256']==self.lock[p['dataset']]['manifest_sha256']
        args=copy.deepcopy(p['args']);args.update(device=[torch.device('cuda:0')],locked_weight_path=self.cfg['weight'])
        seed_all(p['seed']);learner=Learner(args);net=learner._network;extend_embedding(net.backbone.assigner,p['seed']);del learner
        allowed={'backbone.'+k for k in net.backbone.weight_load_audit['allowed_missing']}
        assert allowed==set(p['nonshared_keys'])=={k for k,v in net.named_parameters() if v.requires_grad}
        assert tensor_digest((k,v) for k,v in net.state_dict().items() if k not in allowed)==p['shared_sha256']
        state=net.state_dict();state.update(p['delta']);net.load_state_dict(state,strict=True)
        assert network_hash(net)==p['network_sha256'];net.cuda().eval()
        if not (self.pub/'SYNTHESIS_RNG_ISOLATION.json').exists():
            from types import SimpleNamespace
            from run_ct1 import CTLearner
            sampler=SimpleNamespace(stream='R',_known_classes=p['known'],epoch_memory=p['raw_memory'],
                                     synth=torch.Generator(device='cuda').manual_seed(46001))
            cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();own=sampler.synth.get_state().clone()
            CTLearner._concm_stage1_sample_memory(sampler)
            assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
            assert not torch.equal(own,sampler.synth.get_state())
            write(self.pub/'SYNTHESIS_RNG_ISOLATION.json',dict(status='PASS',original_CT1_sampler_exercised=True,
                  global_CPU_and_CUDA_RNG_unchanged=True,independent_synthesis_RNG_advanced=True,original_Run_constructed=False))
        row=dict(dataset=p['dataset'],seed=p['seed'],stream=p['stream'],task=p['task']+1,status='PASS',
                 file_sha256=e['sha256'],network_sha256=p['network_sha256'],shared_core_sha256=p['shared_sha256'],
                 trainable_whitelist_by_name={k:group(k) for k in sorted(allowed)})
        jl(self.pub/'restore_audit.jsonl',row);return net,allowed
    def pointwise(self,net):
        net.eval().requires_grad_(False);install_batched_linear(net)
        net.backbone.pool.batchwise_prompt=False;net.backbone.pool_few.batchwise_prompt=False
    def extract(self,net,p,split,tag,subset=None):
        ds=self.dataset(p['dataset'],p['seed'],split,subset=subset);n=len(ds);temp=self.private/'tmp'
        raw=np.lib.format.open_memmap(temp/(tag+'_'+split+'.npy'),mode='w+',dtype='float32',shape=(n,2,768))
        routes=np.empty((n,2),np.int64);scores=np.empty((n,len(p['order'])),np.float32);ys=[]
        loader=DataLoader(ds,batch_size=48,shuffle=False,num_workers=4,drop_last=False,generator=torch.Generator().manual_seed(46001))
        offset=0;start=time.monotonic()
        with torch.inference_mode():
            for _,x,y in loader:
                o=self.forward(net,x.cuda(),train=False,weight=None);nn=len(x)
                raw[offset:offset+nn]=torch.stack([o['pre_logits'],o['pre_logits_few']],1).cpu().numpy()
                routes[offset:offset+nn]=torch.stack([o['prompt_idx'].flatten(),o['prompt_idx_few'].flatten()],1).cpu().numpy()
                scores[offset:offset+nn]=(o['logits']+o['logits_few']).cpu().numpy();ys.append(y.numpy());offset+=nn
        raw.flush();self.ledger();assert offset==n and np.isfinite(raw).all()
        return dict(raw=raw,y=np.concatenate(ys),routes=routes,head=scores,rows=ds.rows,seconds=time.monotonic()-start)
    def save_score(self,p,mode,W,val):
        raw=joint(val['raw'])@W;stem=f'{p["dataset"]}_{p["seed"]}_{p["stream"]}_{mode}'
        rows=val['rows'];np.savez(self.private/'scores'/(stem+'.npz'),raw=raw,y=val['y'],order=np.array(p['order']),
             ids=np.array([x['sample_id'] for x in rows]),original=np.array(p['order'])[val['y']],
             component=np.array([x.get('identity_component',x.get('verified_group','')) for x in rows]),W=W)
        return raw


def source_audit(r):
    binding=read(r.op/'SOURCE_COMMIT_BINDING.json');assert binding['source_commit']==SOURCE
    assert binding['qualified_code_sha256']==read(r.op/'CODE_ENV_LOCK.json')['code_sha256']
    for f,h in binding['qualified_code_sha256'].items():assert sha(r.old/'code'/f)==h and sha(ROOT/f)==h,(f,'SOURCE_DRIFT')
    assert sha(r.cfg['weight'])==WEIGHT_SHA
    expected=0;per_task=[]
    epochs=lines(r.op/'train_epoch_metrics.jsonl');lineage=r.lineage
    for name in ('HK','ISIC'):
        spec=r.cfg['datasets'][name]
        for split in ('train','val'):assert sha(Path(spec['manifests'])/(split+'.csv'))==spec['manifest_sha256'][split]
        with (Path(spec['manifests'])/'train.csv').open() as f:rows=list(csv.DictReader(f))
        for seed in (1993,1994,1995):
            for stream in ('U','R'):
                prev=None
                for t in r.lock[name]['runs'][str(seed)]:
                    n=sum(int(x['original_label']) in t['classes'] for x in rows);steps=10*math.ceil(n/48);expected+=steps
                    ls=[x for x in epochs if (x['dataset'],x['seed'],x['stream'],x['task'])==(name,seed,stream,t['task'])]
                    assert len(ls)==10 and sum(x['optimizer_steps'] for x in ls)==steps==t['steps']
                    e=next(x for x in lineage if (x['dataset'],x['seed'],x['stream'],x['task'])==(name,seed,stream,t['task']))
                    assert e['previous_task_network_sha256']==prev;prev=e['network_sha256']
                    per_task.append(dict(dataset=name,seed=seed,stream=stream,task=t['task'],n_train=n,expected_steps=steps))
    assert expected==32340 and len(epochs)==900 and len(lineage)==90
    math_audit=independent_checks();r.count('new_analytic_fits',3);r.count('engineering_analytic_fits',3)
    write(r.pub/'MATHEMATICAL_AUDIT.json',math_audit)
    csvwrite(r.pub/'task_step_recalculation.csv',per_task)
    lock=read(r.op/'TRAJECTORIES_LOCK.json');assert len(lock['sealed_units'])==270
    ms=[];pcs=[];ers=[]
    for e in lock['sealed_units']:
        path=r.os/e['file'];assert sha(path)==e['sha256']
        with np.load(path) as f:p={k:f[k].copy() for k in f.files}
        assert len(p['order'])>=e['seen'] and np.array_equal(p['order'][p['y']],p['original'])
        assert list(p['ids'])==sorted(p['ids']) and len(set(p['ids']))==len(p['ids'])
        meta={k:e[k] for k in ('dataset','seed','task','method','seen','known')}
        m,pc,er=metrics(p['raw'],p['y'],p['order'][:e['seen']],e['known'],r.cfg['frequency_groups'][e['dataset']],meta,p['component'])
        ms.append(m);pcs+=pc;ers+=er
    assert len(ms)==270 and len(pcs)==2754
    with (r.op/'validation_all_metrics.csv').open() as f:old=list(csv.DictReader(f))
    keys=lambda m:(m['dataset'],int(m['seed']),int(m['task']),m['method'])
    lookup={keys(m):m for m in old}
    for m in ms:
        for k in ('balanced_accuracy','accuracy','old_macro_recall','current_macro_recall','tail_recall'):
            a=lookup[keys(m)][k];b=m[k]
            assert (a=='' and b is None) or (b is not None and abs(float(a)-b)<1e-10),(keys(m),k)
    with (r.op/'validation_all_per_class.csv').open() as f:oldpc=list(csv.DictReader(f))
    lookup={keys(m)+(int(m['original_label']),):m for m in oldpc}
    for m in pcs:
        o=lookup[keys(m)+(m['original_label'],)];assert int(o['n_correct'])==m['n_correct'] and int(o['n_images'])==m['n_images']
    csvwrite(r.pub/'recomputed_ct1_validation_metrics.csv',ms);csvwrite(r.pub/'recomputed_ct1_validation_per_class.csv',pcs)
    csvwrite(r.pub/'restricted_candidate_diagnostics.csv',ms);csvwrite(r.pub/'recomputed_error_decomposition.csv',ers)
    forget=[]
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            for method in {m['method'] for m in ms}:
                for c in range(23 if name=='HK' else 8):
                    rs=sorted([m for m in pcs if (m['dataset'],m['seed'],m['method'],m['original_label'])==(name,seed,method,c)],key=lambda x:x['task'])
                    values=[x['recall'] for x in rs];age=rs[-1]['task']-rs[0]['task']
                    forget.append(dict(dataset=name,seed=seed,method=method,original_label=c,age=age,arrival_task=rs[0]['task'],
                         first_recall=values[0],final_recall=values[-1],first_to_final=None if not age else values[0]-values[-1],max_to_final=None if not age else max(values)-values[-1]))
    csvwrite(r.pub/'recomputed_forgetting.csv',forget)
    transport_rows=lines(r.op/'transport_audit.jsonl')
    assert len(transport_rows)==780
    for q in transport_rows:assert q['anchor_version']==q['task']-1 and q['committed']==(q['epoch']==10)
    csvwrite(r.pub/'transport_log_summary.csv',[dict(dataset=q['dataset'],seed=q['seed'],stream=q['stream'],task=q['task'],epoch=q['epoch'],**q['joint'],old_trace=q['old_trace'],e_mean=q['e_mean']) for q in transport_rows])
    riskrows=lines(r.op/'risk_solver_audit.jsonl')
    shutil.copyfile(r.op/'risk_solver_audit.jsonl',r.pub/'original_risk_solver_audit.jsonl')
    for q in riskrows:assert q['status']=='optimal' and q['normalized_max_constraint_violation']<=1e-6
    csvwrite(r.pub/'original_risk_slack_audit.csv',[dict(dataset=q['dataset'],seed=q['seed'],task=q['task'],status=q['status'],iterations=q['iterations'],primal_residual=q['primal_residual'],dual_residual=q['dual_residual'],classes=len(q['xi']),xi_ge_delta=int(np.sum(np.asarray(q['xi'])>=.1)),xi_max=max(q['xi'])) for q in riskrows])
    write(r.pub/'SOURCE_AND_PROTOCOL_AUDIT.json',dict(status='PASS',delivery_commit=DELIVERY,execution_source_commit=SOURCE,
          verified_source_files=len(binding['qualified_code_sha256']),task_states=90,trajectories=12,epochs=900,steps=expected,
          recomputed_metric_rows=len(ms),recomputed_per_class_rows=len(pcs),all_original_metrics_reproduced=True,
          immutable_anchor_source_review=True,epoch_mapping_not_recomposed=True,
          original_adapter_counter_scope='Nested .pool. up/down projection names include both main and few adapters; excludes keys, assigner and heads',
          independent_fits_in_math_gate=3))


def gaussian(p,known,count,device,cap=None):
    g=torch.Generator(device=device).manual_seed(46001+p['seed']);parts=[[],[]];ys=[]
    for c in range(known):
        s=p['raw_memory'][c]
        for branch in (0,1):
            mu=torch.as_tensor(s['mu'][branch],dtype=torch.float32,device=device);v=torch.as_tensor(s['var'][branch],dtype=torch.float32,device=device)
            parts[branch].append(mu+torch.randn((count,768),device=device,generator=g)*torch.sqrt(v.clamp_min(0)+1e-6))
        ys.extend([c]*count)
    m,f=[torch.cat(x) for x in parts];y=torch.tensor(ys,device=device)
    if cap and len(y)>cap:
        ix=torch.randperm(len(y),generator=g,device=device)[:cap];m=m[ix];f=f[ix];y=y[ix]
    return m,f,y


def gradients(r,net,p,allowed,final_known,stage):
    name=p['dataset'];seed=p['seed'];C=len(p['order']);ds=r.dataset(name,seed,'train',train=True)
    indices=[]
    for c in range(final_known,C):indices += [i for i,z in enumerate(ds.rows) if z['target']==c][:8]
    seed_all(46001);xs=[ds[i] for i in indices];x=torch.stack([v[1] for v in xs]).cuda();y=torch.tensor([v[2] for v in xs],device='cuda')
    before=network_hash(net);net.train()
    for k,v in net.named_parameters():v.requires_grad_(k in allowed)
    params=[(k,v) for k,v in net.named_parameters() if v.requires_grad];allrows=[];vectors={}
    synth=gaussian(p,final_known,4,'cuda',48)
    for mult in (1,2):
        xx=x.repeat(mult,1,1,1);yy=y.repeat(mult);counts=torch.tensor(p['args']['lt_list'],device='cuda')[yy]
        o=r.forward(net,xx,task_id=p['task'],train=True,weight=counts);pool=o['pool_id'].squeeze(1)
        logits=o['logits'][:,:C];few=o['logits_few'][:,:C]
        terms={'main':F.cross_entropy(logits,yy),'sum':F.cross_entropy(logits+few,yy),
               'few':-(pool*(F.one_hot(yy,C)*torch.log(torch.softmax(few,-1)+1e-7)).sum(1)).sum()}
        terms['assignment']=(((torch.where(counts<=100,1.,.1)-pool)**2)*torch.where(counts<=100,10.,1.)).sum() if stage=='pre' else ((1-pool)**2*pool*10).sum()
        terms['pull']=-.1*(o['reduce_sim']+o['reduce_sim_few'])
        sm,sf,sy=synth;terms['synthetic']=F.cross_entropy(net.backbone.head(sm.repeat(mult,1))[:,:C]+net.backbone.head_few(sf.repeat(mult,1))[:,:C],sy.repeat(mult))
        terms['total']=(terms['main']+terms['sum']+terms['few'])/3+terms['assignment']+terms['pull']+(final_known/(C-final_known) if p['stream']=='R' else 0)*terms['synthetic']
        grad={}
        for term,loss in terms.items():
            gs=torch.autograd.grad(loss,[v for _,v in params],retain_graph=True,allow_unused=True);r.count('backward_probe_calls')
            bygroup={}
            for (k,v),g in zip(params,gs):bygroup.setdefault(group(k),[]).append(torch.zeros(v.numel()) if g is None else g.detach().cpu().flatten())
            grad[term]={k:torch.cat(v) for k,v in bygroup.items()}
            norms={k:float(v.double().norm()) for k,v in grad[term].items()}
            if term=='synthetic':assert all(norms[k]==0 for k in norms if 'head' not in k)
            allrows.append(dict(multiplier=mult,batch=len(xx),term=term,value=float(loss.detach()),gradient_norms=norms))
        vectors[mult]=grad
        allrows.append(dict(multiplier=mult,batch=len(xx),term='pool_diagnostic',unweighted_few_CE=float(F.cross_entropy(few,yy).detach()),
              pool_mean=float(pool.detach().mean()),pool_min=float(pool.detach().min()),pool_max=float(pool.detach().max()),
              pool_near_zero=float((pool.detach()<.01).float().mean()),pool_near_one=float((pool.detach()>.99).float().mean()),
              pool_histogram=torch.histc(pool.detach(),bins=10,min=0,max=1).cpu().tolist()))
        del o,terms,gs
    comparisons=[]
    for term in ('main','sum','few','assignment','pull','synthetic'):
        vals=[next(q['value'] for q in allrows if q['term']==term and q['multiplier']==mult) for mult in (1,2)]
        np.testing.assert_allclose(vals[1],vals[0]*(2 if term in ('few','assignment') else 1),rtol=1e-4,atol=1e-5,
                                   err_msg='BLOCKED_REDUCTION_SCALE_PROBE')
    for term in vectors[1]:
        for g in vectors[1][term]:
            a=vectors[1][term][g].double();b=vectors[2][term][g].double();aa=float(a.norm());bb=float(b.norm())
            comparisons.append(dict(term=term,group=g,B_norm=aa,double_B_norm=bb,ratio=None if aa==0 else bb/aa,
                cosine=None if aa*bb==0 else float(torch.dot(a,b)/(aa*bb))))
    cosines=[]
    for a in vectors[1]:
        for b in vectors[1]:
            if a>=b:continue
            for g in vectors[1][a]:
                xg=vectors[1][a][g].double();yg=vectors[1][b][g].double();den=float(xg.norm()*yg.norm())
                cosines.append(dict(term_a=a,term_b=b,group=g,inner_product=float(torch.dot(xg,yg)),cosine=None if den==0 else float(torch.dot(xg,yg)/den)))
    assert all(v.grad is None for k,v in net.named_parameters() if k not in allowed)
    net.zero_grad(set_to_none=True);net.eval();assert network_hash(net)==before
    filename='gradient_probes.jsonl' if r.cfg['mode']=='formal' else 'gradient_engineering.jsonl'
    jl(r.pub/filename,dict(dataset=name,seed=seed,stream=p['stream'],stage=stage,source_task=p['task']+1,
          candidate_classes=C,current_classes=list(range(final_known,C)),image_count=len(indices),rows=allrows,
          B_to_double_B=comparisons,within_B_gradient_inner_products=cosines,synthetic_direct_adapter_gradient_zero=True,
          risk_outside_torch_graph=True,model_unchanged=True,optimizer_steps=0,historical_minibatch_replay=False))


def synthetic_diagnostic(r,net,p,train):
    known=p['known'];order=np.array(p['order']);C=len(order);m,f,y=gaussian(p,known,32,'cuda')
    mu=np.stack([p['raw_memory'][c]['mu'] for c in range(known)])
    cases=[('synthetic32',m,f,y.cpu().numpy()),('stored_means',torch.tensor(mu[:,0],dtype=torch.float32,device='cuda'),torch.tensor(mu[:,1],dtype=torch.float32,device='cuda'),np.arange(known))]
    ix=train['y']<known;real=np.asarray(train['raw'][ix]);cases.append(('real_old_fit',torch.tensor(real[:,0],device='cuda'),torch.tensor(real[:,1],device='cuda'),train['y'][ix]))
    for label,m,f,ys in cases:
        with torch.no_grad():logits=(net.backbone.head(m)+net.backbone.head_few(f)).cpu().numpy()
        pred=classify(logits,order);masked=logits.copy();masked[np.arange(len(ys)),ys]=-np.inf
        ce=np.logaddexp.reduce(logits.astype(np.float64),axis=1)-logits[np.arange(len(ys)),ys]
        for c in range(known):
            mask=ys==c;rr=real[train['y'][ix]==c];a=rr[:,0].astype(np.float64);b=rr[:,1].astype(np.float64)
            a-=a.mean(0);b-=b.mean(0);den=np.sqrt(np.sum(a*a)*np.sum(b*b))
            jl(r.pub/'synthetic_diagnostics.jsonl',dict(dataset=p['dataset'],seed=p['seed'],stream='R',case=label,
               original_label=int(order[c]),n=int(mask.sum()),n_correct=int(np.sum(pred[mask]==c)),
               recall=100*float(np.mean(pred[mask]==c)),old_to_current=100*float(np.mean(pred[mask]>=known)),
               CE=float(ce[mask].mean()),margin=float(np.mean(logits[mask,c]-masked[mask].max(1))),
               real_main_few_centered_correlation=None if den==0 else float(np.sum(a*b)/den),synthetic_branch_independence=True))


def risk_details(state,W0,sol,val,meta):
    c=len(state['n']);v=state['v'];alpha=10/(state['n']+10)
    vt=(1-alpha)*v+alpha*v.mean(1,keepdims=True)+1e-8/len(W0);u=vt/state['n']+state['e'];W=sol['W'];margins=[]
    for j in range(c):
        rr=[]
        for k in range(c):
            if j==k:continue
            q=W[:,j]-W[:,k];rr.append(float(state['mu'][:,j]@q-np.sqrt(np.sum(vt[:,j]*q*q))-np.sqrt(np.sum(u[:,j]*q*q))))
        margins.append(min(rr))
    B=sol['B'];H=W0.T@(state['S']/c+.001*np.eye(len(W0)))@W0;D=B-np.eye(c)
    return dict(meta,B_minus_I_norm=float(np.linalg.norm(D)),W_relative_change=float(np.linalg.norm(W-W0)/np.linalg.norm(W0)),
        no_slack_margin=margins,xi=sol['xi'].tolist(),xi_ge_delta=int(np.sum(sol['xi']>=.1)),
        fit_objective_increase=float(np.trace(D.T@H@D)),risk_objective_term=.1*float(sol['xi'].mean()),
        uniqueness_term=1e-10*float(np.sum(D*D)),prediction_flips=int(np.sum(classify(joint(val['raw'])@W,np.array(meta['order']))!=classify(joint(val['raw'])@W0,np.array(meta['order'])))))


def oracle(r,net,p,train,val):
    C=len(p['order']);known=p['known'];fresh,newS=moments(train['raw'],train['y'],range(known,C));stored=p['stats']
    for key in ('n','mu','v'):
        a=fresh[key][known:] if key=='n' else fresh[key][:,known:];b=stored[key][known:] if key=='n' else stored[key][:,known:]
        np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5,err_msg='BLOCKED_CURRENT_MOMENT_REPRODUCTION')
    for c in range(known,C):
        raw=np.asarray(train['raw'][train['y']==c],dtype=np.float64)
        np.testing.assert_allclose(raw.mean(0),p['raw_memory'][c]['mu'],atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(raw.var(0),p['raw_memory'][c]['var'],atol=1e-5,rtol=1e-5)
    states,cov_audit=factorial(stored,fresh,newS,known);order=np.array(p['order'])
    name=f'{p["dataset"]}_{p["seed"]}_{p["stream"]}';method='CT-J-CB' if p['stream']=='U' else 'CT-ConCM-CB'
    with np.load(r.os/f'{name}_W_t{p["task"]+1:02d}.npz') as f:oldW=f[method].copy()
    with np.load(r.os/f'{p["dataset"]}_{p["seed"]}_{method}_t{p["task"]+1:02d}.npz') as f:oldscore={k:f[k].copy() for k in f.files}
    assert list(oldscore['ids'])==[x['sample_id'] for x in val['rows']]
    fitrecords=[];Ws={}
    for mode,st in states.items():
        W,diag=ridge(st);r.count('new_analytic_fits');Ws[mode]=W
        scores=r.save_score(p,mode,W,val)
        if mode=='Q00':
            np.testing.assert_allclose(W,oldW,atol=1e-5,rtol=1e-5)
            np.testing.assert_allclose(scores,oldscore['raw'],atol=1e-5,rtol=1e-5)
            newpred=classify(scores,order);oldpred=classify(oldscore['raw'],order);changed=newpred!=oldpred
            if changed.any():
                np.savez(r.private/'scores'/(name+'_reproduction_mismatch.npz'),ids=oldscore['ids'][changed],
                          old=oldscore['raw'][changed],new=scores[changed],y=oldscore['y'][changed])
                write(r.pub/'REPRODUCTION_MISMATCH.json',dict(dataset=p['dataset'],seed=p['seed'],stream=p['stream'],
                    rows=int(changed.sum()),max_logit_difference=float(np.max(abs(scores-oldscore['raw']))),
                    changed_old_top2_margins=(np.sort(oldscore['raw'][changed],axis=1)[:,-1]-np.sort(oldscore['raw'][changed],axis=1)[:,-2]).tolist(),
                    private_per_sample_evidence=True))
            assert not changed.any(),'BLOCKED_Q00_PREDICTION_DRIFT'
        fm,_,_=metrics(joint(train['raw'])@W,train['y'],order,known,r.cfg['frequency_groups'][p['dataset']],{})
        fitrecords.append(dict(dataset=p['dataset'],seed=p['seed'],stream=p['stream'],mode=mode,fit_BA=fm['balanced_accuracy'],fit_accuracy=fm['accuracy'],**diag))
    z=joint(train['raw']);weights=1/(C*np.bincount(train['y'])[train['y']]);G=z.T@(weights[:,None]*z);R=z.T@(weights[:,None]*np.eye(C)[train['y']])
    independent=np.linalg.solve((G+G.T)/2+.001*np.eye(len(G)),R);r.count('new_analytic_fits');r.count('independent_batch_verification_fits')
    np.testing.assert_allclose(independent,Ws['Q11'],atol=1e-9,rtol=1e-9)
    if p['stream']=='U':
        sol,rec=risk(fresh,Ws['Q11']);r.count('new_analytic_fits');assert sol is not None,('PARTIAL_SOLVER',rec)
        r.save_score(p,'Q11-Risk',sol['W'],val)
        np.savez(r.private/'scores'/(name+'_risk_solution.npz'),**sol)
        fm,_,_=metrics(z@sol['W'],train['y'],order,known,r.cfg['frequency_groups'][p['dataset']],{})
        fitrecords.append(dict(dataset=p['dataset'],seed=p['seed'],stream=p['stream'],mode='Q11-Risk',fit_BA=fm['balanced_accuracy'],fit_accuracy=fm['accuracy']))
        jl(r.pub/'risk_counterfactual_audit.jsonl',dict(risk_details(fresh,Ws['Q11'],sol,val,dict(dataset=p['dataset'],seed=p['seed'],order=p['order'],mode='Q11-Risk')),solver=rec))
        with np.load(r.os/f'{p["dataset"]}_{p["seed"]}_risk_t{p["task"]+1:02d}.npz') as f:oldsol={k:f[k].copy() for k in f.files}
        jl(r.pub/'risk_counterfactual_audit.jsonl',risk_details(stored,oldW,oldsol,val,dict(dataset=p['dataset'],seed=p['seed'],order=p['order'],mode='CT1-stored-Risk')))
    for row in fitrecords:jl(r.pub/'oracle_fit_audit.jsonl',row)
    jl(r.pub/'reproduction_audit.jsonl',dict(dataset=p['dataset'],seed=p['seed'],stream=p['stream'],status='PASS',current_moments=True,
          Q00_weights_logits_predictions=True,Q11_independent_weighted_batch=True,covariance=cov_audit))


def geometry(r,p,pre,post,preval,postval,oldstate):
    known=p['known'];y=post['y'];assert np.array_equal(pre['y'],y)
    x=joint(pre['raw']);z=joint(post['raw']);cur=y>=known;a,b,err,diag=fit_map(x[cur],z[cur],y[cur])
    r.count('diagnostic_affine_fits')
    diag.update(a_quantiles=np.quantile(a,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
                a_histogram=np.histogram(a,bins=np.linspace(.5,2,31))[0].tolist())
    np.savez(r.private/'scores'/f'{p["dataset"]}_{p["seed"]}_{p["stream"]}_map.npz',a=a,b=b,residual=err)
    prevstats=oldstate['stats'];mapped=transport(prevstats,a,b,err,p['task']+1)
    for k in ('mu','v','e'):np.testing.assert_allclose(mapped[k],p['stats'][k][:,:known],atol=1e-5,rtol=1e-5)
    original=next(q for q in lines(r.op/'transport_audit.jsonl') if (q['dataset'],q['seed'],q['stream'],q['task'],q['epoch'])==(p['dataset'],p['seed'],p['stream'],p['task']+1,10))
    for k in ('a_mean','b_norm','residual_mean','clip_fraction'):np.testing.assert_allclose(diag[k],original['joint'][k],atol=1e-5,rtol=1e-5)
    meta=dict(dataset=p['dataset'],seed=p['seed'],stream=p['stream'],from_task=p['task'],to_task=p['task']+1)
    for split,xx,zz,yy,r0,r1 in [('fit',x,z,y,pre['routes'],post['routes']),('val',joint(preval['raw']),joint(postval['raw']),postval['y'],preval['routes'],postval['routes'])]:
        for scope,mask in [('old',yy<known),('current',yy>=known)]:
            residual=zz[mask]-xx[mask]*a-b
            jl(r.pub/'geometry.jsonl',dict(meta,split=split,scope=scope,diagnostic='map_generalization',n=int(mask.sum()),
                 coordinate_MSE=float(np.mean(residual**2)),RMS_L2=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
                 a_lower_clip_fraction=float(np.mean(a==.5)),a_upper_clip_fraction=float(np.mean(a==2)),**diag))
            for branch in (0,1):
                jl(r.pub/'routing.jsonl',dict(meta,split=split,scope=scope,pool='main' if branch==0 else 'few',n=int(mask.sum()),
                   switch_rate=float(np.mean(r0[mask,branch]!=r1[mask,branch])),pre_usage=np.bincount(r0[mask,branch],minlength=5).tolist(),post_usage=np.bincount(r1[mask,branch],minlength=5).tolist()))
        for version,features in [('pre',xx),('post',zz),('current_map_applied',xx*a+b)]:
            means=[]
            for c in range(len(p['order'])):
                q=features[yy==c];mu=q.mean(0);means.append(mu)
                jl(r.pub/'geometry.jsonl',dict(meta,split=split,scope='old' if c<known else 'current',diagnostic='class_geometry',version=version,
                    original_label=p['order'][c],age=p['task']+1-p['stats']['arrival'][c],n=len(q),mean_norm=float(np.linalg.norm(mu)),within_scatter=float(np.mean(np.sum((q-mu)**2,axis=1))),
                    stored_post_mean_error=float(np.linalg.norm(p['stats']['mu'][:,c]-mu)) if version=='post' else None,
                    true_post_mean_error=float(np.linalg.norm(zz[yy==c].mean(0)-mu))))
            means=np.array(means)
            for scope,ix in [('old',np.arange(known)),('current',np.arange(known,len(means)))]:
                mm=means[ix]
                jl(r.pub/'geometry.jsonl',dict(meta,split=split,scope=scope,diagnostic='between_class_geometry',version=version,
                    centered_between_scatter=float(np.mean(np.sum((mm-mm.mean(0))**2,axis=1))),n_classes=len(ix)))
            for c in range(len(means)):
                for k in range(c):jl(r.pub/'class_pair_distances.jsonl',dict(meta,split=split,version=version,label_a=p['order'][c],label_b=p['order'][k],distance=float(np.linalg.norm(means[c]-means[k]))))


def engineering(r):
    p,e=r.acquire('HK',1993,'U',11);net,allowed=r.restore(p,e);r.pointwise(net)
    ds=r.dataset('HK',1993,'train');ix=[i for i,x in enumerate(ds.rows) if x['target']>=20][:96]
    features=r.extract(net,p,'train','engineering',ix)
    x=features['raw'];assert len(x)==96
    # Pointwise schedule and batch companion controls, using the same decoded tensors.
    ds=r.dataset('HK',1993,'train',subset=ix[:2]);items=[ds[i] for i in range(2)];xx=torch.stack([v[1] for v in items]).cuda()
    with torch.inference_mode():
        one=r.forward(net,xx[:1],train=False,weight=None);two=r.forward(net,xx,train=False,weight=None)
    for k in ('pre_logits','pre_logits_few'):torch.testing.assert_close(one[k],two[k][:1],atol=1e-5,rtol=1e-5)
    assert network_hash(net)==p['network_sha256']
    seconds_per_batch=features['seconds']/2
    projected=math.ceil(220256/48)*seconds_per_batch*1.3+1200
    status='READY' if projected+r.ledger()['GPU_process_residence_seconds']<10800 else 'BLOCKED_GPU_BUDGET'
    write(r.pub/'RESOURCE_ADMISSION.json',dict(status=status,measured_two_batches_seconds=features['seconds'],projected_seconds=projected,
       full_final_model_image_rows=165192,additional_pre_model_rows=55064,GPU_budget_seconds=10800,active_limit_bytes=1024**3,persistent_limit_bytes=512*1024**2,
       pointwise_batch_companion=True,source_parent_restored=True))
    del features,net,p;gc.collect();torch.cuda.empty_cache()
    for f in (r.private/'tmp').glob('engineering_*.npy'):f.unlink()
    (r.private/'parent.pt').unlink();assert status=='READY',status


def formal(r):
    assert read(r.pub/'RESOURCE_ADMISSION.json')['status']=='READY'
    assert not (r.pub/'FORMAL_STARTED.json').exists(),'BLOCKED_ALREADY_RELEASED'
    write(r.pub/'FORMAL_STARTED.json',dict(status='RUNNING',source_commit=r.cfg['source_commit'],unix=time.time()))
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            for stream in ('U','R'):
                task=11 if name=='HK' else 4;p,e=r.acquire(name,seed,stream,task);net,allowed=r.restore(p,e)
                if seed==1993:gradients(r,net,p,allowed,p['known'],'post')
                r.pointwise(net);tr=r.extract(net,p,'train','post');va=r.extract(net,p,'val','post')
                began=time.monotonic();oracle(r,net,p,tr,va)
                r.count('CPU_analytic_milliseconds',int(1000*(time.monotonic()-began)))
                if stream=='R':synthetic_diagnostic(r,net,p,tr)
                assert network_hash(net)==p['network_sha256'],'BLOCKED_DIAGNOSTIC_STATE_MUTATION'
                del net;gc.collect();torch.cuda.empty_cache()
                if seed==1993:
                    pre,pe=r.acquire(name,seed,stream,task-1);pn,allowed=r.restore(pre,pe)
                    gradients(r,pn,pre,allowed,p['known'],'pre');r.pointwise(pn)
                    ptr=r.extract(pn,pre,'train','pre');pva=r.extract(pn,pre,'val','pre')
                    began=time.monotonic();geometry(r,p,ptr,tr,pva,va,pre)
                    r.count('CPU_analytic_milliseconds',int(1000*(time.monotonic()-began)))
                    assert network_hash(pn)==pre['network_sha256'],'BLOCKED_DIAGNOSTIC_STATE_MUTATION'
                    del pn,pre,ptr,pva;gc.collect();torch.cuda.empty_cache()
                del tr,va,p
                for f in (r.private/'tmp').glob('*.npy'):f.unlink()
                (r.private/'parent.pt').unlink();r.ledger();print('MODEL_COMPLETE',name,seed,stream,flush=True)
    write(r.pub/'SCORES_LOCK.json',dict(source_commit=r.cfg['source_commit'],
          files={p.name:sha(p) for p in sorted((r.private/'scores').glob('*.npz'))},formal_metric_units=54))
    write(r.pub/'GPU_PHASE_COMPLETE.json',dict(status='PASS',models=12,transitions=4,new_training_epochs=0,optimizer_steps=0))
    write(r.private/'STOP_TRANSFER.json',dict(status='STOP'))


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    cfg=read(os.environ['P21_CONFIG']);r=Diagnosis(cfg)
    try:
        with threadpool_limits(limits=4):
            if cfg['mode']=='audit':source_audit(r)
            elif cfg['mode']=='engineering':engineering(r)
            elif cfg['mode']=='gradient_gate':
                p,e=r.acquire('HK',1993,'R',11);net,allowed=r.restore(p,e)
                gradients(r,net,p,allowed,p['known'],'post')
                assert network_hash(net)==p['network_sha256']
                write(r.pub/'GRADIENT_ENGINEERING_PASS.json',dict(status='PASS',unchanged_model=True,optimizer_steps=0))
                del net,p;(r.private/'parent.pt').unlink()
            elif cfg['mode']=='formal':formal(r)
            else:raise ValueError(cfg['mode'])
    except BaseException as e:
        write(r.pub/('FAILURE_'+cfg['mode']+'.json'),dict(status='BLOCKED',exception=type(e).__name__,reason=str(e)))
        raise
    finally:r.ledger(False)

if __name__=='__main__':main()
