"""Frozen CPU production fixtures: real callers, synthetic inputs, no medical data."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import run_module_additions as training
import run_paired_transport_cycle as pcycle
import run_snapshot_guard_scope as scope
import run_snapshot_guard_worker as worker
import snapshot_guard_production as guard
import paired_moment_transport as pmt
import paired_transport_design as affine
import prototype_coherent as native
import core1_competition as competition
from check_lca_production import Encoder,SyntheticImages,IndexedSynthetic,features,rejected
from check_paired_transport_production import paired_features
from check_full_fit_guard_design import sample,label_counts,bank


def exact(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def chain(config):
    scope.validate(config)
    for key,value in [('gpus',[0,2]),('jobs',config['jobs'][:2])]:rejected(lambda:scope.validate(dict(config,**{key:value})))
    for key,value in [('snapshot_guard_version','bad'),('guard_scoring_dtype','float32'),('expected_steps',1),('guard_cap',128),('evaluation_split','official_test')]:
        c=copy.deepcopy(config['jobs'][0]);c[key]=value;rejected(lambda:scope.validate_job(c))
    spec=importlib.util.spec_from_file_location('frozen_original',os.environ['Q239_ORIGINAL']);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    as_tensor=torch.as_tensor;empty=native.empty;set_threads=torch.set_num_threads
    def tensor(*a,**k):
        if str(k.get('device','')).startswith('cuda'):k['device']='cpu'
        return as_tensor(*a,**k)
    def loader(*a,**k):
        for key in ('multiprocessing_context','prefetch_factor','persistent_workers'):k.pop(key,None)
        k['num_workers']=0;return DataLoader(*a,**k)
    class CountEncoder(Encoder):
        calls=0;images=0
        def forward(self,x):
            CountEncoder.calls+=1;CountEncoder.images+=len(x);return super().forward(x)
    steps=stage_checks=preflights=events=invalidations=evals=0;maximum=0.;training_effect=False;records=[]
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp,patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self), \
        patch.object(torch,'as_tensor',tensor),patch.object(torch.cuda,'manual_seed_all',lambda *a:None), \
        patch.object(torch.cuda,'max_memory_allocated',lambda *a:0),patch.object(native,'empty',lambda dim,device='cpu':empty(dim,'cpu')), \
        patch.object(torch,'set_num_threads',lambda n:set_threads(2)),patch.object(pcycle,'validate_job',lambda c:None), \
        patch.object(scope,'validate_job',lambda c:None),patch.object(pmt,'extract',paired_features):
        for ds in ('ISIC','HK'):
            rows=[dict(label=c,sample=i,relative_path=f'{c}_{i}',identity_component=f'{c}_{i}') for c in scope.ORDERS[ds] for i in range(8)]
            for module in (original,training):
                module.ApartFeatures=CountEncoder;module.Images=SyntheticImages;module.IndexedImages=IndexedSynthetic;module.DataLoader=loader;module.manifests=lambda c:(rows,[])
            with patch.object(pcycle,'read_training',lambda c:rows),patch.object(training.modules,'extract',features):
                for arm in ('base','local_transport'):
                    template=copy.deepcopy(next(j for j in config['jobs'] if j['name']==ds and j['module']==arm))
                    pos=count=0;stage_counts=[]
                    for size in scope.SIZES[ds]:
                        labels=scope.ORDERS[ds][pos:pos+size];pos+=size
                        fit,_=training.fit_split([r for r in rows if r['label'] in labels],74002);nsteps=2*((len(fit)+63)//64);count+=nsteps;stage_counts.append(nsteps)
                    frozen_steps=scope.STEPS[ds];scope.STEPS[ds]=stage_counts
                    outs=[];forward_counts=[]
                    # Exact Core/LT regression, exact PMT regression, then guarded PMT.
                    for number,(module,transport,enabled) in enumerate([(original,False,False),(training,False,False),(original,True,False),(training,True,False),(training,True,True)]):
                        c=copy.deepcopy(template);c.update(output=str(Path(tmp)/f'{ds}_{arm}_{number}'),expected_steps=count,max_steps=count,original_deadline=time.time()+1800,max_wall_seconds=1800,
                            paired_moment_transport=transport,snapshot_guard_mode='on' if enabled else 'off')
                        CountEncoder.calls=CountEncoder.images=0
                        if transport:
                            with patch.object(module,'manifests',side_effect=AssertionError('Early val manifest')):module.run(c)
                        else:module.run(c)
                        outs.append(c['output']);forward_counts.append((CountEncoder.calls,CountEncoder.images));steps+=count
                        assert json.loads(Path(c['output'],'STATUS.json').read_text())['status']=='TRAINED'
                    assert forward_counts[0]==forward_counts[1] and forward_counts[2]==forward_counts[3]==forward_counts[4]
                    for task in range(1,len(scope.SIZES[ds])+1):
                        states=[torch.load(Path(out,f'stage_{task}.pt'),weights_only=False) for out in outs]
                        for a,b in ((states[0],states[1]),(states[2],states[3])):
                            for key in ('model','adapter','head','bank','module_memory','boundary_pair_weights','selected_coefficients','steps','seen'):exact(a[key],b[key])
                        st=states[4];head,audit=guard.verify_checkpoint(st,c,task,st['steps']);maximum=max(maximum,audit['stationarity'])
                        z=torch.randn(3,6,dtype=torch.float64)
                        scores,sa=guard.scores(z,st,c,task,st['steps']);assert torch.equal(scores,z@head)
                        for mutate in ('steps','force','frame','float_head','source'):
                            bad=copy.deepcopy(st)
                            if mutate=='steps':bad['snapshot_guard_state']['steps']+=1
                            elif mutate=='force':bad['snapshot_guard_state']['force'][0,0]+=.01
                            elif mutate=='frame':bad['bank']['_frame']='stale'
                            elif mutate=='float_head':bad['head'][0,0]+=.01
                            else:bad['guard_metadata']['source_commit']='stale'
                            rejected(lambda:guard.verify_checkpoint(bad,c,task,st['steps']))
                        training_effect|=any(not torch.equal(st['model'][k],states[3]['model'][k]) for k in st['model'])
                        stage_checks+=1
                    budget=json.loads(Path(outs[-1],'GUARD_BUDGET.json').read_text())
                    assert len(budget['events'])==3*len(scope.SIZES[ds]) and budget['invalidations']==2*len(scope.SIZES[ds])
                    assert budget['extra_image_extractions']==budget['optimizer_updates_added']==0
                    events+=len(budget['events']);invalidations+=budget['invalidations']
                    records.append(dict(dataset=ds,module=arm,canonical_and_train_forward_counts=forward_counts[-1],guard_events=len(budget['events']),saved_guard_payload_bytes=budget['saved_guard_payload_bytes']))
                    eval_config=copy.deepcopy(c)
                    def numpy_extract(model,data,budget):
                        zs=[];ys=[]
                        with torch.no_grad():
                            for images,labels in data:zs.append(model(images).double());ys.append(labels)
                        return torch.cat(zs).numpy(),torch.cat(ys).numpy()
                    def history(root):
                        model=CountEncoder();result=[]
                        for t in range(1,len(scope.SIZES[ds])+1):
                            st=torch.load(Path(root,f'stage_{t}.pt'),weights_only=False);model.load_state_dict(st['model']);model.eval()
                            data=loader(SyntheticImages([v for v in rows if v['label'] in st['seen']],None,None,0),batch_size=64)
                            xx,yy=numpy_extract(model,data,None);pred=(xx@st['head'].numpy()).argmax(1)
                            result.append(dict(per_class_recall={str(cl):float((pred[yy==cl]==i).mean()) for i,cl in enumerate(st['seen'])}))
                        return dict(stages=result)
                    paths=[]
                    for number in (0,2):
                        path=Path(tmp)/f'history_{ds}_{arm}_{number}.json';path.write_text(json.dumps(history(outs[number])));paths.append(str(path))
                    eval_config.update(control_checkpoint_root=outs[0],pmt_control_checkpoint_root=outs[2],control_metrics=paths[0],pmt_control_metrics=paths[1],pmt_control_source_commit=c['source_commit'],operation='evaluate')
                    with patch.object(worker,'check_barrier',lambda c:None),patch.object(worker,'validate_job',lambda c:None),patch.object(worker,'ApartFeatures',CountEncoder),patch.object(worker,'Images',SyntheticImages),patch.object(worker,'DataLoader',loader),patch.object(worker,'extract',numpy_extract),patch.object(worker,'manifests',lambda c:(rows,rows)):
                        worker.evaluate(eval_config)
                    m=json.loads(Path(outs[-1],'metrics.json').read_text());assert m['test_accessed'] is False and len(m['stages'])==len(scope.SIZES[ds])
                    assert all('guard_vs_pmt' in v and 'same_state_OFF' in v and 'same_state_unconstrained' in v for v in m['stages']);evals+=1
                    scope.STEPS[ds]=frozen_steps
                    c=copy.deepcopy(template);c.update(output=str(Path(tmp)/f'{ds}_{arm}_pre'),preflight=True,expected_steps=0,max_steps=count,workers=0,original_deadline=time.time()+600,max_wall_seconds=600)
                    with patch.object(training,'manifests',side_effect=AssertionError('Preflight val manifest')):training.run(c)
                    f=json.loads(Path(c['output'],'PREFLIGHT.json').read_text())
                    assert f['parameters_unchanged'] and f['optimizer_updates']==0 and f['snapshot_guard']['epoch_event']['status']=='CERTIFIED' and f['gradient_norm']>0
                    preflights+=1
    # Barrier invoked through the actual evaluator before any manifest/model/image access.
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
        campaign=copy.deepcopy(config)
        for i,j in enumerate(campaign['jobs']):
            j['output']=str(Path(tmp)/f'job_{i}');p=Path(j['output']);p.mkdir()
            (p/'STATUS.json').write_text(json.dumps(dict(status='TRAINED',steps=j['expected_steps'],stages=len(j['task_sizes']),policy_updates=0)))
            (p/'GUARD_TRAINING_SEAL.json').write_text(json.dumps([scope.metadata(j,t,sum(scope.STEPS[j['name']][:t])) for t in range(1,len(j['task_sizes'])+1)]))
            (p/'TRAINING_SEAL.json').write_text(json.dumps([pcycle.checkpoint_metadata(j,t,sum(scope.STEPS[j['name']][:t])) for t in range(1,len(j['task_sizes'])+1)]))
        cf=Path(tmp)/'campaign';cf.write_text(json.dumps(campaign));gate=Path(tmp)/'gate';gate.write_text(json.dumps(dict(phase='TRAIN')))
        candidate=dict(campaign['jobs'][0],evaluation_gate=str(gate))
        with patch.dict(os.environ,Q128_CONFIG=str(cf)),patch.object(worker,'manifests',side_effect=AssertionError('Val opened before barrier')):
            rejected(lambda:worker.evaluate(candidate))
            gate.write_text(json.dumps(dict(phase='EVALUATE',training_sealed_at=1)));scope.check_barrier(candidate)
            wrong=Path(campaign['jobs'][1]['output'])/'GUARD_TRAINING_SEAL.json';wrong.write_text('[]');rejected(lambda:worker.evaluate(candidate))
    return dict(status='PASS',disabled_Core_LT_exact=True,disabled_PMT_exact=True,full_training_stage_checkpoints=stage_checks,
        synthetic_optimizer_updates=steps,synthetic_zero_update_preflights=preflights,real_GPU_preflights=0,synthetic_complete_evaluations=evals,
        snapshot_events=events,invalidations=invalidations,extra_medical_extractions=0,full_chain_guard_training_changed=training_effect,
        max_checkpoint_stationarity=maximum,budget_checks=records,scope_and_sealing_rejections=True)


def large(c,config):
    start=time.monotonic();d=c['dimension'];k=c['classes'];old=c['old_classes'];gen=torch.Generator().manual_seed(241+k)
    counts=c['current_train_counts'];fit_counts=label_counts(c['fit_n'],counts)
    all_x=torch.cat([sample(n,d,old+i,gen) for i,n in enumerate(counts)])
    all_y=torch.cat([torch.full((n,),old+i,dtype=torch.long) for i,n in enumerate(counts)])
    fit=torch.cat([torch.arange(sum(counts[:i]),sum(counts[:i])+v) for i,v in enumerate(fit_counts)])
    x,y=all_x[fit],all_y[fit];ids=torch.arange(len(all_x))
    samples=[sample(n,d,i,gen) for i,n in enumerate(c['old_train_counts'])];off_old=bank(samples)
    scale=torch.ones(d,dtype=x.dtype);scale[0]=2.8;scale[1]=.65;on_old=bank([z*scale for z in samples])
    xx,yy=(x,y) if c['event']=='epoch_init' else (all_x,all_y)
    current=bank([xx[yy==i] for i in range(old,k)])
    for part in current['components']:part['label']+=old
    def joined(b):return dict(mu=torch.cat([b['mu'],current['mu']]),Q=torch.cat([b['Q'],current['Q']]),n=b['n']+current['n'],components=b['components']+current['components'])
    off,on=joined(off_old),joined(on_old);task=c['task'];version=sum(scope.STEPS[c['dataset']][:task]);frame=f'task_{task}_epoch_1' if c['event']=='epoch_init' else f'task_{task}_sealed'
    on=affine.framed(on,frame);local=F.normalize(torch.randn(k,4,d,generator=gen,dtype=x.dtype),dim=-1) if c['LT'] else None
    module='local_transport' if c['LT'] else 'base';order=scope.ORDERS[c['dataset']][:k]
    R=training.modules.metric(on,module,c['dataset'],order,local)[0]
    initial=training.modules.head(on,module,c['dataset'],order,local)[0];pairs=competition.competition(on,initial)
    head,ha=competition.bank_head(on,pairs,regularizer=R)
    cfg=copy.deepcopy(next(j for j in config['jobs'] if j['name']==c['dataset'] and j['module']==module))
    session=guard.Session(cfg);session.version=version;session.register(task,fit,ids)
    # Reject support/time/frame changes through the production API before solving.
    rejected(lambda:session.project(head,on,pairs,R,off,local,x[:-1],y[:-1],fit[:-1],ids,old,c['event'],task,1,version,version))
    rejected(lambda:session.project(head,on,pairs,R,off,local,x,y,fit,ids,old,c['event'],task,1,version+1,version))
    frozen_inputs=copy.deepcopy((off,on,local,x,y,fit,ids))
    result,cert=session.project(head,on,pairs,R,off,local,x,y,fit,ids,old,c['event'],task,1,version,session.capture(version));session.require(cert,version)
    exact(frozen_inputs,(off,on,local,x,y,fit,ids));del frozen_inputs
    a=copy.deepcopy(session.events[-1]);a['synthetic_rescued']=int((((x@head).argmax(1)!=y)&((x@result).argmax(1)==y)).sum())
    a['synthetic_damaged']=int((((x@head).argmax(1)==y)&((x@result).argmax(1)!=y)).sum())
    gram=x@x.T if len(x)<d else x.T@x;eig=torch.linalg.eigvalsh(gram);rank=int((eig>torch.finfo(x.dtype).eps*max(x.shape)*eig.max()).sum())
    if c['event']=='boundary_final':
        st=dict(snapshot_guard_state=cert,guard_metadata=scope.metadata(cfg,task,version),seen=order,bank=on,module_memory=local,boundary_pair_weights=pairs,
                pmt_frame=frame,pmt_head_double=result,head=result.float())
        buf=io.BytesIO();torch.save(st,buf);checkpoint_bytes=buf.tell();buf.seek(0);st=torch.load(buf,weights_only=True)
        zz,rr=guard.scores(x,st,cfg,task,version);assert torch.equal(zz,x@result)
        a.update(checkpoint_bytes=checkpoint_bytes,checkpoint_rebuild=rr)
    session.advance(version+1);rejected(lambda:session.require(cert,version+1))
    a.update(status='PASS',observed_support_rank=rank,case=c,all_constraints_n=len(x)*(k-1),
        frozen_math_certificate_status=cert['certificate']['status'],input_statistics_unchanged=True,
        pipeline_seconds=time.monotonic()-start)
    return a


def vit(config):
    from run_multilabel import ApartFeatures
    c=config['jobs'][0];model=ApartFeatures(c['legacy_repo'],c['weight'],len(c['order']),'cpu',74002).eval()
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    gen=torch.Generator().manual_seed(239);images=torch.randn(2,3,224,224,generator=gen);y=torch.tensor([0,1])
    with torch.no_grad():z=model(images).double()
    weights=torch.ones(2,dtype=z.dtype);b=native.append(native.empty(model.dim,'cpu'),z,y,y,weights)
    b=affine.framed(b,'task_1_epoch_1');R=training.modules.metric(b,'base','ISIC',c['order'][:2],None)[0]
    pairs=competition.competition(b,training.modules.head(b,'base','ISIC',c['order'][:2],None)[0]);h,ha=competition.bank_head(b,pairs,regularizer=R)
    session=guard.Session(c);ids=torch.arange(2);session.register(1,ids,ids)
    wh,cert=session.project(h,b,pairs,R,b,None,z,y,ids,ids,0,'epoch_init',1,1,0,0)
    target=F.one_hot(y,2).double();value=model(images).double();loss=(value@wh-target).square().mean();loss.backward()
    groups={prefix:sum(float(p.grad.norm())**2 for n,p in model.named_parameters() if prefix in n and p.grad is not None)**.5 for prefix in ('.pool.pool.','.pool_few.pool.')}
    assert all(v>0 and __import__('math').isfinite(v) for v in groups.values())
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',actual_ViT_layers=12,feature_dimension=model.dim,synthetic_images=2,adapter_pool_gradients=groups,
        real_zero_update_CPU_backward=True,optimizer_updates=0,parameters_buffers_unchanged=True,public_cached_pretraining_only=True,
        first_task_only=True,highest_old_bank_covered=False,guard_certificate=cert['certificate']['status'])


def run(case):
    started=time.monotonic();assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2)
    config=json.loads(Path(os.environ['Q239_CPU_CONFIG']).read_text())
    result=chain(config) if case['kind']=='chain' else vit(config) if case['kind']=='vit' else large(case,config)
    assert not torch.cuda.is_initialized()
    result.update(CPU_pipeline_seconds=time.monotonic()-started,GPU_processes=0,medical_images_read=0,medical_checkpoints_read=0,val_reads=0,test_accessed=False,formal_optimizer_updates=0,GPU_ready=False)
    return result
