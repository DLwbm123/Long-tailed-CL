"""V3 fixed-layout train/validation diagnostics; never opens test images."""
import copy
import csv
import gc
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from run_medical_v2 import (MedicalLearner, Images, args_for, seed_all, sha,
                            write_json, network_hash, rng_equal)
from evaluate_medical_v2 import metrics, write_csv

GROUPS = {'head_rank2':[1,0], 'mid_rank4':[5,4,3,2], 'tail_rank2':[7,6]}
SEEDS = (1993,1994,1995)

@contextmanager
def isolated_rng(learner):
    rng=learner._capture_rng_state(); synth=learner.synth_rng.clone()
    loader=learner.loader_generator.get_state(); modes={m:m.training for m in learner._network.modules()}
    try: yield
    finally:
        learner._restore_rng_state(rng);learner.synth_rng=synth;learner.loader_generator.set_state(loader)
        for m,mode in modes.items():m.training=mode
        assert rng_equal(rng,learner._capture_rng_state())

def dataset(config,order,seen,split):
    assert split in ('train','val'), 'V3_TEST_ACCESS_FORBIDDEN'
    d=Images(Path(config['protocol'])/(split+'.csv'),config['images'],order,range(seen),False)
    d.rows.sort(key=lambda r:r['sample_id'])
    return d

def loader(d):
    return DataLoader(d,batch_size=48,shuffle=False,num_workers=8,
                      generator=torch.Generator().manual_seed(0))

def quantiles(x):
    x=np.asarray(x)
    return dict(zip(('min','q05','q25','median','q75','q95','max'),map(float,np.quantile(x,[0,.05,.25,.5,.75,.95,1])))) if x.size else None

def scored(raw,targets,rows,order,known):
    m,pc=metrics(raw,targets,rows,order,known,GROUPS)
    for k in list(m):
        if 'test' in k:m[k.replace('test','eval')]=m.pop(k)
    seen=raw.shape[1]
    for scope,cs in [('current',list(range(known,seen))),('old',list(range(known)))]:
        mask=np.isin(targets,cs)
        if cs and mask.any():
            pred=np.array(cs)[raw[mask][:,cs].argmax(1)];y=targets[mask]
            m['restricted_'+scope+'_ba']=float(np.mean([(pred[y==c]==c).mean() for c in cs])*100)
        else:m['restricted_'+scope+'_ba']=None
    m['zero_recall_classes']=[p['original_label'] for p in pc if p['recall']==0]
    if known:
        current=targets>=known
        m['current_margin']=quantiles(raw[current,known:].max(1)-raw[current,:known].max(1))
    else:m['current_margin']=None
    return m,pc

@torch.no_grad()
def evaluate(learner,split):
    d=dataset(learner.config,learner.order,learner._total_classes,split)
    result={k:[] for k in ('main','few','sum')};features={k:[] for k in ('main','few')};ys=[]
    start=time.monotonic()
    with isolated_rng(learner):
        learner._network.eval()
        for _,x,y in loader(d):
            o=learner._network(x.cuda(),train=False)
            main=o['logits'][:,:learner._total_classes];few=o['logits_few'][:,:learner._total_classes]
            for name,v in [('main',main),('few',few),('sum',main+few)]:
                assert torch.isfinite(v).all(),'BLOCKED_NONFINITE_INFERENCE'
                result[name].append(v.cpu().numpy())
            for name,key in [('main','pre_logits'),('few','pre_logits_few')]:features[name].append(o[key].norm(dim=1).cpu().numpy())
            ys.append(y.numpy())
    y=np.concatenate(ys);ms=[];pcs=[];norms={}
    for name,parts in result.items():
        m,pc=scored(np.concatenate(parts),y,d.rows,learner.order,learner._known_classes)
        ms.append(dict(head=name,split=split,**m));pcs += [dict(head=name,split=split,**r) for r in pc]
    for name,parts in features.items():
        f=np.concatenate(parts)
        norms[name]={str(learner.order[c]):quantiles(f[y==c]) for c in range(learner._total_classes)}
    return dict(metrics=ms,per_class=pcs,feature_norm=norms,images=len(d),seconds=time.monotonic()-start)

def head_stats(learner):
    b=learner._network.backbone;k=learner._known_classes;t=learner._total_classes
    out={}
    for name,h in [('main',b.head),('few',b.head_few)]:
        out[name]={'bias':h.bias[:t].detach().cpu().tolist(),'weight_norm':h.weight[:t].detach().norm(dim=1).cpu().tolist()}
        for scope,sl in [('old',slice(0,k)),('current',slice(k,t))]:
            out[name][scope+'_bias_mean']=float(h.bias[sl].detach().mean()) if h.bias[sl].numel() else None
            out[name][scope+'_weight_norm_mean']=float(h.weight[sl].detach().norm(dim=1).mean()) if h.weight[sl].numel() else None
    return out

def real_terms(output,y,known,total,scope):
    start=known if scope=='current' else 0;y=y-start
    a=output['logits'][:,start:total];b=output['logits_few'][:,start:total]
    return {'main_ce':F.cross_entropy(a,y), 'sum_ce':F.cross_entropy(a+b,y),
            'few_pool_ce':-(output['pool_id'].squeeze(1)*(F.one_hot(y,total-start)*torch.log(b.softmax(-1)+1e-7)).sum(1)).sum()}

def fixed_probe(learner):
    d=dataset(learner.config,learner.order,learner._total_classes,'train')
    d.rows=[r for r in d.rows if r['target']>=learner._known_classes]
    # Fixed image-ID layout, without class buckets. Every epoch uses the same records.
    d.rows=d.rows[:48]
    return next(iter(loader(d)))[1:]

def gradient_probe(learner,epoch=9):
    """Actual image/head paths, no parameter update; synthesis and all RNG restored."""
    with isolated_rng(learner):
        learner._network.eval();x,y=fixed_probe(learner);x=x.cuda();y=y.cuda()
        k=learner._known_classes;t=learner._total_classes;b=learner._network.backbone
        o=learner._network(x,train=True,weight=torch.tensor(learner.args['lt_list'],device='cuda')[y])
        params=[b.head.weight,b.head.bias,b.head_few.weight,b.head_few.bias]
        losssets={scope:real_terms(o,y,k,t,scope) for scope in ('current','all_seen')}
        terms={scope+'/'+name:v for scope,ts in losssets.items() for name,v in ts.items()}
        terms.update({scope+'/real_total':sum(ts.values())/3 for scope,ts in losssets.items()})
        counts=torch.tensor(learner.args['lt_list'],device='cuda')[y];pool=o['pool_id'].squeeze(1)
        terms['assignment/pool_match']=((torch.where(counts<=learner.theta,1.,.1)-pool)**2*torch.where(counts<=learner.theta,10.,1.)).sum() if epoch<5 else ((1-pool)**2*pool*10).sum()
        terms['constraint/pull']=-learner.args['pull_constraint_coeff']*(o['reduce_sim']+o['reduce_sim_few'])
        sampled=learner._concm_stage1_sample_memory()
        synth=None
        if sampled:
            fm,ff,sy=sampled;sl=(b.head(fm)+b.head_few(ff))[:,:t]
            terms['synthetic/raw']=F.cross_entropy(sl,sy)
            terms['synthetic/weighted']=.05*terms['synthetic/raw']
            synth={'main_norm':quantiles(fm.detach().norm(dim=1).cpu()),'few_norm':quantiles(ff.detach().norm(dim=1).cpu()),
                   'logits':quantiles(sl.detach().cpu()),'current_minus_old_margin':quantiles((sl[:,k:].max(1).values-sl[:,:k].max(1).values).detach().cpu()),
                   'labels_counts':torch.bincount(sy,minlength=t).cpu().tolist()}
        for scope in ('current','all_seen'):
            terms[scope+'/total_with_replay']=terms[scope+'/real_total']+terms.get('synthetic/weighted',0)
            terms[scope+'/full_training_total']=terms[scope+'/total_with_replay']+terms['assignment/pool_match']+terms['constraint/pull']
        records={};vectors={}
        for name,term in terms.items():
            grads=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True) if term.requires_grad else [None]*len(params)
            grads=[torch.zeros_like(p) if g is None else g for p,g in zip(params,grads)]
            assert all(torch.isfinite(g).all() for g in grads)
            rec={'loss':float(term.detach()),'weight_in_total':1/3 if name.split('/')[-1] in ('main_ce','sum_ce','few_pool_ce') else (.05 if name=='synthetic/raw' else 1)}
            for scope,slc in [('old',slice(0,k)),('current',slice(k,t))]:
                v=torch.cat([g[slc].reshape(-1) for g in grads]);vectors[name,scope]=v
                rec[scope+'_gradient_norm']=float(v.norm())
                rec[scope+'_weighted_gradient_norm']=rec[scope+'_gradient_norm']*rec['weight_in_total']
            rec['current_main_bias_shift_derivative']=float(grads[1][k:t].sum())
            rec['current_few_bias_shift_derivative']=float(grads[3][k:t].sum())
            records[name]=rec
        shifts={}
        for name,raw,target,restricted in [('current_real',(o['logits']+o['logits_few'])[:,:t],y,True),
                                         ('all_seen_real',(o['logits']+o['logits_few'])[:,:t],y,False)]+([] if sampled is None else [('old_synthetic',sl,sy,False)]):
            z=raw.detach();offset=torch.zeros((),device='cuda',requires_grad=True)
            mask=torch.arange(t,device='cuda')>=k;shifted=z+offset*mask
            loss=F.cross_entropy(shifted[:,k:] if restricted else shifted,target-k if restricted else target)
            g=float(torch.autograd.grad(loss,offset)[0]);mass=float(z.softmax(1)[:,k:].sum(1).mean())
            expected=0. if restricted else mass-(1. if name=='all_seen_real' else 0.)
            assert abs(g-expected)<2e-6, 'BLOCKED_GRADIENT_PROBE_FORMULA'
            shifts[name]={'derivative':g,'expected':expected,'current_probability_mass':mass}
        cosines={}
        if sampled:
            for scope in ('current','all_seen'):
                for rows in ('old','current'):
                    a=vectors[scope+'/real_total',rows];c=vectors['synthetic/weighted',rows]
                    cosines[scope+'/'+rows]=float(F.cosine_similarity(a,c,dim=0))
        return {'terms':records,'common_logit_shift':shifts,'real_replay_cosine':cosines,'head':head_stats(learner),
                'real_main_norm':quantiles(o['pre_logits'].detach().norm(dim=1).cpu()),
                'real_few_norm':quantiles(o['pre_logits_few'].detach().norm(dim=1).cpu()),
                'real_logits':quantiles((o['logits']+o['logits_few'])[:,:t].detach().cpu()),
                'real_margin':quantiles(((o['logits']+o['logits_few'])[:,k:t].max(1).values-(o['logits']+o['logits_few'])[:,:k].max(1).values).detach().cpu()) if k else None,
                'synthetic':synth,'rng_restored':True}

@torch.no_grad()
def batch_probe(learner):
    with isolated_rng(learner):
        learner._network.eval()
        train=dataset(learner.config,learner.order,learner._total_classes,'train')
        val=dataset(learner.config,learner.order,learner._total_classes,'val')
        fixed=val[0][1];outputs=[]
        for peers in ([train[i][1] for i in range(47)],[val[i][1] for i in range(1,48)]):
            x=torch.stack([fixed]+peers).cuda()
            for reverse in (False,True):
                xx=x.flip(0) if reverse else x;index=47 if reverse else 0
                o=learner._network(xx);out={k:o[k][index].detach().cpu() for k in ('logits','logits_few','prompt_idx','prompt_idx_few')}
                outputs.append(out)
        # Frequency input must not affect inference logits/routing.
        a=learner._network(x,weight=torch.ones(48,device='cuda'))
        c=learner._network(x,weight=torch.full((48,),10529,device='cuda'))
        for key in ('logits','logits_few','prompt_idx','prompt_idx_few'):
            assert torch.equal(a[key],c[key]),'BLOCKED_LABEL_DEPENDENCE'
        return [{'context':i,'max_main_change':float((o['logits']-outputs[0]['logits']).abs().max()),
                 'max_few_change':float((o['logits_few']-outputs[0]['logits_few']).abs().max()),
                 'main_adapter':o['prompt_idx'].tolist(),'few_adapter':o['prompt_idx_few'].tolist()}
                for i,o in enumerate(outputs)]

@torch.no_grad()
def native_parity(learner):
    import timm
    from utils.medical_v2 import verified_state
    with isolated_rng(learner):
        learner._network.eval();x,_=fixed_probe(learner);x=x[:2].cuda()
        native=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0).cuda().eval()
        native.load_state_dict(verified_state(learner.config['weight']),strict=True)
        original=learner._network.original_backbone(x)['pre_logits'];ref=native(x)
        result={'original_max_abs':float((original-ref).abs().max()),'original_relative_l2':float((original-ref).norm()/ref.norm())}
        # Only a disposable deep copy is zeroed. Native and adapter block outputs locate deviations.
        adapter=copy.deepcopy(learner._network.backbone).eval()
        for name,p in adapter.named_parameters():
            if name.endswith(('up_proj.weight','up_proj.bias')):p.zero_()
        nv={};av={};hooks=[]
        for i in range(12):
            hooks.append(native.blocks[i].register_forward_hook(lambda m,a,o,i=i:nv.__setitem__(i,o.detach())))
            hooks.append(adapter.blocks[i].register_forward_hook(lambda m,a,o,i=i:av.__setitem__(i,o.detach())))
        ref=native(x);zero=adapter(x,cls_features=original)['pre_logits']
        result['zero_adapter_max_abs']=float((zero-ref).abs().max())
        result['zero_adapter_relative_l2']=float((zero-ref).norm()/ref.norm())
        result['block_max_abs']={str(i):float((nv[i]-av[i]).abs().max()) for i in range(12)}
        for h in hooks:h.remove()
        result['pass']=result['original_max_abs']<1e-5 and result['zero_adapter_relative_l2']<1e-4
        result['operations']='Verified locked QKV split, fc1/fc2 mapping; final norm CLS; zero up projections; native fused attention may have FP32 rounding differences.'
        del adapter,native
        return result

def load_parent(config,entry,output):
    v2=json.loads(Path(config['v2_runtime']).read_text())
    a,order=args_for(v2,entry['seed'],entry['branch'],False);seed_all(entry['seed'])
    l=MedicalLearner(a,v2,order,output)
    assert sha(entry['path'])==entry['sha256'],'BLOCKED_PARENT_HASH'
    state=l.restore(entry['path'])  # Ordinary V2 validation remains intact.
    assert state['phase']=='session_complete' and state['epoch']==10
    assert (state['task'],state['known'],state['total'])==(entry['session'],[0,4,6][entry['session']],[4,6,8][entry['session']])
    return l,state

def p0(config):
    out=Path(config['output'])/'p0';out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((Path(config['v2_output'])/'ALL_CHECKPOINTS_LOCK.json').read_text())
    assert len(lock['checkpoints'])==18
    allrows=[];pcrows=[];grads={};batches={};hashes={};durations=[];streams={}
    for entry in lock['checkpoints']:
        tag=f"{entry['seed']}_{entry['branch']}_s{entry['session']}";dest=out/(tag+'.json')
        if dest.exists():
            r=json.loads(dest.read_text())
        else:
            l,state=load_parent(config,entry,out/'scratch');before=network_hash(l._network)
            r={'parent':entry,'network_hash':before,'diagnostics':{}}
            if entry['session']==0:
                h=hashlib.sha256(repr(state['rng']['python']).encode())
                for v in state['rng']['numpy']:h.update(np.asarray(v).tobytes())
                for v in [state['rng']['torch'],*state['rng'].get('cuda',[]),state['loader_rng'],state['synth_rng']]:h.update(v.numpy().tobytes())
                r['s0_rng_stream_sha256']=h.hexdigest()
            if entry['session']==0 and entry['branch']=='B' and entry['seed']==1993:
                r['native']=native_parity(l);write_json(out/'native_backbone_parity.json',r['native'])
                assert r['native']['pass'],'BLOCKED_NATIVE_PARITY'
            if entry['session']>0:r['gradient']=gradient_probe(l)
            r['batch_context']=batch_probe(l)
            for split in ('train','val'):r['diagnostics'][split]=evaluate(l,split)
            r['head']=head_stats(l)
            assert before==network_hash(l._network),'BLOCKED_DIAGNOSTIC_MUTATION'
            write_json(dest,r);del l,state;gc.collect();torch.cuda.empty_cache()
        if entry['session']==0:
            hashes[f"{entry['seed']}_{entry['branch']}"]=r['network_hash']
            streams[f"{entry['seed']}_{entry['branch']}"]=r['s0_rng_stream_sha256']
        grads[tag]=r.get('gradient');batches[tag]=r['batch_context']
        for split,diag in r['diagnostics'].items():
            prefix=dict(seed=entry['seed'],branch=entry['branch'],session=entry['session'])
            allrows += [dict(**prefix,**m) for m in diag['metrics']]
            pcrows += [dict(**prefix,**p) for p in diag['per_class']]
            durations.append(dict(**prefix,split=split,images=diag['images'],seconds=diag['seconds']))
        write_json(out/'PROGRESS.json',{'last_completed':tag,'completed':len(batches),'test_predictions':0})
    for seed in SEEDS:
        assert hashes[f'{seed}_B']==hashes[f'{seed}_C'],'BLOCKED_S0_PAIR_MISMATCH'
        assert streams[f'{seed}_B']==streams[f'{seed}_C'],'BLOCKED_S0_RNG_PAIR_MISMATCH'
    write_json(out/'gradient_probe.json',grads);write_json(out/'batch_context_probe.json',batches)
    write_json(out/'s0_network_hashes.json',hashes);write_json(out/'resource.json',durations)
    write_csv(out/'val_session_metrics.csv',[r for r in allrows if r['split']=='val'])
    write_csv(out/'val_per_class_metrics.csv',[r for r in pcrows if r['split']=='val'])
    forensic=out/'offline_forensics';forensic.mkdir(exist_ok=True)
    write_csv(forensic/'train_session_metrics.csv',[r for r in allrows if r['split']=='train'])
    write_csv(forensic/'train_per_class_metrics.csv',[r for r in pcrows if r['split']=='train'])
    (out/'P0_OBJECTIVE_DIAGNOSIS.md').write_text('# P0 objective diagnosis\n\n18 locked V2 checkpoints evaluated without updates on image-ID-sorted train/val, batch 48. Train access is offline_forensics only; no statistics feed P2.\n\nGradient probes use actual real-image main/few paths and original synthetic heads. The derivative identities, per-component raw/weighted gradients, totals, row norms and cosines are in gradient_probe.json. These local derivatives do not alone establish the cause of end-to-end collapse.\n\nBatch peer and order dependence is reported in batch_context_probe.json, without changing routing. Native parity uses final normalized CLS and zeroed adapter up projections on a disposable copy.\n\nMemory is extracted after each session using the current training dataset, including its random crop/flip; a separate seed and restored RNG isolate extraction. Old mean/variance entries are retained rather than updated. Main/few Gaussian features are independently sampled. Norm/logit/margin distributions are included in probes and per-checkpoint reports.\n\nNew test predictions: 0. V2 COMPLETE_MIXED_SIGNAL remains unchanged.\n')
    write_json(out/'COMPLETE.json',{'checkpoints':18,'test_predictions':0,'s0_pair_parity':'PASS'})
