"""Sealed first-transition classification pilot on existing anonymous images."""
import collections
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from check_reserved_competition_memory import DT,initial_model,frame,clone_model,scores
from check_fullslot_retention_memory import train,compress
from check_memory_stability import decision
from query_node_weights import rollout,query_loss,project_direction,select,unit


def save(path,value):
    temp=path.with_suffix('.part');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def load(path):return torch.load(path,map_location='cpu',weights_only=True)


def split_rows(rows,seed):
    result={k:[] for k in ('S','Qfit','Qgate')}
    for c in sorted({r['label'] for r in rows}):
        rr=[r for r in rows if r['label']==c]
        groups=sorted({r['identity_component'] for r in rr})
        if len(groups)<3:raise ValueError('Insufficient independent known components for S/Qfit/Qgate')
        random.Random(seed+c).shuffle(groups)
        n=max(1,len(groups)//10)
        names={g:'Qfit' if i<n else 'Qgate' if i<2*n else 'S' for i,g in enumerate(groups)}
        for r in rr:result[names[r['identity_component']]].append(r)
    keys=[{r['identity_component'] for r in rr} for rr in result.values()]
    assert not any(a&b for i,a in enumerate(keys) for b in keys[i+1:])
    assert sum(map(len,result.values()))==len(rows)
    return result


def split(cfg,lock,root):
    result=[]
    for case in lock['datasets']:
        name=case['name'];spec=cfg['datasets'][name];seen=case['old']+case['new'];order=case['order'][:seen]
        parts={}
        for part in ('train','val'):
            with (Path(spec['manifest'])/(part+'.csv')).open() as f:raw=list(csv.DictReader(f))
            rows=[]
            for r in raw:
                if int(r['original_label']) not in order:continue
                p=Path(r['relative_path']);assert not p.is_absolute() and '..' not in p.parts
                rows.append(dict(relative_path=r['relative_path'],identity_component=r['identity_component'],
                    label=order.index(int(r['original_label']))))
            assert rows and all(r['identity_component'] for r in rows)
            assert len({r['relative_path'] for r in rows})==len(rows)
            actual=dict(collections.Counter(str(order[r['label']]) for r in rows))
            expected={str(c):case['expected_'+part+'_counts'][str(c)] for c in order}
            assert actual==expected,'Frozen source class counts differ'
            groups=collections.defaultdict(set)
            for r in rows:groups[r['identity_component']].add(r['label'])
            assert all(len(v)==1 for v in groups.values()),'Known component crosses class labels'
            parts[part]=rows
        assert not {r['identity_component'] for r in parts['train']}&{r['identity_component'] for r in parts['val']}
        assert not {r['relative_path'] for r in parts['train']}&{r['relative_path'] for r in parts['val']}
        roles=split_rows(parts['train'],lock['seed']);roles['E']=parts['val']
        roles['S_old']=[r for r in roles['S'] if r['label']<case['old']]
        roles['S_new']=[r for r in roles.pop('S') if r['label']>=case['old']]
        class_rows=[]
        for role,rows in roles.items():
            save(root/(name+'.'+role+'.rows.private.json'),rows)
            for c in sorted({r['label'] for r in rows}):
                cr=[r for r in rows if r['label']==c]
                class_rows.append(dict(role=role,class_index=c,original_label=order[c],
                    images=len(cr),components=len({r['identity_component'] for r in cr})))
        assert all(any(r['role']==role and r['class_index']==c for r in class_rows)
            for role in ('Qfit','Qgate','E') for c in range(seen))
        result.append(dict(dataset=name,class_rows=class_rows,known_component_overlap=0,
            patient_isolation='UNKNOWN',evaluation_exposure='historically reused development validation'))
    save(root/'SPLIT_LOCK.json',dict(locked_at=time.time(),datasets=result))
    return dict(datasets=result,optimizer_updates=0)


def extract(cfg,lock,root,phase,journal):
    import subprocess
    from torch.utils.data import DataLoader
    from run_multilabel import ApartFeatures
    from run_prototype_single import Images
    from run_medical_v2 import transform,seed_all
    if phase=='gate':assert (root/'CANDIDATES_LOCK.json').is_file()
    if phase=='E':assert (root/'SELECTION_LOCK.json').is_file()
    roles={'fit':('S_old','S_new','Qfit'),'gate':('Qgate',),'E':('E',)}[phase]
    seed_all(lock['seed'])
    encoder=ApartFeatures(os.environ['Q356_SOURCE'],cfg['weight'],23,'cuda',lock['seed']).eval().requires_grad_(False)
    assert not any(p.requires_grad for p in encoder.parameters())
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    own=[v.split(',',1)[1].strip() for v in apps.splitlines() if v.split(',',1)[0].strip()==str(os.getpid())]
    assert own and all(not any(s in n.lower() for s in ('wangbomin','longtailed','query_node')) for n in own)
    save(root/('GPU_PROCESS_'+phase+'.private.json'),dict(pid=os.getpid(),names=own,checked_at=time.time()))
    result=[]
    for case in lock['datasets']:
        name=case['name']
        for role in roles:
            rows=json.loads((root/(name+'.'+role+'.rows.private.json')).read_text())
            ds=Images(rows,cfg['datasets'][name]['images'],transform(False),lock['seed'])
            loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
            chunks=[];labels=[];start=time.monotonic()
            with torch.no_grad():
                for i,(x,y) in enumerate(loader):
                    z=encoder(x.cuda()).cpu().to(DT)
                    assert z.shape[1]==1536 and bool(torch.isfinite(z).all())
                    chunks.append(F.normalize(z,dim=1));labels.append(y)
                    if i%20==0:journal(dict(phase='extract_'+phase,dataset=name,role=role,images=sum(len(a) for a in labels)))
            path=root/(name+'.'+role+'.features.private.pt')
            torch.save(dict(x=torch.cat(chunks),y=torch.cat(labels),components=[r['identity_component'] for r in rows]),path)
            result.append(dict(dataset=name,role=role,images=len(rows),feature_bytes=path.stat().st_size,seconds=time.monotonic()-start))
    return dict(extractions=result,encoder_updates=0,pretrained_encoder_frozen=True,
        seeded_adapter_pools_untrained=True,feature_dimension=1536,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated())


def teacher(case,lock,root,journal):
    name=case['name'];state=load(root/(name+'.S_old.features.private.pt'))
    current={c:state['x'][state['y']==c] for c in range(case['old'])}
    q=frame(1536,len(case['order']),lock['seed']);model=initial_model(1536,lock['seed'])
    journal(dict(phase='teacher_48_steps',dataset=name))
    model,cost=train(model,current,{},q,list(range(case['old'])))
    journal(dict(phase='moment_48_steps',dataset=name))
    memory,mem_cost=compress(current,lock['seed'],'moment')
    torch.save(dict(teacher=model,frame=q,memory=memory),root/(name+'.teacher.private.pt'))
    return dict(dataset=name,teacher_training=cost,memory_construction=mem_cost,
        adapter_updates=48,memory_updates=48,old_memory_nodes=case['old']*16,
        teacher_seen_Qfit=False,teacher_seen_Qgate=False,teacher_seen_E=False)


def candidate(case,lock,root,journal):
    name=case['name'];old=case['old'];seen=old+case['new']
    state=load(root/(name+'.teacher.private.pt'));data=load(root/(name+'.S_new.features.private.pt'))
    fit=load(root/(name+'.Qfit.features.private.pt'))
    current={c:data['x'][data['y']==c] for c in range(old,seen)}
    a=torch.zeros(old,16,dtype=DT,requires_grad=True)
    journal(dict(phase='differentiable_baseline_48_steps',dataset=name))
    model=rollout(state['teacher'],current,state['memory'],state['frame'],a,graph=True)
    baseline=clone_model(model)
    losses=[query_loss(model,state['frame'],fit['x'],fit['y'],cs,seen) for cs in (range(old),range(old,seen))]
    go=torch.autograd.grad(losses[0],a,retain_graph=True)[0]
    gn=torch.autograd.grad(losses[1],a)[0]
    d,info=project_direction(go,gn,old,seen);d=torch.from_numpy(d.reshape(old,16))
    values=[float(v.detach()) for v in losses]
    del losses,model,go,gn
    journal(dict(phase='candidate_single_direction_48_steps',dataset=name))
    if bool(d.any()):
        can=clone_model(rollout(state['teacher'],current,state['memory'],state['frame'],d));updates=48
    else:can=clone_model(baseline);updates=0
    ratios=[]
    for c,(_,p) in state['memory'].items():
        w=(p.log()+d[c]).softmax(0);assert abs(float(w.sum())-1)<1e-12
        ratios.extend((w/p).tolist())
    torch.save(dict(model=baseline,frame=state['frame']),root/(name+'.baseline.private.pt'))
    torch.save(dict(model=can,frame=state['frame']),root/(name+'.candidate.private.pt'))
    torch.save(dict(logit_change=d),root/(name+'.direction.private.pt'))
    return dict(dataset=name,sealed_at=time.time(),baseline_updates=48,candidate_updates=updates,
        hypergradient_backward_sweeps=2,direction=info,baseline_Qfit_old_CE=values[0],baseline_Qfit_new_CE=values[1],
        within_class_weight_ratio_min=min(ratios),within_class_weight_ratio_max=max(ratios),
        node_positions_unchanged=True,CE_class_mass_unchanged=True,retention_uses_original_weights=True,
        old_S_access_in_candidate_stage=False,extra_Qfit_images=len(fit['y']))


def metrics(obj,data,old,seen):
    with torch.no_grad():
        ss=torch.cat([scores(data['x'][i:i+256],obj['model'],obj['frame'])[:,:seen] for i in range(0,len(data['y']),256)])
        if 'bias' in obj:ss=ss+obj['bias']
        pred=ss.argmax(1);y=data['y'];ce=-ss.log_softmax(1)[torch.arange(len(y)),y]
    rows=[]
    for c in range(seen):
        mask=y==c;pr=pred[mask];n=int(mask.sum());assert n>0
        rows.append(dict(class_index=c,group='old' if c<old else 'new',count=n,correct=int((pr==c).sum()),
            recall=float((pr==c).to(DT).mean()),CE=float(ce[mask].mean()),errors_to_old=int(((pr!=c)&(pr<old)).sum()),
            errors_to_new=int(((pr!=c)&(pr>=old)).sum())))
    ba={g:sum(v['recall'] for v in rows if g=='all' or v['group']==g)/sum(1 for v in rows if g=='all' or v['group']==g) for g in ('all','old','new')}
    return pred,rows,ba


def choose(lock,root):
    assert (root/'CANDIDATES_LOCK.json').is_file()
    rows=[];choices=[]
    for case in lock['datasets']:
        name=case['name'];data=load(root/(name+'.Qgate.features.private.pt'));summary={}
        for mode in ('baseline','candidate'):
            _,rr,ba=metrics(load(root/(name+'.'+mode+'.private.pt')),data,case['old'],case['old']+case['new'])
            rows.extend(dict(dataset=name,model=mode,**r) for r in rr);summary[mode]=ba
        choice=select(summary['baseline'],summary['candidate'])
        choices.append(dict(dataset=name,**choice))
    sealed=dict(locked_at=time.time(),choices=choices)
    save(root/'SELECTION_LOCK.json',sealed)
    return dict(class_rows=rows,**sealed)


def cluster_intervals(pairs,old,seed):
    rng=np.random.default_rng(seed);draws=10000;matrix=[];points=[]
    for c in sorted({r['class_index'] for r in pairs}):
        rr=[r for r in pairs if r['class_index']==c]
        count=np.array([r['count'] for r in rr]);delta=np.array([r['rescued']-r['harmed'] for r in rr])
        weights=rng.multinomial(len(rr),np.ones(len(rr))/len(rr),size=draws)
        matrix.append((weights@delta)/(weights@count));points.append(float(delta.sum()/count.sum()))
    a=np.stack(matrix,1);out=[]
    for name,idx in [('all',list(range(len(matrix)))),('old',list(range(old))),('new',list(range(old,len(matrix))))]:
        lo,hi=np.quantile(a[:,idx].mean(1),[.025,.975],method='linear')
        out.append(dict(group=name,BA_difference=float(np.mean(np.array(points)[idx])),lower=float(lo),upper=float(hi),
            confidence=.95,bootstrap_draws=draws,bootstrap_seed=seed,unit='known_identity_component_within_class',
            scope='pointwise conditional development interval; not patients or training-seed generalization'))
    return out


def evaluate(lock,root):
    selection=json.loads((root/'SELECTION_LOCK.json').read_text());rows=[];summaries=[];intervals=[];decisions=[];pair_rows=[]
    for i,case in enumerate(lock['datasets']):
        name=case['name'];old=case['old'];seen=old+case['new'];data=load(root/(name+'.E.features.private.pt'))
        choice=next(v for v in selection['choices'] if v['dataset']==name)
        base=load(root/(name+'.baseline.private.pt'))
        a,ar,ab=metrics(base,data,old,seen)
        if choice['selected']=='baseline':b,br,bb=a,ar,ab
        else:b,br,bb=metrics(load(root/(name+'.candidate.private.pt')),data,old,seen)
        rows.extend(dict(dataset=name,model=mode,**r) for mode,rr in [('baseline',ar),('selected',br)] for r in rr)
        pairs=[];y=data['y'];groups=collections.defaultdict(list)
        for j,(c,key) in enumerate(zip(y.tolist(),data['components'])):groups[c,key].append(j)
        for (c,key),idx in groups.items():
            ac=a[idx]==c;bc=b[idx]==c
            pairs.append(dict(class_index=c,count=len(idx),rescued=int((~ac&bc).sum()),harmed=int((ac&~bc).sum()),
                both_correct=int((ac&bc).sum()),both_wrong=int((~ac&~bc).sum())))
        for c in range(seen):
            rr=[r for r in pairs if r['class_index']==c]
            pair_rows.append(dict(dataset=name,class_index=c,components=len(rr),
                **{k:sum(r[k] for r in rr) for k in ('count','rescued','harmed','both_correct','both_wrong')}))
        ci=cluster_intervals(pairs,old,lock['seed']+100+i)
        intervals.extend(dict(dataset=name,**r) for r in ci)
        decisions.append(dict(dataset=name,selected=choice['selected'],**decision(ci)))
        summaries.extend(dict(dataset=name,group=g,baseline_BA=ab[g],selected_BA=bb[g],delta_BA=bb[g]-ab[g]) for g in ab)
    return dict(class_rows=rows,paired=pair_rows,summaries=summaries,intervals=intervals,decisions=decisions,
        evaluation='reused development val, sealed within this round',selection_locked_at=selection['locked_at'],
        evaluated_models=['baseline','selected'],patient_isolation='UNKNOWN')


def guard(role,root,cfg,lock):
    selected={v['dataset']:v['selected'] for v in json.loads((root/'SELECTION_LOCK.json').read_text())['choices']} if role=='evaluate' else {}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute();s=p.name
        if role!='split':
            for spec in cfg['datasets'].values():
                assert not p.is_relative_to(Path(spec['manifest'])), 'Raw manifest unavailable to model roles'
                if p.is_relative_to(Path(spec['images'])):assert role.startswith('extract_'),'Images only available during authorized extraction'
        if not p.is_relative_to(root):return
        if '.E.' in s:assert role in ('split','extract_E','evaluate'),'E role boundary'
        if '.Qgate.' in s:assert role in ('split','extract_gate','choose'),'Qgate role boundary'
        if '.Qfit.' in s:assert role in ('split','extract_fit','candidate'),'Qfit role boundary'
        if '.S_old.' in s:assert role in ('split','extract_fit','teacher'),'Old support unavailable after memory construction'
        if '.S_new.' in s:assert role in ('split','extract_fit','candidate'),'Current support role boundary'
        if role=='evaluate' and s.endswith('.candidate.private.pt'):
            assert selected[s.split('.')[0]]=='candidate','Rejected candidate unavailable to evaluator'
    sys.addaudithook(audit)


def main():
    root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE']);job=json.loads(os.environ['Q356_JOB'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/'docs/real_query_node1_20261016/PROTOCOL_LOCK.json').read_text())
    role=job['role'];torch.set_num_threads(2);torch.set_default_dtype(torch.float32 if role.startswith('extract_') else DT)
    if not role.startswith('extract_'):assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    guard(role,root,cfg,lock)
    journal=lambda v:save(root/(job['id']+'.progress.json'),v)
    begun=time.monotonic();journal(dict(phase=role))
    if role=='unit':
        result=unit()
        dummy=[dict(label=c,identity_component=str(c)+'_'+str(i)) for c in range(2) for i in range(3)]
        assert all(len(rr)==2 for rr in split_rows(dummy,74002).values())
        zero=[dict(class_index=c,count=1,rescued=0,harmed=0) for c in range(3)]
        assert all(v['lower']==v['upper']==0 for v in cluster_intervals(zero,2,74002))
        result.update(split_and_cluster_checked=True)
        try:
            (root/'unit.E.rows.private.json').read_text()
        except AssertionError:result['unit_E_access_denied']=True
        else:raise AssertionError('Unit role unexpectedly accessed E')
    elif role=='split':result=split(cfg,lock,root)
    elif role.startswith('extract_'):result=extract(cfg,lock,root,role[8:],journal)
    elif role in ('teacher','candidate'):
        case=next(c for c in lock['datasets'] if c['name']==job['dataset'])
        result=(teacher if role=='teacher' else candidate)(case,lock,root,journal)
    elif role=='choose':result=choose(lock,root)
    elif role=='evaluate':result=evaluate(lock,root)
    else:raise ValueError('Unknown frozen role')
    result.update(status='COMPLETED',operation_seconds=time.monotonic()-begun,role=role,
        source_commit=os.environ['Q356_SHA'],test_accessed=False,role_access_guard=True)
    save(root/(job['id']+'.json'),result)


if __name__=='__main__':main()
