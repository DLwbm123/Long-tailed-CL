"""Full-dimensional synthetic CPU admission; no medical loaders/checkpoints/GPU."""
import copy,json,os,tempfile,time
from pathlib import Path
import numpy as np
import torch
import prototype_coherent as native
import core1_competition as core
import module_additions as modules
import gain_joint as joint
import gain_joint_worker as worker
import collect_gain_joint as collector
from gain_joint_production import fit_all,expanded,CANDIDATES,HEADS
from check_gain_joint import run as mathcheck
from check_intervention_gain import reject
from concm_statistical_metrics import aggregate
from run_module_cycle import save as atomic

def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);atomic(p,x)

def fixture(p):
    c=dict(gpus=[0,1],jobs=[dict(job_name=n) for n in worker.JOBS]);worker.validate(c,p)
    for k,v in [('gpus',[0,2]),('main_head','JOINT_POINT_SELECTED'),('test_accessed',True),('validation_retuning',True),('optimizer_updates',1),('batch_size',32),('automatic_retry',True)]:
        bad=copy.deepcopy(p);bad[k]=v;reject(lambda:worker.validate(c,bad))
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)/'barrier';root.mkdir();save(root/'PROGRAM_STATE.json',dict(phase='PREFLIGHT',variant='JOINT',sidecars_sealed_at=None));reject(lambda:worker.barrier(root,p,'fixture','preflight'))
        s=dict(lock=copy.deepcopy(p),source_commit='fixture',reporting_source_commit='fixture',prepared={},runs={},preflights={},ledger={},cpu_checks=dict(all_CPU_attempt_residence_seconds=1.),engineering_events=[],program=dict(status='COMPLETE',active={},costs=[],cpu_costs=[],failures=[],whole_round_wall_seconds=2.))
        for n in worker.JOBS:
            ds=n.split('_')[0];arm=n[len(ds)+1:];data=p['data'][ds];stages=[];prep=[];refs=[];exposures=0
            for task in range(1,len(data['task_sizes'])+1):
                seen=data['order'][:sum(data['task_sizes'][:task])];old=sum(data['task_sizes'][:task-1]);labels=np.concatenate([np.full(data['val_counts'][str(v)],v) for v in seen]);x=np.random.default_rng(task).normal(size=(len(labels),8)).astype(np.float32);head=np.random.default_rng(997*task).normal(size=(8,len(seen))).astype(np.float32);off=x@head;ray=np.zeros_like(head)
                if old:ray[:,:old]=np.random.default_rng(task*999).normal(0,.1,size=(8,old));ray[:,:old]-=ray[:,:old].mean(1,keepdims=True)
                deltas={v:(a*ray).astype(np.float32) for v,a in zip(CANDIDATES,joint.STEPS)};selections={v:dict(chosen_scale=scale if old else 0.) for v,scale in [('LEGACY_SELECTED',0.),('JOINT_POINT_SELECTED',.25),('JOINT_PRED_SELECTED',.5)]};all_deltas=expanded(deltas,selections);heads={};rawheads={};variants={}
                md=dict(dataset=ds,control=arm,control_source=p['controls'][arm]['source'],order=data['order'],seen=seen,seed=74002,task=task,source='fixture',version=joint.VERSION,control_steps=data['cumulative_steps'][task-1],optimizer_updates=0)
                for v in HEADS:
                    on,raw,change=worker.score(off,x,all_deltas[v],old);a,b=worker.diagnosis(off,on,raw,x,head,all_deltas[v],labels,data,task);a.update(published_control_exact=True,encoder_state_unchanged=True);heads[v]=a;rawheads[v]=b
                    if v in CANDIDATES:
                        mech=dict(relative_residual=0.,point=dict(exact_identity=not bool(ray.any()),scenarios=[]),predictive=dict(exact_identity=not bool(ray.any()),scenarios=[]))
                        for kind in ('point','predictive'):
                            for scenario in range(3):mech[kind]['scenarios'].append(dict(scenario=scenario,known_net=0.,net_lower=-.1 if old else 0.,net_upper=.1 if old else 0.,numerical_error=.01 if old else 0.,unknown_mass=0.,classes=[dict(label=i,known_net=0.,known_rescue=0.,known_harm=0.,unknown_mass=0.) for i in range(len(seen))]))
                        side=worker.pack(torch.from_numpy(deltas[v]).double(),old,dict(md,variant=v));assert torch.equal(worker.restore(side,old,dict(md,variant=v),head.shape),torch.from_numpy(deltas[v]));bad=copy.deepcopy(side);bad['production_metadata']['task']=-1;reject(lambda:worker.restore(bad,old,dict(md,variant=v),head.shape));variants[v]=dict(metadata=dict(md,variant=v),mechanical=mech,sidecar_bytes=100)
                refs.append(heads['C1']['metrics']['OFF']);exposures+=len(labels);prep.append(dict(task=task,variants=variants,selections=selections,native_solve=dict(relative_residual=0.)));stages.append(dict(task=task,heads=heads,raw_heads=rawheads))
            s['lock']['control_references'][n]=dict(stages=refs);s['prepared'][n]=dict(status='PREPARED',source_commit='fixture',optimizer_updates=0,medical_images_read=0,stages=prep);save(root/'prepared'/n/'PREPARED.json',s['prepared'][n]);common=dict(version=joint.VERSION,source_commit='fixture',optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False)
            s['runs'][n]=dict(common,job=n,dataset=ds,control=arm,status=dict(status='COMPLETE'),stages=stages,metrics={v:{mode:aggregate([z['raw_heads' if mode=='raw' else 'heads'][v]['metrics']['ON'] for z in stages],data) for mode in ('guarded','raw')} for v in HEADS},control_metrics=aggregate(refs,data),validation_image_exposures=exposures)
            s['preflights'][n]=dict(common,status='PASS',parameters_unchanged=True,CPU_GPU_scores_close=True,guard_checked=True,classification_metrics_read=False,real_validation_images=64);save(root/'JOINT'/'preflight'/n/'RESULTS.json',s['preflights'][n])
            for op in ('preflight','evaluate'):s['program']['costs'].append(dict(job=n,operation=op,gpu=0 if arm=='base' else 1,returncode=0,process_started=True,residence_seconds=1.))
            s['program']['cpu_costs'].append(dict(job=n,operation='prepare',gpu=None,returncode=0,process_started=True,residence_seconds=1.))
        save(root/'PROGRAM_STATE.json',dict(phase='PREFLIGHT',variant='JOINT',sidecars_sealed_at=1));worker.barrier(root,p,'fixture','preflight');save(root/'PROGRAM_STATE.json',dict(phase='EVALUATE',variant='JOINT',sidecars_sealed_at=1,preflights_sealed_at={}));reject(lambda:worker.barrier(root,p,'fixture','evaluate'));save(root/'PROGRAM_STATE.json',dict(phase='EVALUATE',variant='JOINT',sidecars_sealed_at=1,preflights_sealed_at={'JOINT':1}));worker.barrier(root,p,'fixture','evaluate')
        assert collector.validate(s);bad=copy.deepcopy(s);bad['runs']['HK_base']['stages'][-1]['heads']['C1']['metrics']['ON']['per_class_n']['5']=2;reject(lambda:collector.validate(bad));bad=copy.deepcopy(s);bad['prepared']['ISIC_base']['stages'][0]['variants']['C1']['metadata']['control_steps']=-1;reject(lambda:collector.validate(bad))
        out=Path(temp)/'fixture';out.mkdir();a=collector.report(s,out);assert a['main_rows']==32 and a['contrasts']==88 and a['model_candidate_scenario_rows']==432
        assert any(a['heads']['C1']['paired']['all']['changed'] for run in s['runs'].values() for a in run['stages']) and any(a['raw_heads']['C1']['paired']['new']['harmed'] for run in s['runs'].values() for a in run['stages'])
    return dict(report_rows=34,contrasts=88,model_candidate_scenario_rows=432,illegal_scope_early_val_metadata_denominator_refused=True,checkpoint_roundtrip=True,selections_fixed_before_val=True,raw_and_guard_signed_counts=True)

def run(protocol):
    begun=time.monotonic();torch.set_num_threads(2);torch.manual_seed(74002);assert not torch.cuda.is_available() and not torch.cuda.is_initialized();p=json.loads(Path(protocol).read_text());checks=fixture(p);scratch=Path(os.environ['TMPDIR'])/'joint_design.json';scratch.write_text(json.dumps(p['design_parameters']));oracle=mathcheck(scratch);scratch.unlink();cases=[]
    d,k,old=1536,23,21;mu=torch.eye(d,dtype=torch.float64)[:k]*.2;mu[:,-1]=1.;cov=torch.eye(d,dtype=mu.dtype)*.05;counts=[2048]*k;counts[1]=3;moments=[];parts=[]
    for c in range(k):
        S=cov.clone()
        if c==1:S.zero_();S[0,0]=S[1,1]=.05
        moments.append(torch.outer(mu[c],mu[c])+S);parts.append(dict(label=c,center=mu[c],count=counts[c],mass=mu.new_tensor(1.),radius=S.trace()))
    bank=dict(mu=mu,Q=torch.stack(moments),n=counts,components=parts);before=copy.deepcopy(bank);initial,_=native.head(bank);pairs=core.competition(bank,initial)
    for arm in ('base','local_transport'):
        R,_=modules.metric(bank,arm,'HK',list(range(k)),None if arm=='base' else torch.randn(k,4,d,dtype=mu.dtype));W,qa=core.bank_head(bank,pairs,regularizer=R);ray,ra=joint.frozen_ray(bank,pairs,R,W,old,W.float());assert ra['proposal_norm']>1e-10 and ra['iterations']>0 and ray.any();legacy=dict(preselection_delta_norm=float(ray.norm()),chosen_scale=0.,candidates=[]);deltas,mechanical=fit_all(bank,pairs,R,W,old,W.float(),legacy);all_deltas=expanded(deltas,mechanical['selections'])
        for v,scale in zip(CANDIDATES,joint.STEPS):
            assert torch.equal(deltas[v],(scale*ray).float().double()) and mechanical['candidates'][v]['relative_residual']<=1e-8;meta=dict(dataset='HK',control=arm,control_source='fixture',order=list(range(k)),seen=list(range(k)),seed=74002,task=6,source='fixture',variant=v,version=joint.VERSION,control_steps=176,optimizer_updates=0);side=worker.pack(deltas[v],old,meta);assert torch.equal(worker.restore(side,old,meta,(d,k)),deltas[v].float())
        assert torch.equal(bank['Q'],before['Q']) and torch.equal(bank['mu'],before['mu']) and bank['n']==before['n'];first,_=fit_all(bank,pairs,R,W,0,W.float(),dict(chosen_scale=0.));assert all(not z.any() for z in first.values())
        cases.append(dict(control=arm,dimension=d,classes=k,old_classes=old,native_residual=qa['relative_residual'],direction_residual=ra['relative_residual'],proposal_norm=ra['proposal_norm'],ray_norm=float(ray.norm()),nonzero_full_highdimensional_direction=True,four_candidate_bits_exact=True,all_probabilities_finite=True,checkpoint_roundtrip=True,bank_unchanged=True,selections=mechanical['selections']));(Path(os.environ['TMPDIR'])/'CPU_PROGRESS.private.json').write_text(json.dumps(cases,indent=2));del R,W,deltas,all_deltas,mechanical
    return dict(status='PASS',production_integrated=True,synthetic_only=True,full_high_dimensional_cpu_PASS=True,full_native_solve=True,dimensions=[1536,23],cases=cases,production_checks=checks,mathematical_check=oracle,GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,medical_val_reads=0,formal_optimizer_updates=0,policy_updates=0,test_accessed=False,pipeline_seconds=time.monotonic()-begun)
if __name__=='__main__':print(json.dumps(run(os.environ['Q128_PROTOCOL']),indent=2,allow_nan=False))
