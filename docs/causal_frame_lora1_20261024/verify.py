"""Independent NumPy complete 8+4 chain, all 24 scores and actual draw/permission audit."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from foundation import write,build_backbone
from data import read_rows
from inference import body_state,numpy_features,numpy_scores,numpy_objective
from reference import np_tokens
from reporting import report


def independently_selected(rows,cap,seed):
    out=[]
    for c in sorted({r['label'] for r in rows}):
        groups={}
        for r in rows:
            if r['label']==c:groups.setdefault(r['group'],[]).append(r)
        key=lambda g,path='':hashlib.sha256(f'{seed}:{c}:{g}:{path}'.encode()).digest()
        for g in sorted(groups,key=lambda g:(key(g),g))[:cap]:out.append(min(groups[g],key=lambda r:(key(g,r['relative_path']),r['id']))['id'])
    return out


def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);p=Path(cfg['output']);load=lambda name:torch.load(p/name,map_location='cpu',weights_only=False);tick=time.time();gates=[]
    def check(name,value,passed):
        gates.append(dict(check=name,value=value,PASS=bool(passed)));write(p/'VERIFICATION_PROGRESS.json',gates)
        assert passed,name
        assert time.time()<cfg['hard_deadline_epoch']-60
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    for i,r in enumerate(train):r['id']=i
    check('coverage_and_patient_isolation',dict(train=len(train),val=len(val)),len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val})
    seal=json.loads((p/'FIT_SEAL.json').read_text());ss=json.loads((p/'STAGE_SEALS.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text())
    check('24_models_before_query',dict(seal=seal,query_open=opened),len(ss)==6 and seal['models']==24 and max(s['at'] for s in ss)<seal['at']<opened['at'] and seal['backbone_frozen_exact'])
    b=build_backbone(cfg,'cpu');lower=body_state(b,0,8);upper=body_state(b,8,12);query=load('query.private.pt');labels=np.array([r['label'] for r in val]);check('query_shape_labels',dict(shape=list(query['tokens'].shape)),query['tokens'].shape==(300,197,768) and np.array_equal(labels,query['labels'].numpy()))
    native=[];lower_rows=0;ratio_max=0.;proof_error=0.;native_proof_error=0.
    for i in range(0,300,cfg['microbatch']):
        x=query['embedding'][i:i+cfg['microbatch']].numpy();raw=np_tokens(x,lower,apply_norm=False);actual=query['tokens'][i:i+len(raw)].numpy().astype(np.float64)
        ratio=float((abs(raw-actual)/(abs(raw)*2**-11+cfg['score_absolute_tolerance'])).max());ratio_max=max(ratio_max,ratio);lower_rows+=len(raw)
        check('full_lower_query_componentwise_rounding',dict(rows=lower_rows,ratio=ratio),ratio<=1)
        native.append(numpy_features(actual,upper))
    native=np.concatenate(native);expected_memory=[];native_sum=np.zeros((1537,1537));stages=[];score_error=0.;probe_error=0.;mismatch=0;score_rows=0;draw_rows=0;previous_frame=None;normal_error=0.;memory_map={}
    for task in range(6):
        seen=cfg['order'][:10+2*task];new=seen[:10] if task==0 else seen[-2:];pack=load(f'stage_{task}.private.pt');check('arrival_order',dict(task=task,seen=pack['seen']),pack['seen']==seen)
        rows=[r for r in train if r['label'] in new];source=[train[i] for i in expected_memory]+rows;pools={}
        for r in source:pools.setdefault(r['label'],{}).setdefault(r['group'],[]).append(r['id'])
        rng=np.random.default_rng(cfg['seed']+1000+task);draw=[];steps=cfg['base_updates'] if task==0 else cfg['later_updates']
        for _ in range(steps):
            ids=[]
            for c in rng.choice(seen,8,replace=False):
                groups=list(pools[c])
                for _ in range(8):
                    images=pools[c][groups[int(rng.integers(len(groups)))]];ids.append(images[int(rng.integers(len(images)))])
            draw.append(ids)
        equal=np.array_equal(np.array(draw,dtype=np.int32),pack['draws'].numpy());draw_rows+=64*steps
        check('actual_patient_draws',dict(task=task,rows=64*steps),equal and pack['probe']['labels'].tolist()==[seen.index(train[i]['label']) for i in draw[-1]])
        chosen=independently_selected(rows,cfg['memory_per_class'],cfg['seed']);expected_memory+=chosen;check('finite_memory_admission',dict(task=task,images=len(expected_memory)),expected_memory==pack['memory_ids'])
        for c in new:
            mm=load(f'memory_{c}.private.pt');wanted=[i for i in chosen if train[i]['label']==c];check('patient_memory_rows',dict(label=c,rows=len(wanted)),mm['ids']==wanted and len(mm['tokens'])==len(wanted) and mm['tokens'].dtype==torch.float16)
            memory_map.update({i:t for i,t in zip(mm['ids'],mm['tokens'])})
            proof=load(f'proof_{c}.private.pt');rr=[r for r in rows if r['label']==c];check('arrival_moments_counts',dict(label=c,images=proof['images'],patients=proof['patients'],online_error=proof['gram_error']),proof['images']==len(rr) and proof['patients']==len({r['group'] for r in rr}) and proof['gram_error']<1e-9)
            for v in proof['proof']:
                error=float(abs(np_tokens(v['embedding'][None].numpy(),lower,apply_norm=False)-v['lower'][None].numpy()).max());proof_error=max(proof_error,error);lower_rows+=1
                z=numpy_features(v['lower'][None].half().numpy(),upper);ne=float(abs(z-v['native_descriptor'][None].numpy()).max());native_proof_error=max(native_proof_error,ne)
                check('full_training_proof',dict(label=c,id=v['id'],lower8_error=error,native4_error=ne),v['id'] in {r['id'] for r in rr} and error<=cfg['score_absolute_tolerance'] and ne<=cfg['score_absolute_tolerance'])
        probe_ids=draw[-1];checked_memory=0
        for j,i in enumerate(probe_ids):
            if i in memory_map:
                check('immutable_memory_in_actual_probe',dict(task=task,row=j,id=i),torch.equal(memory_map[i],pack['probe']['tokens'][j]));checked_memory+=1
        # Independently reconstruct the arrival-only geometry, using NumPy rather than core.ArrivalFrame.
        means=pack['means'].numpy();mean_error=float(abs(means-pack['native_rhs'].numpy()[:-1,-len(new):].T*len(seen)*np.sqrt(2)).max());check('current_geometry_mean_matches_full_patient_statistics',dict(task=task,error=mean_error),mean_error<1e-10);means=means/np.linalg.norm(means,axis=1,keepdims=True);rho=-1/19
        if task==0:
            n=len(means);g=(1-rho)*np.eye(n)+rho*np.ones((n,n));gv,gu=np.linalg.eigh(g);mv,mu=np.linalg.eigh(means@means.T)
            check('frame_base_rank',dict(eigen_min=float(mv.min())),mv.min()>1e-12)
            frame=(gu*np.sqrt(gv))@gu.T@(mu*(mv**-.5))@mu.T@means
        else:
            frame=previous_frame.copy()
            for mean in means:
                g=frame@frame.T;parallel=np.linalg.solve(g,np.full(len(frame),rho))@frame;radius=1-parallel@parallel
                check('frame_extension_radius',dict(old_rows=len(frame),radius=float(radius)),radius>=-1e-10)
                if len(frame)==19:row=parallel
                else:
                    residual=mean-np.linalg.solve(g,frame@mean)@frame;norm=np.linalg.norm(residual);check('frame_extension_residual',dict(norm=float(norm)),norm>1e-10);row=parallel+np.sqrt(max(radius,0))*residual/norm
                frame=np.concatenate([frame,row[None]])
        actual_frame=pack['frame'].numpy();gram_error=float(abs(actual_frame@actual_frame.T-((1-rho)*np.eye(len(seen))+rho*np.ones((len(seen),len(seen))))).max());frame_error=float(abs(frame-actual_frame).max())
        old_delta=0. if previous_frame is None else float(abs(actual_frame[:len(previous_frame)]-previous_frame).max())
        # NumPy reconstruction may differ at floating roundoff; saved old rows must match the preceding Torch seal exactly.
        exact_old=task==0 or torch.equal(load(f'stage_{task-1}.private.pt')['frame'],pack['frame'][:len(previous_frame)])
        check('independent_causal_frame',dict(task=task,Gram_error=gram_error,frame_error=frame_error,old_delta_against_NumPy=old_delta),gram_error<1e-10 and frame_error<1e-9 and exact_old);previous_frame=frame
        native_sum+=pack['arrived_gram'].numpy();mat=native_sum/len(seen)+cfg['ridge']*np.eye(1537);rhs=pack['native_rhs'].numpy();h=np.linalg.solve(mat,rhs);normal_error=max(normal_error,float(abs(mat@pack['native_head'].numpy()-rhs).max()))
        check('native_full_normal_system',dict(task=task,residual=normal_error),abs(mat-pack['native_mat'].numpy()).max()<1e-10 and abs(mat[-1,-1]-(.5+cfg['ridge']))<1e-10 and abs(rhs[-1]-1/np.sqrt(2)/len(seen)).max()<1e-10 and normal_error<1e-9)
        mask=np.isin(labels,seen);qt=query['tokens'][mask].numpy();nz=native[mask];phi=np.concatenate([nz,np.ones((len(nz),1))],1)/np.sqrt(2);values={'NATIVE_RIDGE':phi@h}
        for a in cfg['arms'][1:]:
            adapter={k:v.numpy() for k,v in pack['adapters'][a].items()}
            if a!='LEARNED_CE':check('fixed_head_permission',dict(task=task,arm=a,max_delta=float(abs(adapter['head']-actual_frame).max())),np.array_equal(adapter['head'],actual_frame))
            values[a]=np.concatenate([numpy_scores(qt[i:i+cfg['microbatch']],upper,adapter) for i in range(0,len(qt),cfg['microbatch'])])
            probe=pack['probe'];ps={k:v.numpy() for k,v in probe['before'][a].items()};loss=numpy_objective(probe['tokens'].numpy(),probe['labels'].numpy(),upper,ps,a);err=abs(loss-probe['losses'][a]);probe_error=max(probe_error,err);check('actual_training_objective',dict(task=task,arm=a,error=err),err<=cfg['score_absolute_tolerance'])
        saved=load(f'scores_{task}.private.pt');metrics={}
        for a,v in values.items():
            er=float(abs(v-saved[a].numpy()).max());score_error=max(score_error,er);mi=int((v.argmax(1)!=saved[a].numpy().argmax(1)).sum());mismatch+=mi;check('full_sealed_scores',dict(task=task,arm=a,error=er,prediction_mismatches=mi),er<=cfg['score_absolute_tolerance'] and mi==0)
            pred=np.array(seen)[v.argmax(1)];lab=labels[mask];correct={str(c):int(((pred==c)&(lab==c)).sum()) for c in seen};metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/len(v));score_rows+=len(v)
        stages.append(dict(task=task,seen=seen,metrics=metrics))
    costs=json.loads((p/'COSTS.json').read_text());check('complete_counts',dict(full_lower_rows=lower_rows,score_rows=score_rows,draw_rows=draw_rows,cost=costs),lower_rows==340 and score_rows==5400 and draw_rows==352000 and costs['science_train_encodes']==68058 and costs['validation_encodes']==300 and costs['qualification_encodes']==20 and costs['qualification_full_chain_rows']==20 and costs['native_solves']==6)
    for a in cfg['arms'][1:]:check('optimizer_and_model_rows',dict(arm=a,science=costs['arms'][a],qualification=costs['qualification_arms'][a]),costs['arms'][a]==dict(updates=5500,forward_rows=352000,backward_rows=352000) and costs['qualification_arms'][a]==dict(updates=2,forward_rows=128,backward_rows=128))
    recorded=json.loads((p/'NUMERIC_GATES.json').read_text());frozen=[g for g in recorded if g.get('check')=='frozen_original_parameters'];whole=[g for g in recorded if g.get('check')=='entire_original_backbone_frozen']
    check('frozen_training_checks',dict(stage_checks=len(frozen),entire_checks=len(whole)),len(frozen)==18 and all(g['max_delta']==0 for g in frozen) and len(whole)==1 and whole[0]['max_delta']==0)
    result,decision=report(stages,cfg,train)
    for name,value in [('RESULTS.json',result),('DECISION.json',decision),('VERIFICATION.json',dict(state='PASS',full_native_lower_rows=lower_rows,score_groups=24,score_rows=score_rows,score_error=score_error,prediction_mismatches=mismatch,training_probe_error=probe_error,training_lower_proof_error=proof_error,native_upper_proof_error=native_proof_error,query_FP16_componentwise_bound_max_ratio=ratio_max,normal_equation_error=normal_error,draw_rows=draw_rows,wall_seconds=time.time()-tick,scope='Independent full8+4 native chain on saved FP16 embedding/lower coordinates and actual LoRA; qualification checks patch embedding on two saved reader arrays. No new raw-image encoding or full optimization trajectory replay. Current full moments independently checked online.'))]:write(p/name,value)
    write(p/'status.private.json',dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time()))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        import traceback
        write(Path(cfg['output'])/'VERIFICATION_FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
