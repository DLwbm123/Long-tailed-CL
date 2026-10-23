"""Independent NumPy full frozen response, readouts, actual training probes and permissions."""
import hashlib,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from core import ARMS,np_scores
from data import read_rows
from foundation import build_backbone,body_state,np_tokens,write
from reporting import report


def independently_selected(rows,cap,seed):
    selected=[]
    for c in sorted({r['label'] for r in rows}):
        groups={}
        for r in rows:
            if r['label']==c:groups.setdefault(r['group'],[]).append(r)
        def key(group,path=''):return hashlib.sha256(f'{seed}:{c}:{group}:{path}'.encode()).digest()
        for group in sorted(groups,key=lambda g:(key(g),g))[:cap]:selected.append(min(groups[group],key=lambda r:(key(group,r['relative_path']),r['id']))['id'])
    return selected


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def global_np(t):return unit(np.concatenate([t[:,0],t[:,1:].mean(1)],1))
def affine_np(t):
    z=global_np(t);return np.concatenate([z,np.ones((len(z),1))],1)/np.sqrt(2)

def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);start=time.time();p=Path(cfg['output']);load=lambda n:torch.load(p/n,map_location='cpu',weights_only=False)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    for i,r in enumerate(train):r['id']=i
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    seals=json.loads((p/'STAGE_SEALS.json').read_text());fit=json.loads((p/'FIT_SEAL.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text());assert len(seals)==6 and fit['models']==24 and max(x['at'] for x in seals)<fit['at']<opened['at'] and fit['backbone_frozen_exact']
    b=build_backbone(cfg,'cpu').double();state=body_state(b);query=load('query.private.pt');labels=np.array([r['label'] for r in val]);assert np.array_equal(labels,query['labels'].numpy()) and query['tokens'].shape==(300,197,768)
    full_rows=0;quant_bound_ratio=0.;proof_error=0.
    for i in range(0,300,cfg['microbatch']):
        assert time.time()<cfg['hard_deadline_epoch']-60
        raw=np_tokens(query['prefix'][i:i+cfg['microbatch']].numpy(),state);actual=query['tokens'][i:i+cfg['microbatch']].numpy().astype(np.float64)
        bound=np.abs(raw)*2**-11+cfg['FP32_probe_tolerance'];ratio=float((abs(actual-raw)/bound).max());assert ratio<=1;quant_bound_ratio=max(quant_bound_ratio,ratio);full_rows+=len(raw)
        write(p/'verify_status.private.json',dict(phase='full12_native_query',rows=full_rows,updated=time.time()))
    stages=[];expected_memory=[];native_sum=np.zeros((1537,1537));score_error=0.;normal_error=0.;mismatch=0;probe_error=0.;all_score_rows=0;all_draws=0
    for task in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        seen=cfg['order'][:10+2*task];new=seen[:10] if task==0 else seen[-2:];pack=load(f'stage_{task}.private.pt');assert pack['seen']==seen
        rows=[r for r in train if r['label'] in new];source=[train[i] for i in expected_memory]+rows;pools={}
        for r in source:pools.setdefault(r['label'],{}).setdefault(r['group'],[]).append(r['id'])
        rng=np.random.default_rng(cfg['seed']+1000+task);draws=[];steps=cfg['base_updates'] if task==0 else cfg['later_updates']
        for _ in range(steps):
            ix=[]
            for c in rng.choice(seen,8,replace=False):
                patients=list(pools[c])
                for _ in range(8):
                    images=pools[c][patients[int(rng.integers(len(patients)))]];ix.append(images[int(rng.integers(len(images)))])
            draws.append(ix)
        assert np.array_equal(np.array(draws,dtype=np.int32),pack['draws'].numpy());all_draws+=64*steps
        chosen=independently_selected(rows,cfg['memory_per_class'],cfg['seed']);expected_memory+=chosen;assert pack['memory_ids']==expected_memory
        for c in new:
            memory=load(f'memory_{c}.private.pt');expected=[i for i in chosen if train[i]['label']==c];assert memory['ids']==expected and memory['tokens'].dtype==torch.float16 and len(expected)==len(memory['tokens'])
            r=load(f'proof_{c}.private.pt');rr=[z for z in rows if z['label']==c];assert r['images']==len(rr) and r['patients']==len({z['group'] for z in rr}) and r['independent_current_gram_error']<1e-9
            for proof in r['proof']:
                assert proof['id'] in {z['id'] for z in rr}
                raw=np_tokens(proof['prefix'][None].numpy(),state);error=float(abs(raw-proof['raw'][None].numpy()).max());proof_error=max(proof_error,error);assert error<cfg['FP32_probe_tolerance'];full_rows+=1
        native_sum+=pack['arrived_gram'].numpy();mat=native_sum/len(seen)+cfg['ridge']*np.eye(1537);assert abs(mat-pack['native_mat'].numpy()).max()<1e-10
        assert np.max(abs(mat-mat.T))<1e-12 and abs(mat[-1,-1]-(.5+cfg['ridge']))<1e-10
        rhs=pack['native_rhs'].numpy();assert np.max(abs(rhs[-1]-1/np.sqrt(2)/len(seen)))<1e-10
        native_head=np.linalg.solve(mat,rhs);normal_error=max(normal_error,float(abs(mat@pack['native_head'].numpy()-rhs).max()));mask=np.isin(labels,seen);qt=query['tokens'][mask].numpy().astype(np.float64)
        values={'NATIVE_RIDGE':affine_np(qt)@native_head}
        for a in ARMS[1:]:
            ss={k:v.numpy().astype(np.float64) for k,v in pack['models'][a].items()};chunks=[]
            for i in range(0,len(qt),cfg['microbatch']):chunks.append(np_scores(qt[i:i+cfg['microbatch']],ss,a))
            values[a]=np.concatenate(chunks)
            probe=pack['probe'];ps={k:v.numpy().astype(np.float64) for k,v in probe['models_before_update'][a].items()};sc=np_scores(probe['tokens'].numpy().astype(np.float64),ps,a);sc-=sc.max(1,keepdims=True);yy=probe['labels'].numpy();loss=float(np.mean(np.log(np.exp(sc).sum(1))-sc[np.arange(64),yy]));probe_error=max(probe_error,abs(loss-probe['losses'][a]));assert abs(loss-probe['losses'][a])<=cfg['FP32_probe_tolerance']
        saved=load(f'scores_{task}.private.pt');mm={}
        for a,s in values.items():
            score_error=max(score_error,float(abs(s-saved[a].numpy()).max()));mismatch+=int((s.argmax(1)!=saved[a].numpy().argmax(1)).sum());pred=np.array(seen)[s.argmax(1)];correct={str(c):int(((pred==c)&(labels[mask]==c)).sum()) for c in seen};mm[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/int(mask.sum()));all_score_rows+=len(s)
        stages.append(dict(task=task,seen=seen,metrics=mm));write(p/'verify_status.private.json',dict(phase='independent_readouts',task=task,score_error=score_error,updated=time.time()))
    assert score_error<=cfg['score_absolute_tolerance'] and mismatch==0 and normal_error<1e-9 and full_rows==340 and all_score_rows==5400 and all_draws==352000
    cost=json.loads((p/'COSTS.json').read_text());assert cost['science_train_encodes']==68058 and cost['validation_encodes']==300 and cost['qualification_encodes']==20 and cost['interface_encodes']==2 and cost['native_solves']==6
    for a in ARMS[1:]:assert cost['arms'][a]==dict(updates=5500,forward_rows=352000,backward_rows=352000)
    result,decision=report(stages,cfg,train);verdict=dict(state='PASS',full_native_NumPy_rows=340,score_groups=24,score_rows=5400,score_error=score_error,prediction_mismatches=mismatch,FP32_proof_error=proof_error,FP16_componentwise_bound_max_ratio=quant_bound_ratio,normal_equation_error=normal_error,training_probe_error=probe_error,paired_draw_rows_independently_reconstructed=all_draws,memory_images=len(expected_memory),patient_overlap=0,new_image_encodes=0,new_optimization=0,wall_seconds=time.time()-start,scope='Independent full12 native response on saved quantized prefixes and actual readouts; FP16 token rounding bounded componentwise, not bitwise raw-image or full optimization trajectory replay. Current arriving full moments independently checked online; cumulative arrival matrices and all normal systems reconstructed here.')
    for n,v in [('RESULTS.json',result),('DECISION.json',decision),('VERIFICATION.json',verdict)]:write(p/n,v)
    write(p/'status.private.json',dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time()))

if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
