"""Independent full native chain, every sealed score group, draw and permission census."""
import hashlib,json,os,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from core import Upper,np_upper,np_scores
from data import read_rows
from foundation import write,build_backbone,body_state,np_tokens
from reporting import report


def lower_state(b):return {k:v for k,v in body_state(b).items() if not k.startswith(('blocks.10.','blocks.11.'))}
def validate_chain(b,template,proofs,cfg):
    low=lower_state(b);up={k:v.detach().cpu().numpy() for k,v in template.state_dict().items()};err=0.;upper_err=0.;n=0
    for v in proofs:
        ref=np_tokens(v['embedding'][None].numpy(),low,apply_norm=False);error=float(abs(ref-v['lower'][None].numpy()).max());err=max(err,error)
        lower=v['lower'][None].half().double();out=np_upper(lower.numpy(),up,cfg['layernorm_epsilon'])
        with torch.no_grad():actual=template.cpu().double()(lower).numpy()
        upper_err=max(upper_err,float(abs(out-actual).max()));n+=1
    r=dict(lower10_FP32_NumPy_error=err,upper2_FP64_NumPy_error=upper_err,rows=n)
    assert err<=cfg['FP32_probe_tolerance'] and upper_err<=cfg['score_absolute_tolerance']
    template.float();return r


def independently_selected(rows,cap,seed):
    out=[]
    for c in sorted({r['label'] for r in rows}):
        gs={}
        for r in rows:
            if r['label']==c:gs.setdefault(r['group'],[]).append(r)
        key=lambda g,path='':hashlib.sha256(f'{seed}:{c}:{g}:{path}'.encode()).digest()
        for g in sorted(gs,key=lambda g:(key(g),g))[:cap]:out.append(min(gs[g],key=lambda r:(key(g,r['relative_path']),r['id']))['id'])
    return out

def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);start=time.time();p=Path(cfg['output']);load=lambda n:torch.load(p/n,map_location='cpu',weights_only=False)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    for i,r in enumerate(train):r['id']=i
    assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val}
    seals=json.loads((p/'STAGE_SEALS.json').read_text());fit=json.loads((p/'FIT_SEAL.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text())
    assert len(seals)==6 and fit['models']==18 and max(s['at'] for s in seals)<fit['at']<opened['at'] and fit['backbone_frozen_exact']
    b=build_backbone(cfg,'cpu').double();template=Upper(b);low=lower_state(b);native={k:v.detach().numpy() for k,v in template.state_dict().items()};q=load('query.private.pt');labels=np.array([r['label'] for r in val]);assert np.array_equal(labels,q['labels'].numpy())
    count=0;quant_ratio=0.
    for i in range(0,300,cfg['microbatch']):
        assert time.time()<cfg['hard_deadline_epoch']-60
        ref=np_tokens(q['embedding'][i:i+cfg['microbatch']].numpy(),low,apply_norm=False);actual=q['tokens'][i:i+len(ref)].numpy().astype(np.float64)
        ratio=float((abs(ref-actual)/(abs(ref)*2**-11+cfg['FP32_probe_tolerance'])).max());quant_ratio=max(quant_ratio,ratio);assert ratio<=1;count+=len(ref)
        write(p/'verify_status.private.json',dict(phase='full_native_lower_query',rows=count,updated=time.time()))
    expected_memory=[];native_sum=np.zeros((1537,1537));stages=[];score_error=0.;probe_error=0.;mismatch=0;score_rows=0;draw_rows=0;previous_heads=None;all_experts={a:[] for a in cfg['arms'][1:]}
    for task in range(6):
        seen=cfg['order'][:10+2*task];new=seen[:10] if task==0 else seen[-2:];pack=load(f'stage_{task}.private.pt');assert seen==pack['seen'];rr=[r for r in train if r['label'] in new];source=[train[i] for i in expected_memory]+rr;pools={}
        for r in source:pools.setdefault(r['label'],{}).setdefault(r['group'],[]).append(r['id'])
        rng=np.random.default_rng(cfg['seed']+1000+task);draw=[];steps=cfg['base_updates'] if task==0 else cfg['later_updates']
        for _ in range(steps):
            ids=[]
            for c in rng.choice(seen,8,replace=False):
                gs=list(pools[c])
                for _ in range(8):
                    images=pools[c][gs[int(rng.integers(len(gs)))]];ids.append(images[int(rng.integers(len(images)))])
            draw.append(ids)
        assert np.array_equal(np.array(draw,dtype=np.int32),pack['draws'].numpy());draw_rows+=64*steps
        assert pack['probe']['labels'].tolist()==[seen.index(train[i]['label']) for i in draw[-1]]
        chosen=independently_selected(rr,cfg['memory_per_class'],cfg['seed']);expected_memory+=chosen;assert expected_memory==pack['memory_ids']
        for c in new:
            mm=load(f'memory_{c}.private.pt');assert mm['ids']==[i for i in chosen if train[i]['label']==c] and len(mm['tokens'])==len(mm['ids']) and mm['tokens'].dtype==torch.float16
            proof=load(f'proof_{c}.private.pt');cr=[r for r in rr if r['label']==c];assert proof['images']==len(cr) and proof['patients']==len({r['group'] for r in cr}) and proof['gram_error']<1e-9
            for v in proof['proof']:assert v['id'] in {r['id'] for r in cr}
            validate_chain(b,template,proof['proof'],cfg);count+=len(proof['proof'])
        native_sum+=pack['arrived_gram'].numpy();mat=native_sum/len(seen)+cfg['ridge']*np.eye(1537);rhs=pack['native_rhs'].numpy()
        assert abs(mat-pack['native_mat'].numpy()).max()<1e-10 and abs(mat[-1,-1]-(.5+cfg['ridge']))<1e-10
        assert abs(rhs[-1]-1/np.sqrt(2)/len(seen)).max()<1e-10
        h=np.linalg.solve(mat,rhs);assert abs(mat@pack['native_head'].numpy()-rhs).max()<1e-9
        qt=q['tokens'][np.isin(labels,seen)].numpy().astype(np.float64);z=np.concatenate([np_upper(qt[i:i+cfg['microbatch']],native,cfg['layernorm_epsilon']) for i in range(0,len(qt),cfg['microbatch'])]);phi=np.concatenate([z,np.ones((len(z),1))],1)/np.sqrt(2);values={'NATIVE_RIDGE':phi@h}
        for a in cfg['arms'][1:]:
            up=load(f'expert_{a}_{task}.private.pt');all_experts[a].append({'upper.'+k:v.numpy().astype(np.float64) for k,v in up.items()})
            ss=[dict(e,head=hh.numpy().astype(np.float64)) for e,hh in zip(all_experts[a],pack['heads'][a])]
            values[a]=np.concatenate([np_scores(qt[i:i+cfg['microbatch']],ss,cfg['layernorm_epsilon']) for i in range(0,len(qt),cfg['microbatch'])])
            probe=pack['probe'];before=probe['before'][a];ps=[dict(e,head=hh.numpy().astype(np.float64)) for e,hh in zip(all_experts[a][:-1],before['heads'])]+[{k:v.numpy().astype(np.float64) for k,v in before['current'].items()}]
            sc=np_scores(probe['tokens'].numpy(),ps,cfg['layernorm_epsilon']);sc-=sc.max(1,keepdims=True);yy=probe['labels'].numpy();ce=float(np.mean(np.log(np.exp(sc).sum(1))-sc[np.arange(64),yy]));pe=abs(ce-probe['losses'][a]);probe_error=max(probe_error,pe);assert pe<=cfg['FP32_probe_tolerance']
            if a=='FULL_SCORE_FROZEN' and task:
                for old,newh in zip(previous_heads,pack['heads'][a]):assert torch.equal(old,newh)
        previous_heads=pack['heads']['FULL_SCORE_FROZEN']
        if task==0:
            assert all(torch.equal(x,y) for x,y in zip(pack['heads']['FULL_SCORE_FROZEN'],pack['heads']['FEATURE_FROZEN']))
            for k,v in all_experts['FULL_SCORE_FROZEN'][0].items():assert np.array_equal(v,all_experts['FEATURE_FROZEN'][0][k])
        saved=load(f'scores_{task}.private.pt');metrics={}
        for a,v in values.items():
            error=float(abs(v-saved[a].numpy()).max());score_error=max(score_error,error);mismatch+=int((v.argmax(1)!=saved[a].numpy().argmax(1)).sum());assert error<=cfg['score_absolute_tolerance'];pred=np.array(seen)[v.argmax(1)];lab=labels[np.isin(labels,seen)];correct={str(c):int(((pred==c)&(lab==c)).sum()) for c in seen};metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/len(v));score_rows+=len(v)
        stages.append(dict(task=task,seen=seen,metrics=metrics));write(p/'verify_status.private.json',dict(phase='all_readouts',task=task,score_rows=score_rows,updated=time.time()))
    costs=json.loads((p/'COSTS.json').read_text());assert count==340 and score_rows==4050 and draw_rows==352000 and mismatch==0
    assert costs['science_train_encodes']==68058 and costs['validation_encodes']==300 and costs['qualification_encodes']==20 and costs['qualification_discarded_updates']==4 and costs['native_solves']==6
    for a in cfg['arms'][1:]:assert costs['arms'][a]==dict(updates=5500,forward_rows=352000,backward_rows=352000)
    result,decision=report(stages,cfg,train)
    for name,value in [('RESULTS.json',result),('DECISION.json',decision),('VERIFICATION.json',dict(state='PASS',full_lower_query_and_proof_rows=count,score_groups=18,score_rows=score_rows,score_error=score_error,prediction_mismatches=mismatch,probe_error=probe_error,FP16_bound_ratio=quant_ratio,draw_rows=draw_rows,wall_seconds=time.time()-start,scope='Independent lower10 and upper2 on saved embedding/lower interfaces; image embedding not reencoded, no full trajectory replay; current full moments independently checked online.'))]:write(p/name,value)
    write(p/'status.private.json',dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time()))

if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
