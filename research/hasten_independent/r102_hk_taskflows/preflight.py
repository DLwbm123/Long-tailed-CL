"""Fresh-teacher native checks plus entire new-stream workload admission."""
import time,math,os,subprocess,copy
from pathlib import Path
import torch
import torch.nn.functional as F
from common import write,guard
from teacher import new_tail
from teacher_sft_decision import query_risk
from folds import assign_folds
from model import Joint,ARMS,deterministic
from train import cycle,optimizers
from data_contract import load
ENGINEERING={'teacher_optimizer':0,'teacher_query_rows':0,'prefix_rows':0,'module_supervised_optimizer':0,'module_policy_optimizer':0,'VAL_TEST':0}

def teacher_budget(c,rows,y,store):
    r=Path(c['output']);cpu=time.process_time();wall=time.time();seen=sum(c['stages'][:1],[])
    ids=[i for i in range(len(rows)) if int(y[i]) in seen];chosen=[];comps=set()
    for i in ids:
        if rows[i]['identity_component'] not in comps:chosen.append(i);comps.add(rows[i]['identity_component'])
        if len(chosen)==2:break
    assert len(chosen)==2
    prefix_times=[]
    for _ in range(2):
        torch.cuda.synchronize();t=time.time();tokens,raw=store.prefix(chosen);torch.cuda.synchronize();prefix_times.append((time.time()-t)/2);ENGINEERING['prefix_rows']+=2
    m,params=new_tail(store.bundle,c['seed'])
    with torch.no_grad():error=float((m(tokens)-raw).abs().max())
    assert error<=2e-5,'GENERIC_PREFIX_NATIVE_PARITY'
    tx=tokens.repeat(32,1,1);bx=raw.repeat(32,1);times=[]
    for C in [5,23]:
        m,params=new_tail(store.bundle,c['seed']);opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001)
        proto=F.normalize(torch.randn(C,512,device='cuda'),dim=1);yy=torch.arange(64,device='cuda')%C;nc=torch.ones(C,dtype=torch.long,device='cuda')
        for _ in range(2):
            guard(c);torch.cuda.synchronize();t=time.time();opt.zero_grad(set_to_none=True)
            for micro in range(4):
                z=m(tx);risk,_=query_risk(z,yy,proto,nc,C);loss=(risk+.01*(z-bx).square().sum(1)).sum()/256;loss.backward();ENGINEERING['teacher_query_rows']+=64
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params) and any(float(p.grad.norm())>0 for p in params)
            opt.step();torch.cuda.synchronize();elapsed=time.time()-t;ENGINEERING['teacher_optimizer']+=1
        times.append(elapsed/256)
    totalN=0;expected_steps=0;micro=0;arrivals=[]
    for s in range(5):
        C=sum(map(len,c['stages'][:s+1]));ii=[i for i in range(len(rows)) if y[i]<C];fold=assign_folds(y[ii].tolist(),[rows[i]['identity_component'] for i in ii]);sizes=[fold.count(k) for k in [0,1]]
        totalN+=len(ii);expected_steps+=10*sum(math.ceil(n/256) for n in sizes);micro+=10*sum(math.ceil(n/64) for n in sizes);arrivals.append({'stage':s,'N':len(ii),'fold_rows':sizes})
    # Backward+Adam cost is a conservative bound for every support/observer/cache tail row.
    teacher_seconds=66*totalN*max(times);generic_seconds=(6821+5*1698)*prefix_times[1]
    module_bound=512*5*(.016728402+.057088852+.059318305+.061660767)*2
    projection=2*(teacher_seconds+generic_seconds+module_bound+600)+7200
    own=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert any(x.split(',')[0].strip()==str(os.getpid()) and c['GPU_UUID'] in x for x in own.splitlines())
    passed=projection<c['deadline']-time.time() and torch.cuda.max_memory_reserved()<12*2**30
    write(r/'TEACHER_BUDGET.json',{'passed':passed,'stage_shapes':arrivals,'expected_teacher_steps':expected_steps,'expected_teacher_micro':micro,'sum_arrived_TRAIN':totalN,'tail_full_update_per_row_second_worst':max(times),'generic_prefix_plus_preprocess_per_row_second':prefix_times[1],'projected_entire_stream_seconds':projection,'remaining_original_clock_seconds':c['deadline']-time.time(),'safety':2,'IO_analysis_reserve_seconds':7200,'tail_rows_bound_multiplier':66,'module_historical_bound_CPU_factor':2,'engineering':ENGINEERING,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'included_in_worker':True,'native_base_tail_maxabs':error,'actual_GPU_UUID_verified':True,'new_teacher_only':True})
    del m,params,opt,tx,bx,tokens,raw;torch.cuda.empty_cache();return passed

def module_budget(c,meta):
    r=Path(c['output']);timings={a:[] for a in ARMS};bindings=[];cpu=time.process_time();wall=time.time()
    for s in [0,4]:
        d=load(c,s,meta)
        for arm in ARMS:
            torch.manual_seed(c['seed']);m=Joint(arm).cuda();sup,pol=optimizers(m);score,*_=deterministic(m,d);err=float((score-d['teacher_scores']).abs().max());assert err<1e-4 and torch.equal(score.argmax(1),d['teacher_scores'].argmax(1))
            for step in range(2):
                torch.cuda.synchronize();t=time.time();cycle(m,sup,pol,d,step);torch.cuda.synchronize()
                if step==1:timings[arm].append(time.time()-t)
                ENGINEERING['module_supervised_optimizer']+=1
                if pol:ENGINEERING['module_policy_optimizer']+=1
            bindings.append({'stage':s,'arm':arm,'initial_score_maxabs':err,'initial_predictions_match':True});del m,sup,pol
        del d
    maximum={a:max(v) for a,v in timings.items()};projection=2*(512*5*sum(maximum.values())+300)+7200
    passed=projection<c['deadline']-time.time()
    write(r/'MODULE_BUDGET.json',{'passed':passed,'worst_second_cycle_seconds':maximum,'projected_remaining_seconds':projection,'remaining_original_clock_seconds':c['deadline']-time.time(),'supervised_steps':10240,'policy_steps':7680,'total_steps':17920,'bindings':bindings,'engineering':ENGINEERING,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'included_in_worker':True})
    return passed
