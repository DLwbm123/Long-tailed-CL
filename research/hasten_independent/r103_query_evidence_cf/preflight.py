"""Native correctness and whole two-stream cost bound, no scientific gain gate."""
import time,subprocess,os
from pathlib import Path
import torch
import torch.nn.functional as F
from teacher_base import new_tail
from teacher_sft_decision import query_risk
from fusion import Fusion,full_step
from evidence_views import crop,prefix
from router import Router
from objectives import objective
from common import write,guard
from PIL import Image

def run(c,rows,y,store,counts):
    chosen=[];seen=set()
    for i in sorted(store.position):
        g=rows[i]['identity_component']
        if g not in seen:chosen.append(i);seen.add(g)
        if len(chosen)==2:break
    assert len(chosen)==2;timings=[]
    for _ in range(2):
        torch.cuda.synchronize();start=time.time();pixels=[]
        for i in chosen:
            with Image.open(Path(c['image_root'])/rows[i]['relative_path']) as im:
                im=im.convert('RGB');views=[store.tf(crop(im,k)) for k in range(6)]
                assert all(not torch.equal(views[0],v) for v in views[1:]),'CROPS_MUST_BE_REAL_DISTINCT_INPUTS'
                pixels.extend(views)
        tok=prefix(store,torch.stack(pixels).cuda());torch.cuda.synchronize();timings.append((time.time()-start)/12);counts['engineering_prefix_rows']+=12
    m,params=new_tail(store.bundle,c['seed']);tx,raw,_=store.batch(chosen)
    with torch.no_grad():err=float((m(tx)-raw).abs().max())
    assert err<=2e-5
    # Initial and final class counts use legal repeated engineering tokens only.
    tail_times=[];fusion_times=[];router_times=[]
    for C in [5,23]:
        m,params=new_tail(store.bundle,c['seed']);opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001);tokens=tx.repeat(32,1,1);base=raw.repeat(32,1);labels=torch.arange(64,device='cuda')%C;P=F.normalize(torch.randn(C,512,device='cuda'),dim=-1)
        for _ in range(2):
            torch.cuda.synchronize();st=time.time();opt.zero_grad(set_to_none=True)
            for micro in range(4):
                z=m(tokens);risk,_=query_risk(z,labels,P,torch.ones(C,device='cuda'),C);((risk+.01*(z-base).square().sum(-1)).sum()/256).backward()
            opt.step();torch.cuda.synchronize();dt=time.time()-st;counts['engineering_teacher_optimizer']+=1;counts['engineering_teacher_query_rows']+=256
        tail_times.append(dt/256)
        N=3331 if C==5 else 6821;zz=F.normalize(torch.randn(N,2,512,device='cuda'),dim=-1);yy=torch.arange(N,device='cuda')%C;nc=torch.bincount(yy,minlength=C).float();fm=Fusion().cuda();fo=torch.optim.AdamW(fm.parameters(),lr=1e-4,weight_decay=.01)
        for _ in range(2):
            torch.cuda.synchronize();st=time.time();full_step(fm,fo,zz,yy,P,nc);torch.cuda.synchronize();dt=time.time()-st;counts['engineering_fusion_optimizer']+=1;counts['engineering_fusion_fit_rows']+=N
        fusion_times.append(dt/N)
        for arm in ['SUP_CF','RAW_EXACT_CF','GROUP_EXACT_CF','GRPO_CF','GRPO_IN']:
            rm=Router().cuda();ro=torch.optim.AdamW(rm.parameters(),lr=.001,weight_decay=.0001);xx=torch.randn(128,3,19,device='cuda');cc=torch.randint(2,(128,3),device='cuda');mask=torch.arange(128,device='cuda')%2==0
            for _ in range(2):
                old=rm(xx).double().softmax(-1).detach();actions=torch.multinomial(old,8,replacement=True);torch.cuda.synchronize();st=time.time()
                for inner in range(4):
                    ro.zero_grad(set_to_none=True);objective(arm,rm(xx),old,cc,mask,C,max(1,C//2),actions).backward();torch.nn.utils.clip_grad_norm_(rm.parameters(),1.);ro.step();counts['engineering_router_optimizer']+=1
                torch.cuda.synchronize();dt=time.time()-st
            router_times.append(dt/4)
    # Per stream: sum of A+B arrived sizes = FULL arrived size. FULL/folds fusion
    # fit total is 4*sumN across the two heads; all three banks encode all queries.
    sumN=sum(sum(c['train_counts'][:sum(map(len,c['stages'][:s+1]))]) for s in range(5))
    # Metadata-only VAL class sizes; image bytes remain unread before global seal.
    import csv
    val=list(csv.DictReader((Path(c['manifest'])/'val.csv').open()));sumV=sum(sum(c['order'].index(int(row['original_label']))<sum(map(len,c['stages'][:s+1])) for row in val) for s in range(5))
    prefix_seconds=6*(sumN+sumV)*timings[1]
    teacher_seconds=80*sumN*max(tail_times)
    crop_tail_seconds=6*(3*sumN+sumV)*max(tail_times)
    fusion_seconds=512*4*sumN*max(fusion_times)
    router_seconds=512*25*max(router_times)
    projection=2*(prefix_seconds+teacher_seconds+crop_tail_seconds+fusion_seconds+router_seconds+600)+7200
    own=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);assert any(x.split(',')[0].strip()==str(os.getpid()) and c['GPU_UUID'] in x for x in own.splitlines())
    passed=projection<c['deadline']-time.time() and torch.cuda.max_memory_reserved()<12*2**30
    receipt={'passed':passed,'native_tail_parity_maxabs':err,'six_real_crop_inputs_verified':True,'actual_GPU_UUID_verified':True,'generic_prefix_per_view_second':timings[1],'tail_backward_per_row_second_worst':max(tail_times),'full_fit_fusion_per_row_second_worst':max(fusion_times),'router_step_second_worst':max(router_times),'sum_arrived_TRAIN_rows':sumN,'sum_stage_VAL_rows_metadata_only':sumV,'prefix_seconds':prefix_seconds,'fold_teacher_seconds':teacher_seconds,'crop_tail_seconds':crop_tail_seconds,'fusion_seconds':fusion_seconds,'router_seconds':router_seconds,'safety':2,'IO_analysis_reserve_seconds':7200,'projected_entire_stream_seconds':projection,'remaining_original_clock_seconds':c['deadline']-time.time(),'all_fixture_parameters_discarded':True,'engineering_counts':dict(counts),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'VAL_TEST_images':0}
    write(Path(c['output'])/'BUDGET.json',receipt);guard(c);return passed
