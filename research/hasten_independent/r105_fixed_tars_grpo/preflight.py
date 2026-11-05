"""Fixed producer parity, real TRAIN crops, all-arm gradient/profile and full budget."""
import json,time,subprocess,os,gc
from pathlib import Path
import torch
from common import write,guard
from router import Router,ARMS,features
from objectives import objective

def run(c,rows,y,head,table,weights,counts):
    from teacher_base import Prefixes,new_tail
    from teacher_vision import load_delta
    from evidence_views import encode_views
    from fusion import Fusion,scores
    r=Path(c['output']);p=Path(c['frozen_output']);store=Prefixes(c,rows,y);m,_=new_tail(store.bundle,c['seed']);load_delta(m,head['delta']);m.eval().requires_grad_(False);models=[]
    for a in [1,2]:
        fm=Fusion().cuda();fm.load_state_dict(torch.load(p/'s0/FULL'/f'fusion{a}_step512.private.pt',weights_only=False)['model']);fm.eval().requires_grad_(False);models.append(fm)
    assert not any(v.requires_grad for mod in [m,*models,store.visual] for v in mod.parameters())
    saved=torch.load(p/'s0/FULL/features.private.pt',map_location='cpu',weights_only=False);assert saved['ids']==list(range(6821))
    ss=scores(saved['views'],head['prototypes'],models);phi=features(ss,saved['views'].cuda(),head['train_counts'].cuda(),torch.zeros(23,device='cuda',dtype=torch.long),0)
    serr=float((ss-table['scores']).abs().max());ferr=float((phi-table['features']).abs().max());assert serr<=2e-5 and ferr<=1e-4;assert torch.equal(ss.argmax(-1)==y.cuda()[:,None],table['correct']);del ss,phi
    selected=[];groups=set()
    for i,row in enumerate(rows):
        if row['identity_component'] not in groups:selected.append(i);groups.add(row['identity_component'])
        if len(selected)==2:break
    times=[];errors=[]
    for _ in range(2):
        torch.cuda.synchronize();st=time.perf_counter();z=encode_views(c,rows,selected,store,{'FULL':m},counts,'ENGINEERING')['FULL'];torch.cuda.synchronize();times.append((time.perf_counter()-st)/2);err=float((z-saved['views'][selected]).abs().max());assert err<=2e-5;errors.append(err)
    timing=[];gradients={};torch.manual_seed(c['seed']);xx=table['features'][:128];cc=torch.tensor([[1,0,1],[0,1,0],[0,0,1],[1,1,0]],device='cuda').repeat(32,1);ww=weights[torch.arange(128,device='cuda')%23]
    for arm in ARMS:
        rm=Router().cuda();opt=torch.optim.AdamW(rm.parameters(),lr=.001,weight_decay=.0001)
        for _ in range(2):
            with torch.no_grad():old=rm(xx).double().softmax(-1).detach();actions=torch.multinomial(old,8,replacement=True)
            torch.cuda.synchronize();st=time.perf_counter()
            for inner in range(4):
                opt.zero_grad();loss=objective(arm,rm(xx),old,cc,ww,actions);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(rm.parameters(),1.);assert torch.isfinite(norm) and float(norm)>0;opt.step();counts['engineering_router_optimizer']+=1
            torch.cuda.synchronize();dt=(time.perf_counter()-st)/4
        timing.append(dt);gradients[arm]=float(norm);del rm,opt
    projection=2*(512*sum(timing)+1698*times[1]+300)+7200
    own=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);assert any(line.split(',')[0].strip()==str(os.getpid()) and c['GPU_UUID'] in line for line in own.splitlines())
    passed=projection<c['deadline']-time.time() and torch.cuda.max_memory_reserved()<12*2**30
    write(r/'BUDGET.json',{'passed':passed,'cached_FULL_scores_maxabs':serr,'cached_FULL_features_maxabs':ferr,'two_real_TRAIN_images_native_sixview_maxabs':max(errors),'upstream_frozen_no_grad':True,'own_GPU_verified':True,'fixture_parameters_discarded':True,'synthetic_candidate_rewards_used_only_for_engineering':True,'router_complete_step_second_pass':dict(zip(ARMS,timing)),'nonzero_finite_router_gradients':gradients,'sixview_second_pass_per_image_seconds':times[1],'router_updates':4096,'VAL_images_upper':1698,'safety':2,'IO_profile_overhead_reserve_seconds':300,'save_analysis_delivery_reserve_seconds':7200,'projected_entire_plan_seconds':projection,'remaining_original_clock_seconds':c['deadline']-time.time(),'engineering_counts':dict(counts),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'VAL_TEST_images':0})
    del m,models,saved,z;gc.collect();torch.cuda.empty_cache();guard(c);return passed,store
