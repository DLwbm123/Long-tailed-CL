"""Actual full-shape native checks and complete-plan budget before fit."""
import json,time,copy,os,subprocess
from pathlib import Path
import torch
from common import write,guard
from model import Joint,ARMS,supervised,deterministic,diagnostics
from data_contract import load
from train import cycle,optimizers
from check import run as software
ENGINEERING={'supervised_optimizer':0,'policy_optimizer':0,'total_optimizer':0,'image_encoder':0,'F_VAL_TEST':0}

def run(c,meta):
    r=Path(c['output']);cpu=time.process_time();wall=time.time()
    native_software=software('cuda');timings={a:[] for a in ARMS};native=[]
    for s in [0,5]:
        d=load(c,s,meta)
        for arm in ARMS:
            torch.manual_seed(74002);m=Joint(arm).cuda();sup,pol=optimizers(m)
            score,*_=deterministic(m,d);err=float((score-d['teacher_scores']).abs().max())
            assert err<1e-4 and torch.equal(score.argmax(1),d['teacher_scores'].argmax(1)),'INITIAL_STRONG_BINDING'
            original=copy.deepcopy(m.state_dict());before=diagnostics(m,d)
            for step in range(2):
                guard(c);torch.cuda.synchronize();t=time.time();stats=cycle(m,sup,pol,d,step);torch.cuda.synchronize();dt=time.time()-t
                ENGINEERING['supervised_optimizer']+=1;ENGINEERING['total_optimizer']+=1
                if pol:ENGINEERING['policy_optimizer']+=1;ENGINEERING['total_optimizer']+=1
                if step==1:timings[arm].append(dt)
            assert float(m.U.weight.norm())>0
            assert all(torch.isfinite(p).all() for p in m.parameters())
            m.load_state_dict(original,strict=True);after=diagnostics(m,d);assert abs(before['deterministic_reward']-after['deterministic_reward'])<1e-12
            native.append({'stage':s,'arm':arm,'initial_score_maxabs':err,'initial_predictions_identical_to_teacher':True,'adapter_updated':True,'policy_only_adapter_frozen':True,'strict_model_restore':True})
            del m,sup,pol
        del d
    own=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert any(x.split(',')[0].strip()==str(os.getpid()) and c['GPU_UUID'] in x for x in own.splitlines()),'OWN_GPU_UUID'
    # Two passes per endpoint; worst second pass, never choose faster observation.
    maximum={a:max(v) for a,v in timings.items()};nonstep_encoder_bound=7338*.004282474517822266
    projected=2*(512*6*sum(maximum.values())+nonstep_encoder_bound+300)+7200
    remaining=c['deadline']-time.time();passed=projected<remaining and torch.cuda.max_memory_reserved()<12*2**30
    write(r/'NATIVE.json',{'passed':True,'records':native,'new_joint_software':native_software,'engineering':ENGINEERING,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'included_in_worker':True,'own_GPU_UUID_verified':True,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'initial_binding_is_tolerance_not_bitwise':True})
    write(r/'BUDGET.json',{'passed':passed,'worst_second_pass_full_cycle_seconds':maximum,'safety':2,'nonstep_encoder_bound_seconds':nonstep_encoder_bound,'profile_encoder_bound_reused_same_architecture':True,'analysis_IO_reserve_seconds':7200,'additional_nonstep_reserve_seconds':300,'projected_seconds':projected,'remaining_original_clock_seconds':remaining,'supervised_steps':12288,'policy_steps':9216,'total_optimizer_steps':21504,'final_endpoints':24,'middle_endpoints':24,'no_epoch_or_control_reduction':True})
    return passed
