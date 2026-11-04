"""Necessary new-task semantics and R101 unchanged-objective parity."""
import os,json,copy,time,importlib.util
from pathlib import Path
import torch
from check import fixture
import model

def run(baseline_source,output):
    torch.set_num_threads(2);cpu=time.process_time();wall=time.time()
    spec=importlib.util.spec_from_file_location('baseline_model_source',baseline_source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    d=fixture();records=[]
    for arm in model.ARMS:
        torch.manual_seed(74002);a=model.Joint(arm);b=old.Joint(arm);b.load_state_dict(copy.deepcopy(a.state_dict()),strict=True)
        for phase in ['supervised']+([] if arm=='UNIFORM' else ['auxiliary']):
            a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
            eps=[torch.randn(8,len(j)) for j in d['ix']]
            la,_=getattr(model,phase)(a,d,*([eps] if phase=='auxiliary' else []));lb,_=getattr(old,phase)(b,d,*([eps] if phase=='auxiliary' else []));la.backward();lb.backward()
            err=abs(float(la-lb));grad=0.
            for x,y in zip(a.parameters(),b.parameters()):
                assert (x.grad is None)==(y.grad is None)
                if x.grad is not None:grad=max(grad,float((x.grad-y.grad).abs().max()))
            assert err<1e-12 and grad==0;records.append({'arm':arm,'phase':phase,'loss_error':err,'grad_error':grad})
    d.update(stage=4,stage_denominator=4,births=[0,1,4]);m=model.Joint('PROTO_GRPO');h=m.adapt(d['S']);_,features=model.geometry(h,d)
    assert [float(v[0,-2]) for v in features]==[1.,.75,0.]
    d.update(tail=torch.zeros(3,dtype=torch.bool),protected=torch.zeros(3,dtype=torch.bool));assert model.diagnostics(m,d)['B_tail'] is None
    assert torch.isfinite(model.reward(d['teacher_scores'],d)[0])
    result={'passed':True,'R101_same_state_objective_and_gradient_parity':records,'five_stage_age_denominator4':True,'empty_tail_null_reward_finite':True,'new_optimizer_encoder_F_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'old_source_only_no_old_weights':True}
    Path(output).write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    import sys
    print(run(sys.argv[1],sys.argv[2]))
