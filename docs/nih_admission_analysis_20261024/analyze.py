"""One bounded, data-free mathematical analysis; not a re-run of a closed qualification."""
import json,math,struct,time
from pathlib import Path

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def add(a,b):return f32(f32(a)+f32(b))
def initial_adam_step(g,lr,eps):return -lr*g/(abs(g)+eps)

def run(out):
    start=time.time();lr=.0003;eps=1e-8;gtol=.0003;utol=.00002
    terms=[1.,1e-8,-1.]
    sequential=add(add(terms[0],terms[1]),terms[2])
    cancellation_first=add(add(terms[0],terms[2]),terms[1])
    exact=math.fsum(terms)
    a=initial_adam_step(sequential,lr,eps);b=initial_adam_step(cancellation_first,lr,eps)
    gerr=abs(sequential-cancellation_first);uerr=abs(a-b)
    result=dict(state='PENDING_ASSERTIONS',scope='Scalar IEEE FP32 reduction plus exact real-arithmetic first-step Adam map from zero moments; no Torch/GPU/NIH/old-artifact calls.',terms=terms,real_sum=exact,sequential_FP32=sequential,cancellation_first_FP32=cancellation_first,gradient_difference=gerr,gradient_tolerance=gtol,update_a=a,update_b=b,update_difference=uerr,update_tolerance=utol,gradient_predicate=gerr<gtol,update_predicate=uerr<utol,global_first_step_Lipschitz_bound=lr/eps,sufficient_global_gradient_bound_for_update_tolerance=utol*eps/lr,new_images=0,new_Torch_optimizer_updates=0,scalar_analytical_step_evaluations=2,old_qualification_reruns=0,old_failure_cause='UNKNOWN; the example does not recover or explain its missing errors.',started_epoch=start)
    # Persist all measured numbers before the assertions, including on a failed analysis.
    out.write_text(json.dumps(result,indent=2)+'\n')
    assert gerr<gtol and uerr>utol
    assert abs(a-b)<=(lr/eps)*gerr+1e-15
    result.update(state='PASS',elapsed_seconds=time.time()-start)
    out.write_text(json.dumps(result,indent=2)+'\n')
    return result
