"""New weighting value/gradient and unscaled-KL checks, no visual encoders."""
import json,time
from pathlib import Path
import torch
from objectives import objective,tailness
from grpo_reference import exact_group_surrogate,_brute_force,reference_kl
from common import write

def run(root):
    wall=time.perf_counter();cpu=time.process_time();torch.set_num_threads(1)
    logits=torch.tensor([[.05,.9,.05],[.9,.05,.05]],dtype=torch.double).log().requires_grad_();old=torch.tensor([[.8,.1,.1],[.2,.6,.2]],dtype=torch.double);cc=torch.tensor([[1,0,1],[0,1,0]]);w=torch.tensor([2.,7.],dtype=torch.double);reward=cc.double()-cc[:,0:1].double()
    a=(w*exact_group_surrogate(logits,old,reward)).mean();b=(w*_brute_force(logits,old,reward)).mean();ga,=torch.autograd.grad(a,logits,retain_graph=True);gb,=torch.autograd.grad(b,logits,retain_graph=True)
    err=float((a-b).abs().detach());gerr=float((ga-gb).abs().max());assert err<1e-11 and gerr<1e-11
    loss=objective('TARS_GROUP_EXACT',logits,old,cc,w);assert (loss-(-a+.01*reference_kl(logits).mean())).abs()<1e-12
    actions=torch.tensor([[0,1,2,0,1,2,0,1]]*2)
    for arm in ['SUP','RAW_EXACT','GROUP_EXACT','GRPO']:
        aa=objective(arm,logits,old,cc,torch.ones_like(w),actions);bb=objective('TARS_'+arm,logits,old,cc,torch.ones_like(w),actions);assert torch.equal(aa,bb)
    equal=torch.ones_like(cc);z=objective('TARS_GRPO',logits,old,equal,w,actions);assert (z-.01*reference_kl(logits).mean()).abs()<1e-12
    counts=torch.tensor([100,10,1,5]);correct=torch.tensor([100,5,0,4]);raw=(counts.double().mean()/counts*(1-correct/counts)).pow(.75);expected=(raw/((raw.sort().values[1]+raw.sort().values[2])/2)).clamp_min(1);assert torch.allclose(tailness(counts,correct),expected)
    out={'status':'PASS','weighted_45_counts_vs_6561_value_error':err,'weighted_gradient_error':gerr,'ones_recover_all_four_plain_objectives':True,'KL_not_weighted':True,'zero_surrogate_stays_zero_after_TARS':True,'midpoint_median':True,'formal_optimizer_encoder_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall}
    write(Path(root)/'SOFTWARE_CHECK.json',out);return out
if __name__=='__main__':run(Path(__file__).resolve().parent)
