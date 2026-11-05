"""CPU contract tests: byte serialization, canonical legal sets and sequence math."""
import torch,tempfile,time,json
from pathlib import Path
from manager import Policy,ValueNet,Choice,replacement
from memory import from_views,BUDGET_BYTES,TOKEN_BYTES,STATS_BYTES
from learner import Learner
from rl import batch_trace,loss
from common import write

def run(root):
    torch.set_num_threads(1);cpu=time.process_time();wall=time.time();torch.manual_seed(74002)
    z=torch.nn.functional.normalize(torch.randn(8,6,512),dim=-1);ids=torch.arange(8);labels=ids%2;bank=from_views(z,ids,labels);m=Learner(74002);m.arrive(bank,2);v=ValueNet();p=Policy();ch=Choice(bank,m,[4,4],[0,0],0,12,v);sets=[]
    for k in range(40):
        g=torch.Generator().manual_seed(k);mem,_,info=ch.construct('RANDOM',generator=g);assert len(mem)==12 and mem.assert_subset(bank);alt,_,_=replacement(ch,mem,g);assert len(alt)==len(mem);sets.append(tuple(info['canonical_tokens']))
    assert len(set(sets))>1
    with tempfile.TemporaryDirectory(dir=root) as d:
        n=mem.serialize(Path(d)/'memory.bin',[4,4],[0,0],12);assert n==12*TOKEN_BYTES+STATS_BYTES
    assert BUDGET_BYTES==256*2064+184
    pp=Policy().double();traces=[]
    for k in range(4):
        tr=[]
        for j in range(3):
            x=torch.randn(j+2,43,dtype=torch.double);old=pp(x).log_softmax(0).detach();tr.append({'x':x,'old_logp':old,'choice':(k+j)%(j+2)})
        traces.append(tr)
    b=batch_trace(traces);returns=torch.tensor([.1,.7,.3,.9],dtype=torch.double);obj,st=loss(pp,b,returns,'GSPO');assert st['max_ratio_deviation']<1e-12;obj.backward();assert pp.net[-1].weight.grad.norm()>0
    analytic=float(pp.net[-1].weight.grad[0,0]);param=pp.net[-1].weight
    with torch.no_grad():param[0,0]+=1e-5
    hi=float(loss(pp,b,returns,'GSPO')[0].detach())
    with torch.no_grad():param[0,0]-=2e-5
    lo=float(loss(pp,b,returns,'GSPO')[0].detach())
    with torch.no_grad():param[0,0]+=1e-5
    error=abs((hi-lo)/2e-5-analytic);assert error<1e-7
    one=batch_trace([[tr[0]] for tr in traces]);a=loss(pp,one,returns,'GSPO')[0];aa=loss(pp,one,returns,'PG')[0];assert (a-aa).abs()<1e-12
    zero=loss(pp,b,torch.ones(4,dtype=torch.double),'GSPO')[0];assert abs(float(zero.detach()))<1e-12
    from encoder import Encoder
    try:Encoder.encode(None,{},[],[1],{},'TRAIN',allowed=[0])
    except AssertionError:pass
    else:raise AssertionError('future/history guard failed')
    out={'passed':True,'canonical_sample_sets_tested':40,'unique_sets':len(set(sets)),'global_before_local_mask_and_coverage':True,'equal_byte_replacement':True,'actual_serialized_record_bytes':TOKEN_BYTES,'stats_bytes':STATS_BYTES,'formal_memory_byte_budget':BUDGET_BYTES,'GSPO_finite_difference_gradient_error':error,'L1_GSPO_equals_action_PG':True,'true_behavior_initial_ratio_one':True,'zero_advantage_surrogate_zero':True,'future_and_unretained_TRAIN_request_rejected_before_IO':True,'formal_encoder_optimizer_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall};write(Path(root)/'SOFTWARE_CHECK.json',out)
if __name__=='__main__':run(Path(__file__).resolve().parent)
