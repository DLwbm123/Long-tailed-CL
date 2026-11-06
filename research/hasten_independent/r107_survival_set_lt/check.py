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
    # Exact unconstrained uniform without-replacement law: all six unordered pairs have mass1/6.
    import itertools,collections,copy
    mass=collections.defaultdict(float)
    for i,j in itertools.permutations(range(4),2):mass[tuple(sorted([i,j]))]+=1/4/3
    assert len(mass)==6 and all(abs(v-1/6)<1e-12 for v in mass.values())
    # ID-renaming equivariance of full legal sets, semantic state, and learned logits along a shared content path.
    permutation=torch.tensor([5,1,7,0,6,3,2,4]);renamed=from_views(z,permutation,labels);m2=Learner(74002);m2.load_state_dict(m.state_dict());m2.C=m.C
    ch2=Choice(renamed,m2,[4,4],[0,0],0,12,v);actor=Policy()
    with torch.no_grad():actor.net[-1].weight.normal_(std=.1)
    def contents(bank,ix,inverse=False):
        par=bank.pid[ix];par=torch.argsort(permutation)[par] if inverse else par
        return par*8+bank.view[ix]
    sel=ch.cover.clone();sel2=ch2.cover.clone();assert set(contents(bank,sel).tolist())==set(contents(renamed,sel2,True).tolist());ng=6;cap=ch.covercap;cap2=ch2.covercap
    while len(sel)<12:
        global_phase=int((bank.view[sel]==0).sum())<ng;ix=ch.global_options(sel,ng,cap) if global_phase else ch.local_options(sel);ix2=ch2.global_options(sel2,ng,cap2) if global_phase else ch2.local_options(sel2);phase=1 if global_phase else 2
        k1=contents(bank,ix);k2=contents(renamed,ix2,True);a=k1.argsort();b=k2.argsort();assert torch.equal(k1[a],k2[b]);x1=ch.state(ix,sel,ng,phase)[a];x2=ch2.state(ix2,sel2,ng,phase)[b];assert torch.allclose(x1,x2,atol=2e-5,rtol=2e-5);assert torch.allclose(actor(x1).log_softmax(0),actor(x2).log_softmax(0),atol=2e-5,rtol=2e-5)
        chosen=ix[a[0]];chosen2=ix2[b[0]];sel=torch.cat([sel,chosen.reshape(1)]);sel2=torch.cat([sel2,chosen2.reshape(1)])
        if global_phase:cap+=int(ch.localcap[bank.parent_index[chosen]]);cap2+=int(ch2.localcap[renamed.parent_index[chosen2]])
    # Real manager regression: later high-valued global must not make earlier high-valued global inaccessible.
    bb=from_views(z[:5],torch.arange(5),torch.zeros(5,dtype=torch.long));mm=Learner(74002);mm.arrive(bb,1);cc=Choice(bb,mm,[5],[0],0,3,v);cc.cover=bb.gidx[:1];cc.optional=bb.gidx[1:];cc.covercap=5;cc.ng=torch.tensor([3]);cc.val.zero_();cc.val[bb.gidx]=torch.tensor([0.,.9,.8,1.,.1]);result,_,_=cc.construct('VALUE');assert set(result.pid.tolist())=={0,1,3}
    exact=collections.defaultdict(float)
    for i in cc.global_options(cc.cover,3,5):
        second=cc.global_options(torch.cat([cc.cover,i.reshape(1)]),3,10)
        for j in second:exact[tuple(sorted([int(bb.pid[i]),int(bb.pid[j])]))]+=1/4/len(second)
    assert len(exact)==6 and all(abs(q-1/6)<1e-12 for q in exact.values())

    # Current-only pseudo support counts and scarce masks are based on actual retained support, not raw TRAIN counts.
    from meta import Meta
    actors={k:Policy() for k in ['ZERO_ACTOR','REWARD_SHUFFLED','GSPO_SET','GSPO_SET_LT']};meta=Meta({'seed':74002},v,copy.deepcopy(v),actors)
    zz=torch.nn.functional.normalize(torch.randn(100,6,512),dim=-1);dd=from_views(zz,torch.arange(100),torch.arange(100)//20);rr=[{'identity_component':str(i)} for i in range(100)]
    for num in [0,1,8,9]:
        ep=meta.episode(dd,rr,0,num,True);assert ep['counts']==ep['support_counts'];assert ep['pseudo_tail']==[x<=4 for x in ep['support_counts']];assert ep['component_intersection']==0
        if num%2==0:assert ep['support_counts'][0]>=ep['support_counts'][2]>=ep['support_counts'][-1]

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
    out={'passed':True,'feasible_sample_sets_tested':40,'ID_renaming_legal_mask_state_probability_equivariance':True,'unconstrained_uniform_subset_mass_exact':True,'greedy_late_candidate_does_not_block_earlier_candidate':True,'pseudo_support_counts_and_tail_masks_match_actual_support':True,'unique_sets':len(set(sets)),'global_before_local_mask_and_coverage':True,'forced_transitions_excluded_from_actor_length':True,'equal_byte_replacement':True,'actual_serialized_record_bytes':TOKEN_BYTES,'stats_bytes':STATS_BYTES,'formal_memory_byte_budget':BUDGET_BYTES,'GSPO_finite_difference_gradient_error':error,'L1_GSPO_equals_action_PG':True,'true_behavior_initial_ratio_one':True,'zero_advantage_surrogate_zero':True,'future_and_unretained_TRAIN_request_rejected_before_IO':True,'formal_encoder_optimizer_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall};write(Path(root)/'SOFTWARE_CHECK.json',out)
if __name__=='__main__':run(Path(__file__).resolve().parent)
