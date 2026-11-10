"""Algebraic identity checks only; no medical data, model or performance simulation."""
import json,math,itertools,time
from pathlib import Path


def run(output):
    start=time.perf_counter();cpu=time.process_time();max_residual=0.;count=0
    for bits in itertools.product([0.,1.],repeat=8):
        if sum(bits) in [0,8]:continue
        for h in [[i/7 for i in range(8)],[float(i%2) for i in range(8)],[.5]*8]:
            mu={b:sum(v for v,z in zip(h,bits) if z==b)/bits.count(b) for b in [0.,1.]}
            r=[b+.5*v for b,v in zip(bits,h)];mean_r=sum(r)/8;mean_b=sum(bits)/8
            for b,v,value in zip(bits,h,r):
                reconstructed=(1+.5*(mu[1.]-mu[0.]))*(b-mean_b)+.5*(v-mu[b]);max_residual=max(max_residual,abs(value-mean_r-reconstructed))
            count+=1
    assert max_residual<1e-14
    minimum_gap=min(1+.5*(a-b) for a,b in itertools.product([0.,1.],repeat=2));assert minimum_gap==.5
    q=[.05,.15,.3,.5];p=[.2,.1,.1,.6]
    kl=lambda a,b:sum(x*math.log(x/y) for x,y in zip(a,b))
    qO=sum(q[:2]);pO=sum(p[:2]);full=kl(q,p);parts=kl([qO,1-qO],[pO,1-pO])+qO*kl([x/qO for x in q[:2]],[x/pO for x in p[:2]])+(1-qO)*kl([x/(1-qO) for x in q[2:]],[x/(1-pO) for x in p[2:]])
    assert abs(full-parts)<1e-14
    result={'scope':'algebraic_identity_checks_not_medical_performance_or_RL_experiment','mixed_binary_group_cases':count,'maximum_reward_decomposition_residual':max_residual,'minimum_correct_minus_wrong_total_reward_lambda_half':minimum_gap,'CTAN_initial_coefficient_after128':.999**128,'CTAN_initial_coefficient_after512':.999**512,'full_candidate_KL_chain_rule_residual':abs(full-parts),'query_constant_reward_cancels_under_group_mean':True,'conditional_old_relation_leaves_old_new_mass_unidentified':True,'conditional_candidate_distribution_not_full_free_generation_semantic_probability':True,'new_model_calls':0,'new_image_encodings':0,'optimizer_updates':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}
    Path(output).write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(run(Path(__file__).with_name('ALGEBRA_CHECKS.json'))))
