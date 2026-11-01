"""Synthetic algebra check only: exact finite-group clipped objective integration.

This checks no vision model and establishes no benefit of GRPO. Old probabilities,
reward tables, and group coefficients are constants during the surrogate update.
"""
import itertools
import json
import math
import time


def softmax(x):
    z=[math.exp(v-max(x)) for v in x]
    return [v/sum(z) for v in z]


def policy(theta):
    first=softmax(theta[:2])
    second=[softmax(theta[2:4]),softmax(theta[4:6])]
    return [(first[i]*second[i][j],(first[i],second[i][j]))
            for i in range(2) for j in range(2)]


def counts(total,bins):
    if bins==1:
        yield (total,)
    else:
        for k in range(total+1):
            for rest in counts(total-k,bins-1):
                yield (k,)+rest


def coefficients(old,rewards,group):
    levels=sorted(set(rewards))
    index=[levels.index(r) for r in rewards]
    mass=[sum(p[0] for p,t in zip(old,index) if t==k) for k in range(len(levels))]
    positive=[0.]*len(levels)
    negative=[0.]*len(levels)
    integrated_probability=0.
    cases=0
    for ns in counts(group,len(levels)):
        probability=math.factorial(group)
        for k,n in enumerate(ns):
            probability*=mass[k]**n/math.factorial(n)
        integrated_probability+=probability
        mean=sum(n*r for n,r in zip(ns,levels))/group
        std=math.sqrt(sum(n*(r-mean)**2 for n,r in zip(ns,levels))/group)
        for k,n in enumerate(ns):
            advantage=(levels[k]-mean)/(std+1e-8)
            positive[k]+=probability*n/group*max(advantage,0.)
            negative[k]+=probability*n/group*min(advantage,0.)
        cases+=1
    assert abs(integrated_probability-1)<1e-12
    return positive,negative,mass,index,cases


def ratios(old,current,t):
    return [a/b for a,b in zip(current[t][1],old[t][1])]


def reduced(old,current,co):
    positive,negative,mass,index,_=co
    loss=0.
    for t in range(len(old)):
        k=index[t]
        for ratio in ratios(old,current,t):
            clipped=max(.8,min(1.2,ratio))
            loss-=old[t][0]/mass[k]*(positive[k]*min(ratio,clipped)+negative[k]*max(ratio,clipped))
    return loss


def exhaustive(old,current,rewards,group):
    loss=0.
    for trajectory_group in itertools.product(range(len(old)),repeat=group):
        probability=math.prod(old[t][0] for t in trajectory_group)
        rr=[rewards[t] for t in trajectory_group]
        mean=sum(rr)/group
        std=math.sqrt(sum((r-mean)**2 for r in rr)/group)
        for t,r in zip(trajectory_group,rr):
            advantage=(r-mean)/(std+1e-8)
            for ratio in ratios(old,current,t):
                loss-=probability/group*min(ratio*advantage,max(.8,min(1.2,ratio))*advantage)
    return loss


def run():
    cpu=time.process_time();start=time.perf_counter()
    old=policy([.1,-.2,.3,-.1,-.4,.2])
    rewards=[-.7,.2,.2,1.1]
    co=coefficients(old,rewards,3)
    points=[[.3,-.4,.1,-.6,.8,-.7],[-1.1,.6,.9,-.8,.2,-.3],[.1,-.2,.3,-.1,-.4,.2]]
    loss_error=0.;derivative_error=0.
    for theta in points:
        current=policy(theta)
        loss_error=max(loss_error,abs(exhaustive(old,current,rewards,3)-reduced(old,current,co)))
        for j in range(len(theta)):
            plus=theta.copy();minus=theta.copy();plus[j]+=1e-6;minus[j]-=1e-6
            direct=(exhaustive(old,policy(plus),rewards,3)-exhaustive(old,policy(minus),rewards,3))/2e-6
            exact=(reduced(old,policy(plus),co)-reduced(old,policy(minus),co))/2e-6
            derivative_error=max(derivative_error,abs(direct-exact))
    zero=coefficients(old,[.5]*4,8)
    assert reduced(old,policy(points[0]),zero)==0
    assert loss_error<1e-12 and derivative_error<2e-9
    # This is raw expected reward, not the normalized/clipped grouped surrogate.
    raw=sum(p[0]*r for p,r in zip(policy(points[0]),rewards))
    assert abs(raw-(-reduced(old,policy(points[0]),co)))>1e-4
    return {'passed':True,'scope':'synthetic objective integration, no model or optimizer',
            'two_slot_trajectory_count':4,'group_size':3,'ordered_group_cases':4**3,
            'distinct_reward_levels':len(set(rewards)),'multinomial_cases':co[-1],
            'max_loss_error':loss_error,'max_finite_difference_error':derivative_error,
            'same_reward_zero_group_pass':True,'raw_reward_objective_distinct':True,
            'prospective_16_trajectory_G8_worst_distinct_reward_count_cases':math.comb(23,15),
            'prospective_16_trajectory_G8_ordered_groups':16**8,
            'new_real_forward':0,'optimizer_steps':0,'VAL_TEST_reads':0,
            'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
