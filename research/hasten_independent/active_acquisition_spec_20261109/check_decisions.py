"""Exhaustive specification witness, not a medical model or training experiment."""
import itertools
import json
import time
from pathlib import Path


ACTIONS=('stop','scan','inspect0','inspect1','inspect2','inspect3')


def transition(hidden,action):
    if action=='scan':return ('location',hidden),0
    if action=='stop':return ('stopped',None),0
    return ('hit' if int(action[-1])==hidden else 'miss',None),int(int(action[-1])==hidden)


def score(first,second):
    value=0.
    for hidden in range(4):
        observation,reward=transition(hidden,first)
        if first=='stop' or reward:value+=reward/4;continue
        _,later=transition(hidden,second(observation));value+=max(reward,later)/4
    return value


def expected_cost(first,second):
    total=0.
    for hidden in range(4):
        observation,reward=transition(hidden,first)
        if first=='stop':continue
        total+=(1 if reward else 1+int(second(observation)!='stop'))/4
    return total


def run():
    cpu=time.process_time();start=time.time()
    # An open-loop choice cannot condition the second action on new observations.
    open_loop=[score(a,lambda observation,b=b:b) for a,b in itertools.product(ACTIONS,repeat=2)]
    closed_loop=[0.]
    for followups in itertools.product(ACTIONS,repeat=4):
        closed_loop.append(score('scan',lambda observation,followups=followups:followups[observation[1]]))
    for first in ACTIONS[2:]:
        for second in ACTIONS:closed_loop.append(score(first,lambda observation,second=second:second))
    assert len(open_loop)==36 and len(closed_loop)==1321
    # One-step greedy chooses an inspect (.25 immediate) over scan (0 immediate).
    myopic=score('inspect0',lambda observation:'inspect1')
    chosen=score('scan',lambda observation:'inspect'+str(observation[1]))
    assert max(open_loop)==myopic==.5 and max(closed_loop)==chosen==1.
    assert chosen-myopic==.5
    scan_cost=expected_cost('scan',lambda observation:'inspect'+str(observation[1]))
    greedy_cost=expected_cost('inspect0',lambda observation:'inspect1')
    assert scan_cost==2 and greedy_cost==1.75
    # Exact randomized mixture matches expected cost as well as the hard cap.
    mixed_scan_probability=greedy_cost/scan_cost
    matched_cost_success=mixed_scan_probability*chosen
    assert mixed_scan_probability*scan_cost==greedy_cost and matched_cost_success==.875
    # Giving the policy the hidden location before it acts is an extra-information oracle.
    oracle=sum(transition(z,'inspect'+str(z))[1] for z in range(4))/4
    assert oracle==1.
    result={'scope':'constructed_finite_specification_only','new_model_queries_updates':0,'medical_examples':0,
            'latent_state':'uniform location0..3; initially unobserved','optional_acquisition_budget':2,
            'reward':'target evidence observed; not a measured medical classification reward',
            'open_loop_policies_enumerated':len(open_loop),'closed_loop_policies_enumerated':len(closed_loop),
            'best_open_loop_success':max(open_loop),'myopic_success':myopic,'best_closed_loop_success':max(closed_loop),
            'same_information_exact_control_success':max(closed_loop),'pre_action_hidden_location_oracle':oracle,
            'acquisition_costs':{'scan_then_inspect_expected':scan_cost,'myopic_expected':greedy_cost,
                                 'same_expected_cost_scan_stop_mixture_probability':mixed_scan_probability,
                                 'same_expected_cost_mixture_success':matched_cost_success,
                                 'hard_cap_for_all':2},
            'inferences':['Conditioning on a genuinely new observation can matter; not all two-step tasks collapse to a root-only bandit.',
                          'This tiny known model is exactly solvable; the gap does not show a need for sampled RL or medical efficacy.',
                          'An oracle with hidden-location access is not an admissible deployable comparison.'],
            'cost':{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'image_encodings':0,'optimizer_updates':0}}
    (Path(__file__).parent/'DECISION_WITNESS.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
