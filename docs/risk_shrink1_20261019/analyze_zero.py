"""Exact certificate verification from saved quadratics; no fitting or scientific solve."""
import json,os,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
start=time.time();root=Path(os.environ['RUN_OUTPUT'])
p=np.load(root/'problem_2.private.npz');candidate=np.load(root/'zero_direction_certificate.private.npz')
audit=json.loads((root/'TRAINING_AUDIT.json').read_text())[2]
# Round certificate weights for a compact, exact rational witness; do not alter problem data.
weights=[F(format(float(x),'.8f')) for x in candidate['multipliers_normalized']]
total=sum(weights);weights=[x/total for x in weights];support=[i for i,x in enumerate(weights) if x]
assert weights and all(x>=0 for x in weights) and sum(weights)==1
k=len(p['b']);scales=[F.from_float(max(float(x),1e-8)) for x in p['baselines']]
factors={i:weights[i]/scales[i] for i in support}
v=[sum(2*factors[i]*F.from_float(float(p['bs'][i,j])) for i in support) for j in range(k)]
q=[[sum(factors[i]*F.from_float(float(p['Qs'][i,j,l])) for i in support) for l in range(k)] for j in range(k)]
assert all(q[j][l]==q[l][j] for j in range(k) for l in range(k))
# Exact rational LDL decomposition proves positive definiteness, not just a float eigenvalue sign.
l=[[F(int(i==j)) for j in range(k)] for i in range(k)];diagonal=[]
for j in range(k):
 pivot=q[j][j]-sum(l[j][h]*l[j][h]*diagonal[h] for h in range(j))
 assert pivot>0;diagonal.append(pivot)
 for i in range(j+1,k):l[i][j]=(q[i][j]-sum(l[i][h]*l[j][h]*diagonal[h] for h in range(j)))/pivot
assert all(sum(l[i][h]*diagonal[h]*l[j][h] for h in range(min(i,j)+1))==q[i][j] for i in range(k) for j in range(k))
gamma=F(1468,10_000_000);assert all(x>=gamma for x in v)
eps=F(str(json.loads((root/'config.private.json').read_text())['feasibility_tolerance']))
bound=eps/gamma
controls=[]
for stage in [0,1]:
 a=json.loads((root/'TRAINING_AUDIT.json').read_text())[stage]
 aa=np.array(a['risk_limited_alpha']); pp=np.load(root/f'problem_{stage}.private.npz')
 assert aa.max()>0 and a['solver']['success']
 direction=aa/aa.sum(); linear=2*pp['bs']@direction/np.maximum(pp['baselines'],1e-8)
 controls.append(dict(stage=stage,accepted_nonzero_alpha_max=float(aa.max()),normalized_direction_objective_derivative=float(2*pp['b']@direction),max_linearized_normalized_constraint=float(linear.max()),saved_full_constraint_violation=a['solver']['max_normalized_violation']))
rows=[]
for i in support:
 m=audit['constraints'][i]
 rows.append(dict(constraint_index=i,fold=m['fold'],class_id=m['class_id'],kind=m['kind'],images=m['images'],patients=m['patients'],weight_exact=str(weights[i]),weight_float=float(weights[i])))
result=dict(status='EXACT_RATIONAL_CERTIFICATE_VERIFIED',stage=2,classes=k,scope='saved float64 quadratic data interpreted as exact rational numbers',nonzero_linearized_feasible_direction_exists=False,nonzero_exact_quadratic_feasible_point_exists=False,unique_exact_feasible_point='alpha=0',zero_is_global_optimum_for_any_objective_over_this_feasible_set=True,certificate_support=rows,normalized_gradient_combination=[float(x) for x in v],gradient_lower_bound_exact=str(gamma),weighted_quadratic_positive_definite_exact_LDL=True,weighted_quadratic_LDL_pivots_float=[float(x) for x in diagonal],maximum_L1_alpha_under_original_normalized_feasibility_tolerance=float(bound),feasibility_tolerance_exact=str(eps),gradient_lower_bound_verified_with_exact_arithmetic=True,first_two_stages=controls,full_risk_and_solver_calls_added=0,new_training_encodes=0,new_validation_encodes=0,diagnostic_LP_calls_before_certificate=2,diagnostic_LP_search_seconds_approximately=.02779531478881836,certificate_verification_seconds=time.time()-start)
(root/'ZERO_DIRECTION_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
