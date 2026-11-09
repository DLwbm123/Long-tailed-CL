"""Saved group-composition audit and exact scalar PG conditioning identities."""
import csv
import json
import math
import time
from pathlib import Path


def run(base):
    started=time.time();cpu=time.process_time();base=Path(base);out=base/'method_audit_20261109'
    folders=['r118_reward_support_bootstrap','r120_native448_bootstrap','r121_r117_paired_seed']
    info=json.loads((base/'r114_tars_rl/TARS_WEIGHTS.json').read_text())
    rows=[];summaries=[]
    for folder in folders:
        verification=json.loads((base/folder/'FINAL_VERIFICATION.json').read_text())
        assert verification['passed']
        tiny=[c for c,n in enumerate(info['counts']) if n<25]
        for arm,learning in verification['learning'].items():
            if 'classes' not in learning:continue
            classes=learning['classes'];queries=sum(r['queries'] for r in classes);mixed=sum(r['mixed_groups'] for r in classes)
            assert queries==2048 and mixed>0
            tq=sum(classes[c]['queries'] for c in tiny);tm=sum(classes[c]['mixed_groups'] for c in tiny)
            for c,r in enumerate(classes):
                assert r['class']==c and r['queries']==sum(r[k] for k in ['allwrong_groups','allright_groups','mixed_groups'])
                rows.append({'experiment':folder,'arm':arm,'class':c,'tiny':c in tiny,'queries':r['queries'],'mixed_groups':r['mixed_groups'],
                             'share_of_all_queries':r['queries']/queries,'share_of_all_mixed_groups':r['mixed_groups']/mixed})
            summaries.append({'experiment':folder,'arm':arm,'queries':queries,'mixed_groups':mixed,'tiny_queries':tq,'tiny_mixed_groups':tm,
                              'tiny_share_before':tq/queries,'tiny_share_retained':tm/mixed,
                              'observed_zero_mixed_classes':[r['class'] for r in classes if r['mixed_groups']==0]})
    # Single-logit Bernoulli policy, same group-normalized binary reward, G8.
    fixtures=[];max_error=0.
    for p in [0.,.001,.01,.1,.5,.9,.999,1.]:
        G=8;terms=[];accepted=0.
        for k in range(G+1):
            probability=math.comb(G,k)*p**k*(1-p)**(G-k);fraction=k/G
            gradient=fraction*(1-fraction)/(math.sqrt(fraction*(1-fraction))+1e-6)
            assert 0<k<G or gradient==0
            terms.append((probability,gradient))
            if 0<k<G:accepted+=probability
        unconditional=sum(prob*g for prob,g in terms)
        conditional=sum(prob*g for prob,g in terms[1:-1])/accepted if accepted else None
        error=abs(unconditional-accepted*conditional) if accepted else abs(unconditional)
        assert error<1e-14;max_error=max(max_error,error)
        fixtures.append({'p':p,'mixed_probability':accepted,'unconditional_PG_direction':unconditional,'conditional_mixed_PG_direction':conditional,'global_mass_identity_error':error})
    # Nonzero off-reference full KL survives constant reward groups.
    q=[.5,.5];ps=[.01,.5];refs=[.1,.1]
    acceptance=[1-p**8-(1-p)**8 for p in ps]
    kl_gradient=[p*(1-p)*math.log((p/(1-p))/(r/(1-r))) for p,r in zip(ps,refs)]
    full=sum(w*g for w,g in zip(q,kl_gradient));only_mixed=sum(w*a*g for w,a,g in zip(q,acceptance,kl_gradient))
    assert abs(full-only_mixed)>1e-3
    result={'scope':'saved_group_counterfactual_and_scalar_math_not_performance','new_model_queries_updates':0,'summaries':summaries,
            'PG_conditioning':{'fixtures':fixtures,'max_error':max_error,'identity':'For fixed old policy and unchanged PG surrogate, g=0 outside mixed M, so E[g]=P(M)E[g|M]. Global rescaling does not imply identical Adam trajectories or equivalence of whole DAPO.'},
            'off_reference_KL_counterexample':{'class_current_p':ps,'class_reference_p':refs,'class_mixed_probability':acceptance,
                                              'full_expected_KL_logit_gradient':full,'mixed_mass_times_conditional_KL_gradient':only_mixed},
            'limitations':['Retained group share is not gradient share or actual dynamic-resampling accuracy.',
                           'Fixed saved responses do not predict a changed future policy; absolute mixed count is not reduced by merely discarding constants.',
                           'Zero observed mixed groups is not proof of zero population acceptance probability.',
                           'R117 lacks exact saved per-class group composition; excluded rather than reconstructed.',
                           'Synthetic Bernoulli identities are not VLM efficacy results.'],
            'cost':{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-started,'image_encodings':0,'optimizer_updates':0}}
    (out/'SAVED_GROUP_AND_MATH_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'CLASS_GROUP_COMPOSITION.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


if __name__=='__main__':
    result=run(Path(__file__).resolve().parent.parent)
    print(json.dumps({k:v for k,v in result.items() if k in ['summaries','off_reference_KL_counterexample','cost']},indent=2))
