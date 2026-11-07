"""Aggregate four fixed feedback selectors; AUDIT never chooses an action."""
import csv,json
from pathlib import Path
from .supply import write,table


def mean(values):
    v=[x for x in values if x is not None];return sum(v)/len(v) if v else None


def ba(point,classes=None):
    vals=[a/n for c,(a,n) in enumerate(zip(point['correct'],point['denominator'])) if n and (classes is None or c in classes)]
    return mean(vals)


def run(root):
    root=Path(root);rows=[];all_candidates=[];classes=[];total_updates=0
    for name in ['a','b']:
        result=json.loads((root/name/'science/RESULTS.json').read_text());total_updates+=result['science_classifier_updates']
        for rec in result['roots']:
            candidates=rec['candidates'];tapes=rec['tapes'];tail=set(rec['metadata']['real_oldtail_shadow']);n=len(candidates)
            anchor=[i for i,c in enumerate(candidates) if c['family']=='anchor'];bal=next((i for i in anchor if candidates[i]['anchor']=='E-BAL'),0)
            fit=lambda i:mean([t['FIT'][i]['R'] for t in tapes])
            full=lambda role,i:bool(tail) and all(t[role][i]['full_oldtail_evaluable'] for t in tapes)
            qualified=lambda role,i:bool(tail) and full(role,i) and all(t[role][i]['qualified'] for t in tapes)
            def choose(ids):
                if tail and not full('FIT',0):return 0,'unknown_risk'
                allowed=[i for i in ids if i==0 or not tail or qualified('FIT',i)]
                chosen=max(allowed,key=lambda i:((fit(i) if fit(i) is not None else float('-inf')),-i))
                if chosen==0 or fit(chosen)<=0:return 0,'no_positive_qualified_FIT'
                return chosen,'FIT_qualified' if tail else 'risk_not_applicable'
            choices={'E-ZERO':(0,'fixed'),'E-BAL':(bal,'fixed'),'E-ANCHOR':choose(anchor),'E-LOCAL':choose(range(n))}
            for i in range(n):
                all_candidates.append({'stream':result['stream'],'seed':rec['seed'],'stage':rec['stage'],'scenario':rec['scenario'],'candidate':i,'family':candidates[i]['family'],
                                       'FIT_qualified':qualified('FIT',i),'AUDIT_complete':full('AUDIT',i),'false_safe':not qualified('AUDIT',i) if i and qualified('FIT',i) and full('AUDIT',i) else None})
            for rule,(i,reason) in choices.items():
                pts=[p for t in tapes for p in t['AUDIT'][i]['endpoints']];zeros=[p for t in tapes for p in t['AUDIT'][0]['endpoints']]
                observed=[p['tail'] for p in pts if p['tail'] is not None];oldBA=[ba(p,tail) for p in pts];zeroBA=[ba(p,tail) for p in zeros]
                complete=full('AUDIT',i);action=n>1 and rec['metadata']['S_root_tokens']>rec['metadata']['S_capacity'];informative=bool(tail) and action and complete
                rescued=[];harmed=[];floor_rescued=[];nonfloor_harmed=[]
                for p,z in zip(pts,zeros):
                    assert all(n==1 for n in p['denominator']),'QUERY_PACKAGE_CHANGED'
                    rescued.append(sum(max(p['correct'][c]-z['correct'][c],0) for c in tail));harmed.append(sum(max(z['correct'][c]-p['correct'][c],0) for c in tail))
                    if tail and ba(z,tail)==0:floor_rescued.append(rescued[-1])
                    if tail and ba(z,tail)>0:nonfloor_harmed.append(harmed[-1])
                row={'stream':result['stream'],'seed':rec['seed'],'stage':rec['stage'],'scenario':rec['scenario'],'rule':rule,'selected_candidate':i,'family':candidates[i]['family'],
                     'R':mean([t['AUDIT'][i]['R'] for t in tapes]),'macro_delta':mean([t['AUDIT'][i]['macro'] for t in tapes]),'absolute_macro_BA':mean([ba(p) for p in pts]),'E_ZERO_absolute_macro_BA':mean([ba(p) for p in zeros]),
                     'absolute_oldtail_BA':mean(oldBA) if complete else None,'E_ZERO_absolute_oldtail_BA':mean(zeroBA) if complete else None,
                     'oldtail_mean_delta':mean(observed) if complete else None,'worst_oldtail_delta':min(observed) if observed else None,'terminal_oldtail_delta':mean([t['AUDIT'][i]['endpoints'][-1]['tail'] for t in tapes]) if complete else None,
                     'observed_tail_violations':sum(x<-.005 for x in observed),'observed_tail_checks':len(observed),'nonzero':i!=0,'reason':reason,'unknown_fallback':reason=='unknown_risk',
                     'full_risk_evaluable':complete,'real_oldtail_present':bool(tail),'has_action':action,'informative':informative,
                     'E_ZERO_floor_horizons':sum(x==0 for x in zeroBA if x is not None),'E_ZERO_nonfloor_horizons':sum(x>0 for x in zeroBA if x is not None),
                     'oldtail_rescued_component_evaluations':sum(rescued),'oldtail_harmed_component_evaluations':sum(harmed),'floor_oldtail_rescued_evaluations':sum(floor_rescued),'nonfloor_oldtail_harmed_evaluations':sum(nonfloor_harmed),
                     'one_probe_macro_flip_pp':[100/len(p['denominator']) for p in pts[:3]],'one_probe_oldtail_flip_pp':100/len(tail) if tail else None,
                     'false_safe':not qualified('AUDIT',i) if i and qualified('FIT',i) and full('AUDIT',i) else None,
                     'logical_feedback_H3_returns':2*(len(anchor) if rule=='E-ANCHOR' else n if rule=='E-LOCAL' else 0),'physical_branches_shared':2*n,'patient_denominator':'NA'}
                rows.append(row)
                # Public class counts aggregate repeated probe evaluations; no IDs.
                for h in range(3):
                    for c in range(len(tapes[0]['AUDIT'][i]['endpoints'][h]['correct'])):
                        classes.append({'stream':result['stream'],'seed':rec['seed'],'stage':rec['stage'],'rule':rule,'horizon':h+1,
                                        'real_class':rec['metadata']['shadow_to_real'][c],'correct':sum(t['AUDIT'][i]['endpoints'][h]['correct'][c] for t in tapes),
                                        'denominator_evaluations':sum(t['AUDIT'][i]['endpoints'][h]['denominator'][c] for t in tapes),'is_protected_oldtail':c in tail})
    assert len(rows)==256 and total_updates<=1170864
    table(root/'P1C_RULE_SELECTION.csv',rows);table(root/'P1C_ALL_CANDIDATE_DIAGNOSTIC.csv',all_candidates)
    agg={}
    for r in classes:
        key=tuple(r[k] for k in ['stream','seed','stage','rule','horizon','real_class'])
        if key not in agg:agg[key]={**r,'correct':0,'denominator_evaluations':0,'same_probe_reused_across_roots':True,'patient_denominator':'NA'}
        agg[key]['correct']+=r['correct'];agg[key]['denominator_evaluations']+=r['denominator_evaluations']
    table(root/'P1C_CLASS_EVALUATIONS.csv',list(agg.values()))
    blocks=[]
    for stream in sorted({r['stream'] for r in rows}):
        for seed in [74003,74004]:
            for rule in ['E-ZERO','E-BAL','E-ANCHOR','E-LOCAL']:
                rr=[r for r in rows if r['stream']==stream and r['seed']==seed and r['rule']==rule];ir=[r for r in rr if r['informative']];fs=[r['false_safe'] for r in rr if r['false_safe'] is not None]
                blocks.append({'stream':stream,'seed':seed,'rule':rule,'roots':len(rr),'complete_R_roots':sum(r['R'] is not None for r in rr),'mean_R':mean([r['R'] for r in rr]) if all(r['R'] is not None for r in rr) else None,'mean_macro_delta':mean([r['macro_delta'] for r in rr]),
                               'absolute_macro_BA':mean([r['absolute_macro_BA'] for r in rr]),'informative_roots':len(ir),'informative_mean_R':mean([r['R'] for r in ir]),
                               'nonzero_roots':sum(r['nonzero'] for r in rr),'informative_nonzero':sum(r['nonzero'] for r in ir),'unknown_fallback_roots':sum(r['unknown_fallback'] for r in rr),
                               'floor_horizons':sum(r['E_ZERO_floor_horizons'] for r in rr),'nonfloor_horizons':sum(r['E_ZERO_nonfloor_horizons'] for r in rr),
                               'false_safe_numerator':sum(fs),'false_safe_denominator':len(fs),'false_safe_rate':sum(fs)/len(fs) if fs else None})
    table(root/'P1C_BLOCK_SUMMARY.csv',blocks)
    increments=[]
    for stream in sorted({r['stream'] for r in rows}):
        for seed in [74003,74004]:
            a=next(b for b in blocks if b['stream']==stream and b['seed']==seed and b['rule']=='E-ANCHOR');l=next(b for b in blocks if b['stream']==stream and b['seed']==seed and b['rule']=='E-LOCAL')
            increments.append({'stream':stream,'seed':seed,'L_minus_A_R':l['mean_R']-a['mean_R'] if l['mean_R'] is not None and a['mean_R'] is not None else None,'informative_L_minus_A_R':l['informative_mean_R']-a['informative_mean_R'] if l['informative_mean_R'] is not None and a['informative_mean_R'] is not None else None,
                               'local_mean_R':l['mean_R'],'informative_local_mean_R':l['informative_mean_R'],'informative_nonzero':l['informative_nonzero']})
    table(root/'P1C_LOCAL_INCREMENT.csv',increments)
    d=all(x['L_minus_A_R'] is not None and x['L_minus_A_R']>0 and x['local_mean_R']>=0 and x['informative_L_minus_A_R'] is not None and x['informative_L_minus_A_R']>0 and x['informative_local_mean_R']>=0 and x['informative_nonzero'] for x in increments)
    write(root/'P1C_RESULTS.json',{'stage':'real classifier branches completed','decision':'D_development_signal_only' if d else 'C_no_consistent_local_advantage',
                                'science_classifier_updates':total_updates,'blocks':blocks,'increments':increments,'independent_population_confirmation':False,'controller_RL_updates':0,'VAL_TEST':0,'patient_denominator':'NA','automatic_successor':False})
