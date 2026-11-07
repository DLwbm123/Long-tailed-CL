"""Post-hoc A1--A4 on sealed integer outcomes. No torch/model/query forward."""
import csv,json,os,random,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def csvout(path,rows):
    assert rows
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def mean(values):
    values=[x for x in values if x is not None]
    return sum(values)/len(values) if values else None


def score(endpoints,T):
    out=[]
    for e in endpoints:
        correct=np.array(e['correct']);n=np.array(e['denominator']);valid=n>0
        ba=float((correct[valid]/n[valid]).mean());mask=valid & np.array([c in T for c in range(len(n))])
        tail=float((correct[mask]/n[mask]).mean()) if mask.any() else None
        out.append((ba,tail))
    return out


def utility(deltas):
    macro=mean([x[0] for x in deltas]);tail=[x[1] for x in deltas if x[1] is not None]
    return macro-2*max([max(0.,-v-.005) for v in tail],default=0.),macro


def context(root,cfg):
    m=root['metadata'];stage=root['stage'];supported=sorted(set(range(m['real_seen_classes']))-set(m['unavailable_support_classes']))
    future=[c for c in cfg['stages'][stage] if c in supported];random.Random(m['seed']).shuffle(future)
    prefix=sorted(set(supported)-set(future[:2]));order=prefix+future[:2];mapping={c:i for i,c in enumerate(order)}
    realT={c for c in range(sum(map(len,cfg['stages'][:stage]))) if cfg['train_counts'][c]<250}
    T={mapping[c] for c in realT if c in mapping and mapping[c]<len(prefix)}
    assert len(T)==m['oldtail_classes']
    return order,mapping,realT,T


def run(source,metadata,output):
    start=time.time();cpu=time.process_time();source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(Path(metadata).read_text());original=np.array(metadata['original_labels']);coverage=[];classes=[];tails=[];opportunity=[];selections=[];risks=[];token_changes=[];components=[];rootmeta=[]
    streams={};max_error=0.;verified_components=0
    for folder in ['a','b']:
        cfg=json.loads((source/folder/'CONFIG.private.json').read_text());res=json.loads((source/folder/'RESULTS.json').read_text());stream=res['stream'];label=np.array([cfg['order'].index(int(x)) for x in original])
        stream_risks=[];stream_roots=[]
        for root in res['roots']:
            assert 'skipped' not in root
            stage=root['stage'];scenario=root['scenario'];meta=root['metadata'];n=len(root['candidates']);order,mapping,realT,T=context(root,cfg)
            prefix=len(order)-2;base={'stream':stream,'stage':stage,'scenario':scenario};A=[i for i,c in enumerate(root['candidates']) if c['family']=='anchor']
            assert 0 in A
            L=A+[i for i,c in enumerate(root['candidates']) if c['family'] in {'risk','cover','random'} and c['removals']<=(2 if c['family']=='random' else 4)]
            anchor={name:next((i for i in A if root['candidates'][i]['anchor']==name),0) for name in ['ZERO','BAL']}
            R={};M={};delta={};absolute={}
            raw_tapes=[json.loads((source/folder/f'RAW_{stage}_{scenario}_{t}.private.json').read_text()) for t in [31001,31002]]
            for role in ['FIT_large','AUDIT_large']:
                for tape_idx,tape in enumerate(root['tapes']):
                    baseline=score(tape[role][0]['endpoints'],T)
                    for i in range(n):
                        values=score(tape[role][i]['endpoints'],T)
                        ds=[(a[0]-b[0],None if a[1] is None else a[1]-b[1]) for a,b in zip(values,baseline)]
                        R[role,tape_idx,i],M[role,tape_idx,i]=utility(ds);delta[role,tape_idx,i]=ds;absolute[role,tape_idx,i]=values
                        old=tape[role][i]
                        for key,val in [('R',R[role,tape_idx,i]),('macro',M[role,tape_idx,i])]:max_error=max(max_error,abs(val-old['H3'][key]));verified_components+=1
                        for h,(ba,tail) in enumerate(values):
                            orig=old['endpoints'][h];max_error=max(max_error,abs(ds[h][0]-orig['macro']))
                            if ds[h][1] is not None:max_error=max(max_error,abs(ds[h][1]-orig['tail']))
                            den=orig['denominator'];covered={c for c in T if c<len(den) and den[c]>0}
                            full=bool(realT) and len(covered)==len(realT)
                            ztail=baseline[h][1]
                            raw=raw_tapes[tape_idx];allowed=set(raw['query_subsets'][str(h)][role]);ee=raw['runs'][i]['endpoints'][h];zz=raw['runs'][0]['endpoints'][h]
                            oldmask=[pid in allowed and raw['query_labels'][h][j] in T for j,pid in enumerate(ee['ids'])]
                            rescued=sum(mask and a and not b for mask,a,b in zip(oldmask,ee['good'],zz['good']));harmed=sum(mask and b and not a for mask,a,b in zip(oldmask,ee['good'],zz['good']))
                            floor='unavailable' if tail is None else 'both_exact_zero' if tail==ztail==0 else ('preserved_correct_predictions_nonzero' if rescued==harmed==0 else 'offsetting_prediction_changes') if abs(tail-ztail)<1e-12 else 'changed'
                            tails.append({**base,'role':role,'tape':[31001,31002][tape_idx],'horizon':h+1,'candidate':i,'family':root['candidates'][i]['family'],'anchor':root['candidates'][i]['anchor'],
                                          'ZERO_observed_oldtail_BA':ztail,'candidate_observed_oldtail_BA':tail,'observed_oldtail_delta':ds[h][1],
                                          'full_real_T_evaluable':full,'ZERO_full_oldtail_BA':ztail if full else None,'candidate_full_oldtail_BA':tail if full else None,
                                          'observed_tail_class_count':len(covered),'required_real_T_count':len(realT),'oldtail_rescued_components':rescued if tail is not None else None,'oldtail_harmed_components':harmed if tail is not None else None,'floor_status':floor,'verified_patient_denominator':'NA'})
            assert max_error<1e-12
            avg=lambda role,i:mean([R[role,t,i] for t in [0,1]])
            avgmacro=lambda role,i:mean([M[role,t,i] for t in [0,1]])
            # Use sealed FIT scalar order for exact floating ties after verifying it.
            fit=lambda i:mean([t['FIT_large'][i]['H3']['R'] for t in root['tapes']])
            best=lambda indices:max(indices,key=lambda i:(fit(i),-i))
            fullwinner=best(range(n));anchorwinner=best(A)
            full_coverage={};partial={};fixed_intersections={}
            shadow_full={}
            for role in ['FIT_large','AUDIT_large']:
                covered_h=[]
                for h in range(3):
                    den=root['tapes'][0][role][0]['endpoints'][h]['denominator']
                    E={order[c] for c,count in enumerate(den) if count>0}
                    covered_h.append(realT&E)
                full_coverage[role]=bool(realT) and all(x==realT for x in covered_h)
                shadowT={order[c] for c in T};shadow_full[role]=bool(T) and all(shadowT<=x for x in covered_h)
                partial[role]=[bool(x) for x in covered_h]
                fixed_intersections[role]=set.intersection(*covered_h)
            for h in range(3):
                fitden=root['tapes'][0]['FIT_large'][0]['endpoints'][h]['denominator'];auden=root['tapes'][0]['AUDIT_large'][0]['endpoints'][h]['denominator']
                E={order[c] for c in range(len(fitden)) if fitden[c]>0 and auden[c]>0}
                coverage.append({**base,'horizon':h+1,'real_seen_classes':list(range(meta['real_seen_classes'])),'shadow_supported_classes':order,'common_E':sorted(E),
                                 'real_oldtail_T':sorted(realT),'original_shadow_T':[order[c] for c in sorted(T)],'T_intersect_E':sorted(realT&E),
                                 'fixed_oldtail_intersection':sorted(set.intersection(fixed_intersections['FIT_large'],fixed_intersections['AUDIT_large'])),
                                 'missing_required_classes':sorted(realT-E),'action_count':n,'has_action':n>1,'memory_pressure':meta['pressure'],
                                 'FIT_any_tail_evaluable':partial['FIT_large'][h],'AUDIT_any_tail_evaluable':partial['AUDIT_large'][h],
                                 'FIT_full_real_T':full_coverage['FIT_large'],'AUDIT_full_real_T':full_coverage['AUDIT_large'],
                                 'FIT_full_original_shadow_T':shadow_full['FIT_large'],'AUDIT_full_original_shadow_T':shadow_full['AUDIT_large'],
                                 'action_and_original_partial_risk':n>1 and all(partial['FIT_large']) and all(partial['AUDIT_large']),
                                 'action_and_full_real_risk':n>1 and full_coverage['FIT_large'] and full_coverage['AUDIT_large'],'verified_patient_denominator':'NA'})
                for c in range(meta['real_seen_classes']):
                    shadow=mapping.get(c);seen=shadow is not None and shadow<prefix+h
                    nf=fitden[shadow] if seen else 0;na=auden[shadow] if seen else 0
                    reason='not_a_required_oldtail'
                    if c in realT:
                        reason='no_legal_support' if shadow is None else 'FIT_and_AUDIT_missing' if nf==na==0 else 'FIT_missing' if nf==0 else 'AUDIT_missing' if na==0 else 'complete_singleton_query' if min(nf,na)==1 else 'complete'
                    classes.append({**base,'horizon':h+1,'real_class':c,'original_label':cfg['order'][c],'shadow_class':shadow,'seen_in_shadow':seen,'required_oldtail':c in realT,
                                    'FIT_image_component_denominator':nf,'AUDIT_image_component_denominator':na,'verified_patient_denominator':'NA','coverage_reason':reason})
            qualifies={}
            for role in ['FIT_large','AUDIT_large']:
                for i in range(n):
                    qualifies[role,i]=bool(realT) and full_coverage[role] and all(d is not None and d>=-.005 for t in [0,1] for _,d in delta[role,t,i])
            def rule(pool,gate):
                if not gate:return best(pool),'original_argmax'
                if not realT:allowed=pool;reason='risk_not_applicable'
                elif not full_coverage['FIT_large']:return 0,'unknown_risk'
                else:allowed=[i for i in pool if i==0 or qualifies['FIT_large',i]];reason='FIT_risk_checked'
                choice=best(allowed)
                return (choice,reason) if choice and fit(choice)>0 else (0,'no_positive_qualified_FIT')
            choices={'ZERO':(0,'fixed'),'BAL':(anchor['BAL'],'fixed'),'FULL_ORIGINAL':(fullwinner,'historical_full_pool')}
            for name,pool in [('A',A),('L',L)]:
                choices[name+'0']=rule(pool,False);choices[name+'1']=rule(pool,True)
            for name,(choice,reason) in choices.items():
                observed=[d for t in [0,1] for _,d in delta['AUDIT_large',t,choice] if d is not None]
                fulltail=mean(observed) if full_coverage['AUDIT_large'] else None
                false_safe=None
                if choice!=0 and realT and qualifies['FIT_large',choice] and full_coverage['AUDIT_large']:
                    false_safe=not qualifies['AUDIT_large',choice]
                pool=A if name.startswith('A') else L if name.startswith('L') else list(range(n))
                row={**base,'rule':name,'selected_candidate':choice,'family':root['candidates'][choice]['family'],'anchor':root['candidates'][choice]['anchor'],
                     'R':avg('AUDIT_large',choice),'macro_delta':avgmacro('AUDIT_large',choice),'FIT_R':fit(choice),'nonzero':choice!=0,'reason':reason,
                     'real_oldtail_present':bool(realT),'FIT_full_risk':full_coverage['FIT_large'],'AUDIT_full_risk':full_coverage['AUDIT_large'],
                     'full_oldtail_mean_delta':fulltail,'original_observed_oldtail_mean_delta':mean(observed),'worst_observed_tail_delta':min(observed) if observed else None,
                     'original_observed_violation_checks':sum(x<-.005 for x in observed),'original_observed_checks':len(observed),
                     'false_safe_fully_evaluable':false_safe,'has_action':n>1,'pool_actions':len(pool),'candidate_feedback_branches_used':0 if name in {'ZERO','BAL'} else 2*len(pool),'shared_new_classifier_updates':0,
                     'full_risk_action_nonzero':choice!=0 and n>1 and full_coverage['FIT_large'] and full_coverage['AUDIT_large']}
                selections.append(row)
            auditfeasible=[i for i in range(n) if qualifies['AUDIT_large',i]]
            anchorfeasible=[i for i in A if qualifies['AUDIT_large',i]]
            shadowfeasible=[i for i in range(n) if shadow_full['AUDIT_large'] and all(d is not None and d>=-.005 for t in [0,1] for _,d in delta['AUDIT_large',t,i])]
            best_audit=lambda ids:max((avg('AUDIT_large',i) for i in ids),default=None)
            opportunity.append({**base,'risk_status':'not_applicable' if not realT else 'complete' if full_coverage['AUDIT_large'] else 'unknown',
                                'all_pool_AUDIT_oracle':best_audit(range(n)),'anchor_AUDIT_oracle':best_audit(A),'full_risk_feasible_candidate_count':len(auditfeasible) if full_coverage['AUDIT_large'] else None,
                                'full_risk_feasible_nonzero_count':sum(i!=0 for i in auditfeasible) if full_coverage['AUDIT_large'] else None,'full_risk_feasible_oracle':best_audit(auditfeasible),
                                'anchor_full_risk_feasible_oracle':best_audit(anchorfeasible),'FIT_winner_AUDIT_R':avg('AUDIT_large',fullwinner),
                                'original_shadow_T_complete':shadow_full['AUDIT_large'],'original_shadow_T_feasible_count':len(shadowfeasible) if shadow_full['AUDIT_large'] else None,
                                'original_shadow_T_feasible_oracle':best_audit(shadowfeasible),'original_shadow_T_feasible_nonzero':sum(i!=0 for i in shadowfeasible) if shadow_full['AUDIT_large'] else None,
                                'anchor_FIT_winner_AUDIT_R':avg('AUDIT_large',anchorwinner),'original_selection_regret':best_audit(range(n))-avg('AUDIT_large',fullwinner),
                                'feasible_selection_regret':None if not auditfeasible else best_audit(auditfeasible)-avg('AUDIT_large',fullwinner)})
            penalty=avgmacro('AUDIT_large',fullwinner)-avg('AUDIT_large',fullwinner)
            rootmeta.append({**base,'has_action':n>1,'action_count':n,'original_partial_risk':all(partial['AUDIT_large']),'full_risk_both':full_coverage['FIT_large'] and full_coverage['AUDIT_large'],
                             'real_oldtail_present':bool(realT),'penalty':penalty,'R':avg('AUDIT_large',fullwinner),'regret':opportunity[-1]['original_selection_regret']})
            if folder=='a' and stage==3 and scenario in [0,3]:
                ownanchor=anchor[root['candidates'][fullwinner]['anchor']];raw=json.loads((source/folder/f'RAW_3_{scenario}_31001.private.json').read_text())
                keys=[set(x['memory_keys']) for x in raw['runs']]
                for reference,name in [(0,'ZERO'),(ownanchor,'OWN_ANCHOR')]:
                    removed=keys[reference]-keys[fullwinner];added=keys[fullwinner]-keys[reference]
                    for c in range(meta['real_seen_classes']):
                        for local in [False,True]:
                            count=lambda values:sum(label[k//8]==c and (k%8>0)==local for k in values)
                            token_changes.append({**base,'reference':name,'real_class':c,'original_label':cfg['order'][c],'required_oldtail':c in realT,'kind':'local' if local else 'global',
                                                  'reference_count':count(keys[reference]),'candidate_count':count(keys[fullwinner]),'removed':count(removed),'added':count(added)})
                for ti,tape in enumerate(root['tapes']):
                    raw=json.loads((source/folder/f'RAW_3_{scenario}_{[31001,31002][ti]}.private.json').read_text())
                    for role in ['FIT_large','AUDIT_large']:
                        for h in range(3):
                            W=tape[role][fullwinner]['endpoints'][h];Z=tape[role][0]['endpoints'][h];B=tape[role][ownanchor]['endpoints'][h]
                            for c in realT:
                                j=mapping.get(c);den=W['denominator'][j] if j is not None else 0
                                risks.append({**base,'tape':[31001,31002][ti],'horizon':h+1,'role':role,'candidate':fullwinner,'family':root['candidates'][fullwinner]['family'],
                                              'anchor':root['candidates'][fullwinner]['anchor'],'real_class':c,'original_label':cfg['order'][c],
                                              'count':den,'ZERO_correct':Z['correct'][j] if den else None,'anchor_correct':B['correct'][j] if den else None,'candidate_correct':W['correct'][j] if den else None,
                                              'ZERO_BA':Z['correct'][j]/den if den else None,'anchor_BA':B['correct'][j]/den if den else None,'candidate_BA':W['correct'][j]/den if den else None})
                            ids=raw['runs'][fullwinner]['endpoints'][h]['ids'];yl=raw['query_labels'][h];inside=set(raw['query_subsets'][str(h)][role]);ga=raw['runs'][fullwinner]['endpoints'][h]['good'];gz=raw['runs'][0]['endpoints'][h]['good'];gb=raw['runs'][ownanchor]['endpoints'][h]['good']
                            for pos,pid in enumerate(ids):
                                if pid in inside and yl[pos] in T and (ga[pos]!=gz[pos] or ga[pos]!=gb[pos]):
                                    components.append({**base,'tape':[31001,31002][ti],'horizon':h+1,'role':role,'private_parent':pid,'original_label':cfg['order'][order[yl[pos]]],
                                                       'ZERO_good':gz[pos],'anchor_good':gb[pos],'candidate_good':ga[pos]})
                stream_risks.append({**base,'selected':root['candidates'][fullwinner],'macro':avgmacro('AUDIT_large',fullwinner),'R':avg('AUDIT_large',fullwinner),'penalty':penalty,
                                     'FIT_tail_deltas':[[d for _,d in delta['FIT_large',t,fullwinner]] for t in [0,1]],
                                     'AUDIT_tail_deltas':[[d for _,d in delta['AUDIT_large',t,fullwinner]] for t in [0,1]],
                                     'AUDIT_anchor_tail_deltas':[[d for _,d in delta['AUDIT_large',t,ownanchor]] for t in [0,1]]})
            stream_roots.append(rootmeta[-1])
        rules={}
        for name in ['ZERO','BAL','FULL_ORIGINAL','A0','A1','L0','L1']:
            rows=[x for x in selections if x['stream']==stream and x['rule']==name];fs=[x['false_safe_fully_evaluable'] for x in rows if x['false_safe_fully_evaluable'] is not None]
            rules[name]={'mean_R':mean([x['R'] for x in rows]),'mean_macro':mean([x['macro_delta'] for x in rows]),'mean_original_observed_tail':mean([x['original_observed_oldtail_mean_delta'] for x in rows]),
                         'mean_full_tail':mean([x['full_oldtail_mean_delta'] for x in rows]),'nonzero_roots':sum(x['nonzero'] for x in rows),'unknown_fallback_roots':sum(x['reason']=='unknown_risk' for x in rows),
                         'full_risk_action_nonzero_roots':sum(x['full_risk_action_nonzero'] for x in rows),'observed_subset_violation_checks':sum(x['original_observed_violation_checks'] for x in rows),
                         'observed_subset_checks':sum(x['original_observed_checks'] for x in rows),'false_safe_numerator':sum(fs),'false_safe_denominator':len(fs),'false_safe_rate':sum(fs)/len(fs) if fs else None,
                         'logical_feedback_branches_used':sum(x['candidate_feedback_branches_used'] for x in rows),'new_physical_branches':0}
        stage_rows={str(t):{name:{'mean_R':mean([x['R'] for x in selections if x['stream']==stream and x['stage']==t and x['rule']==name]),
                                       'nonzero':sum(x['nonzero'] for x in selections if x['stream']==stream and x['stage']==t and x['rule']==name)} for name in rules} for t in [0,2,3,4]}
        stratified={}
        for category in ['no_oldtail','action_full_risk','action_partial_only','action_no_observed_tail','no_action_with_tail']:
            pred=lambda x: not x['real_oldtail_present'] if category=='no_oldtail' else x['real_oldtail_present'] and x['has_action'] and x['full_risk_both'] if category=='action_full_risk' else x['real_oldtail_present'] and x['has_action'] and x['original_partial_risk'] and not x['full_risk_both'] if category=='action_partial_only' else x['real_oldtail_present'] and x['has_action'] and not x['original_partial_risk'] if category=='action_no_observed_tail' else not x['has_action'] and x['real_oldtail_present']
            rr=[x for x in stream_roots if pred(x)];stratified[category]={'count':len(rr),'original_mean_R':mean([x['R'] for x in rr])}
        risk_ids={(3,0),(3,3)};risk_regret=sum(x['regret'] for x in stream_roots if (x['stage'],x['scenario']) in risk_ids)
        total_regret=sum(x['regret'] for x in stream_roots)
        streams[stream]={'rules':rules,'stages':stage_rows,'coverage':{'original_tail_evaluable_roots':sum(x['original_partial_risk'] for x in stream_roots),
                       'action_and_original_partial_risk':sum(x['has_action'] and x['original_partial_risk'] for x in stream_roots),
                       'action_and_full_real_risk':sum(x['has_action'] and x['full_risk_both'] for x in stream_roots),'strata':stratified},
                       'A2_risk_roots':stream_risks,'all_root_penalty':sum(x['penalty'] for x in stream_roots)/16,
                       'other_penalties_nonzero':sum(abs(x['penalty'])>1e-12 for x in stream_roots if (x['stage'],x['scenario']) not in risk_ids),
                       'total_selection_regret':total_regret,'two_ordered_risk_root_regret_fraction':risk_regret/total_regret if folder=='a' and total_regret else None}
    assert max_error<1e-12
    for name,rows in [('A1_RISK_COVERAGE.csv',coverage),('A1_CLASS_COVERAGE.csv',classes),('A1_ABSOLUTE_OLDTAIL.csv',tails),('A2_RISK_TIMELINE.private.csv',risks),('A2_TOKEN_CHANGES.csv',token_changes),('A3_OPPORTUNITY_REGRET.csv',opportunity),('A4_RULE_SELECTION.csv',selections)]:csvout(output/name,rows)
    csvout(output/'A4_STREAM_COMPARISON.csv',[{'stream':s,'rule':name,**values} for s,v in streams.items() for name,values in v['rules'].items()])
    csvout(output/'A4_STAGE_COMPARISON.csv',[{'stream':s,'stage':stage,'rule':name,**values} for s,v in streams.items() for stage,rules in v['stages'].items() for name,values in rules.items()])
    increments=[]
    for s,v in streams.items():
        for gate in ['0','1']:
            a=v['rules']['A'+gate];l=v['rules']['L'+gate]
            increments.append({'stream':s,'gate':gate=='1','L_minus_A_R':l['mean_R']-a['mean_R'],'L_minus_A_macro':l['mean_macro']-a['mean_macro'],
                               'A_nonzero':a['nonzero_roots'],'L_nonzero':l['nonzero_roots'],'roots':16,'independent_patients':'NA'})
    csvout(output/'A4_LOCAL_INCREMENT.csv',increments)
    write(output/'A2_COMPONENT_EVENTS.private.json',components);write(output/'A_SUMMARY.json',streams)
    verification={'passed':True,'roots':32,'recomputed_R_components':verified_components,'max_float_error':max_error,'forward':0,'optimizer':0,'patient_denominators':'NA',
                  'source_verification_receipt':'R110 VERIFICATION.json; archived integer counts and original R reproduced, not rewritten','original_FULL_means':{s:v['rules']['FULL_ORIGINAL']['mean_R'] for s,v in streams.items()},
                  'protected_classes':'real-seen old-tail set; missing shadow support/query is unknown, not removed from protection','A2_reference_clarification':'compare both ZERO and own anchor; scenario3 risk is already present in BAL, not isolated to the four-token edit'}
    write(output/'A_VERIFICATION.json',verification);write(output/'A_COSTS.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'forward':0,'optimizer':0,'new_physical_candidate_branches':0,'source_branches_reused':886})
    print(json.dumps({'verification':verification,'summary':streams},ensure_ascii=False,indent=2))

if __name__=='__main__':run(os.environ['Q_SAVED'],os.environ['Q_METADATA'],os.environ['Q_OUTPUT'])

