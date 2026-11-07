"""Independent saved-only integer/reward/legality/influence audit. No torch."""
import csv,hashlib,json,os,random
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np


def reward(endpoints):
    macro=sum(e['macro'] for e in endpoints)/len(endpoints)
    available=[e['tail'] for e in endpoints if e['tail'] is not None]
    penalty=max([max(0.,-x-.005) for x in available],default=0.)
    return dict(R=macro-2*penalty,macro=macro,tail_coverage=len(available)/len(endpoints))


def class_counts(labels,good,C):
    return np.bincount(labels[good],minlength=C),np.bincount(labels,minlength=C)


def endpoint(labels,good,zero,C,oldtail):
    ca,n=class_counts(labels,good,C);cz,_=class_counts(labels,zero,C)
    present=n>0;delta=np.zeros(C);delta[present]=(ca[present]-cz[present])/n[present]
    tail=present & np.array([c in oldtail for c in range(C)])
    return dict(macro=float(delta[present].mean()),tail=float(delta[tail].mean()) if tail.any() else None,
                correct=ca.tolist(),denominator=n.tolist(),rescued=int((good&~zero).sum()),harmed=int((~good&zero).sum()))


def profile(keys,anchor,real_label,shadow_map):
    a=set(anchor);counts=Counter()
    for key in keys:
        label=shadow_map[int(real_label[key//8])]
        counts[(label,key%8>0,'count')]+=1
        counts[(label,key%8>0,'overlap')]+=int(key in a)
    return dict(counts)


def influence(raws,choice,oldtail,meta):
    contexts=[];contributions=defaultdict(float);allids=set();inestimable=set()
    for raw in raws:
        stages=[]
        for h in range(3):
            A=raw['runs'][choice]['endpoints'][h];Z=raw['runs'][0]['endpoints'][h]
            ids=np.array(A['ids']);y=np.array(raw['query_labels'][h]);keep=set(raw['query_subsets'][str(h)]['AUDIT_large'])
            mask=np.array([int(i) in keep for i in ids]);ids=ids[mask];y=y[mask]
            ga=np.array(A['good'],dtype=bool)[mask];gz=np.array(Z['good'],dtype=bool)[mask];C=meta['prefix_classes']+h
            ca,n=class_counts(y,ga,C);cz,_=class_counts(y,gz,C);present=n>0;E=int(present.sum());T=sum(present[c] for c in oldtail if c<C)
            ep=endpoint(y,ga,gz,C,oldtail);lookup={int(pid):(int(c),int(a)-int(z)) for pid,c,a,z in zip(ids,y,ga,gz)}
            for pid,(c,diff) in lookup.items():
                allids.add(pid);contributions[pid]+=diff/(2*3*E*n[c])
                if n[c]==1:inestimable.add(pid)
            stages.append((ep,lookup,ca-cz,n,E,T))
        contexts.append(stages)
    baseline=sum(reward([v[0] for v in stages])['R'] for stages in contexts)/2
    loo={}
    for pid in allids-inestimable:
        values=[]
        for stages in contexts:
            changed=[]
            for ep,lookup,diff,n,E,T in stages:
                copy=dict(ep)
                if pid in lookup:
                    c,d=lookup[pid];adjust=(diff[c]-d)/(n[c]-1)-diff[c]/n[c]
                    copy['macro']+=adjust/E
                    if c in oldtail and T:copy['tail']+=adjust/T
                changed.append(copy)
            values.append(reward(changed)['R'])
        loo[pid]=sum(values)/2
    return contributions,loo,inestimable,dict(
        max_single_component_macro_contribution=max(contributions.values(),default=0.),
        max_defined_single_component_R_drop=max((baseline-v for v in loo.values()),default=0.),
        positive_R_sign_flip_on_defined_deletion=baseline>0 and any(v<=0 for v in loo.values()),
        deletion_inestimable_components=len(inestimable),deletion_evaluable_components=len(loo))


def run(saved,metadata,output):
    saved=Path(saved);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(Path(metadata).read_text());labels=np.array(metadata['original_labels']);components=metadata['components']
    assert len(labels)==len(set(components))==6821
    roles={}
    for label in sorted(set(labels)):
        units=sorted([components[i] for i in np.where(labels==label)[0]]);random.Random(947110+int(label)).shuffle(units)
        q=max(1,int(.2*len(units))) if len(units)>=3 else 0
        for j,u in enumerate(units):roles[u]='FIT' if j<q else 'AUDIT' if j<2*q else 'S'
    p_roles={i:roles[g] for i,g in enumerate(components)}
    max_error=0.;cases=0;class_rows=[];root_rows=[];matched_rows=[];streams={};all_comparisons=0
    for folder in ['a','b']:
        cfg=json.loads((saved/folder/'CONFIG.private.json').read_text());real_label=np.array([cfg['order'].index(int(l)) for l in labels])
        saved_result=json.loads((saved/folder/'RESULTS.json').read_text());summary_rows=[];macro_by_pid=defaultdict(float);R_drop_by_pid=defaultdict(float);inestimable_global=set();candidate_sum=0
        old_donors=metadata['donor_keys'][folder]
        for root in saved_result['roots']:
            assert 'skipped' not in root
            meta=root['metadata'];stage=root['stage'];seed=75110+stage*100+root['scenario'];assert seed==meta['seed']
            previous=set(old_donors[str(stage-1)]) if stage else set()
            current=set(np.where(np.isin(real_label,cfg['stages'][stage]))[0].tolist())
            legal_parents={k//8 for k in previous}|current
            supported=sorted({int(real_label[i]) for i in legal_parents if p_roles[i]=='S'})
            future=[c for c in cfg['stages'][stage] if c in supported];random.Random(seed).shuffle(future);future=future[:2]
            prefix=sorted(set(supported)-set(future));order=prefix+future;mapping={c:j for j,c in enumerate(order)}
            assert len(prefix)==meta['prefix_classes'] and len(order)==meta['supported_shadow_classes']
            oldtail={mapping[c] for c in prefix if c<sum(map(len,cfg['stages'][:stage])) and cfg['train_counts'][c]<250}
            assert len(oldtail)==meta['oldtail_classes']
            eligible=set.intersection(*[{int(real_label[i]) for i in legal_parents if p_roles[i]==role} for role in ['S','FIT','AUDIT']])
            files=[saved/folder/f"RAW_{stage}_{root['scenario']}_{t}.private.json" for t in [31001,31002]]
            raws=[json.loads(p.read_text()) for p in files];n=len(root['candidates']);candidate_sum+=n
            anchor_indices={a:next((i for i,c in enumerate(root['candidates']) if c['family']=='anchor' and c['anchor']==a),0) for a in ['ZERO','BAL']}
            keys=[r['memory_keys'] for r in raws[0]['runs']]
            for i,candidate in enumerate(root['candidates']):
                k=keys[i];assert len(k)==len(set(k))==min(256,meta['root_tokens'])
                assert hashlib.sha256(','.join(map(str,k)).encode()).hexdigest()==raws[0]['runs'][i]['candidate_key']
                assert k==raws[1]['runs'][i]['memory_keys']
                for token in k:
                    pid,view=divmod(token,8);assert view<6 and pid in legal_parents and p_roles[pid]=='S'
                    assert pid in current or token in previous
                    if view:assert pid*8 in k
                assert {mapping[int(real_label[t//8])] for t in k if t%8==0}==set(range(len(prefix)))
                anchor=keys[anchor_indices[candidate['anchor']]]
                actual=profile(k,anchor,real_label,mapping);baseline=profile(anchor,anchor,real_label,mapping)
                for key,value in baseline.items():
                    if key[2]=='count':assert actual.get(key,0)==value
                assert len(set(anchor)-set(k))==candidate['removals']
                if candidate['family'] in ['risk','cover']:assert candidate['removals']<=4
                if candidate['family']=='random':assert candidate['removals']<=2
                cases+=1
            assert len(set(r['candidate_key'] for r in raws[0]['runs']))==n
            for match in root['matched_reconstructions']:
                if match['matched']:
                    anchor=keys[anchor_indices[match['anchor']]]
                    assert profile(keys[match['structural_index']],anchor,real_label,mapping)==profile(keys[match['random_index']],anchor,real_label,mapping)
            recomputed=[]
            for ti,raw in enumerate(raws):
                assert len(raw['runs'])==n
                counters={name:[] for name in ['FIT_small','FIT_large','AUDIT_small','AUDIT_large']}
                for h in range(3):
                    C=len(prefix)+h;E={c for c in eligible if mapping.get(c,1000)<C}
                    for role in ['FIT','AUDIT']:
                        expected=sorted(i for i in legal_parents if p_roles[i]==role and int(real_label[i]) in E)
                        assert set(expected)==set(raw['query_subsets'][str(h)][role+'_large'])
                        small=[]
                        for c in sorted(mapping[x] for x in E):
                            ids=[i for i in expected if mapping[int(real_label[i])]==c];random.Random(seed+17*c).shuffle(ids);small.extend(ids[:2])
                        assert set(small)==set(raw['query_subsets'][str(h)][role+'_small'])
                for a,run in enumerate(raw['runs']):
                    for name in counters:
                        eps=[]
                        for h in range(3):
                            A=run['endpoints'][h];Z=raw['runs'][0]['endpoints'][h];assert A['ids']==Z['ids']
                            ids=np.array(A['ids']);yy=np.array(raw['query_labels'][h]);assert yy.tolist()==[mapping[int(real_label[i])] for i in ids]
                            keep=set(raw['query_subsets'][str(h)][name]);mask=np.array([int(i) in keep for i in ids])
                            ga=np.array(A['good'],dtype=bool)[mask];gz=np.array(Z['good'],dtype=bool)[mask];y=yy[mask];C=len(prefix)+h
                            e=endpoint(y,ga,gz,C,oldtail);archived=root['tapes'][ti][name][a]['endpoints'][h]
                            for key in ['correct','denominator','rescued','harmed']:assert e[key]==archived[key]
                            for key in ['macro','tail']:
                                assert (e[key] is None)==(archived[key] is None)
                                if e[key] is not None:max_error=max(max_error,abs(e[key]-archived[key]))
                            for c,(correct,den) in enumerate(zip(e['correct'],e['denominator'])):
                                if den:class_rows.append(dict(stream=saved_result['stream'],stage=stage,scenario=root['scenario'],tape=[31001,31002][ti],candidate=a,family=root['candidates'][a]['family'],anchor=root['candidates'][a]['anchor'],horizon=h+1,query=name,shadow_class=c,original_label=cfg['order'][order[c]],oldtail=c in oldtail,correct=correct,count=den))
                            eps.append(e)
                        values={name:reward(ends) for name,ends in [('H1',eps[:1]),('H3',eps)]}
                        for H,value in values.items():
                            for key in ['R','macro','tail_coverage']:max_error=max(max_error,abs(value[key]-root['tapes'][ti][name][a][H][key]));all_comparisons+=1
                        counters[name].append(values)
                recomputed.append(counters)
            assert max_error<1e-12
            # Preserve the official float/tie decisions; independently recompute their outcomes.
            avg=lambda q,a,H,key='R':sum(t[q][a][H][key] for t in recomputed)/2
            archived_avg=lambda q,a:sum(t[q][a]['H3']['R'] for t in root['tapes'])/2
            pick=max(range(n),key=lambda a:(archived_avg('FIT_large',a),-a));small=max(range(n),key=lambda a:(archived_avg('FIT_small',a),-a))
            anchor_pick=max(set(anchor_indices.values()),key=lambda a:(archived_avg('FIT_large',a),-a))
            row=dict(stream=saved_result['stream'],stage=stage,scenario=root['scenario'],candidate_count=n,root_tokens=meta['root_tokens'],pressure=meta['pressure'],
                     selected=pick,selected_family=root['candidates'][pick]['family'],selected_anchor=root['candidates'][pick]['anchor'],
                     large_FIT_large_AUDIT_R=avg('AUDIT_large',pick,'H3'),large_FIT_large_AUDIT_macro=avg('AUDIT_large',pick,'H3','macro'),
                     small_FIT_large_AUDIT_R=avg('AUDIT_large',small,'H3'),anchor_only_FIT_large_AUDIT_R=avg('AUDIT_large',anchor_pick,'H3'),
                     AUDIT_self_oracle_R=max(avg('AUDIT_large',a,'H3') for a in range(n)))
            tail=[v['tail'] for t in root['tapes'] for v in t['AUDIT_large'][pick]['endpoints'] if v['tail'] is not None]
            row['oldtail_evaluable']=len(tail)==6;row['oldtail_mean_delta']=sum(tail)/len(tail) if tail else None
            contribution,loo,inestimable,inf=influence(raws,pick,oldtail,meta);row.update(inf)
            for pid,value in contribution.items():macro_by_pid[pid]+=value/16
            for pid,value in loo.items():R_drop_by_pid[pid]+=(row['large_FIT_large_AUDIT_R']-value)/16
            inestimable_global.update(inestimable)
            for match in root['matched_reconstructions']:
                if match['matched']:
                    matched_rows.append(dict(stream=saved_result['stream'],stage=stage,scenario=root['scenario'],anchor=match['anchor'],distinct=match['distinct'],structural_minus_matched_random_R=avg('AUDIT_large',match['structural_index'],'H3')-avg('AUDIT_large',match['random_index'],'H3')))
            summary_rows.append(row);root_rows.append(row)
        mean=lambda key:sum(r[key] for r in summary_rows)/len(summary_rows)
        risk=[r['oldtail_mean_delta'] for r in summary_rows if r['oldtail_evaluable']]
        stream=dict(mean_R=mean('large_FIT_large_AUDIT_R'),mean_macro=mean('large_FIT_large_AUDIT_macro'),small_FIT_mean_R=mean('small_FIT_large_AUDIT_R'),
                    anchor_only_descriptive_mean_R=mean('anchor_only_FIT_large_AUDIT_R'),AUDIT_oracle_descriptive_mean_R=mean('AUDIT_self_oracle_R'),
                    positive_roots=sum(r['large_FIT_large_AUDIT_R']>0 for r in summary_rows),oldtail_roots=len(risk),mean_oldtail=sum(risk)/len(risk) if risk else None,
                    no_op_only_roots=sum(r['candidate_count']==1 for r in summary_rows),candidate_sets=candidate_sum,
                    max_component_net_macro_contribution=max(macro_by_pid.values(),default=0.),
                    max_defined_component_mean_R_drop=max([v for p,v in R_drop_by_pid.items() if p not in inestimable_global],default=0.),
                    singleton_deletion_inestimable_components=len(inestimable_global),
                    selected_families=dict(Counter(r['selected_family'] for r in summary_rows)),stages={})
        for stage in [0,2,3,4]:
            rows=[r for r in summary_rows if r['stage']==stage]
            stream['stages'][str(stage)]={key:sum(r[key] for r in rows)/4 for key in ['large_FIT_large_AUDIT_R','large_FIT_large_AUDIT_macro','small_FIT_large_AUDIT_R']}
        matched=[r['structural_minus_matched_random_R'] for r in matched_rows if r['stream']==saved_result['stream'] and r['distinct']]
        stream['structural_minus_matched_random_mean_R']=sum(matched)/len(matched) if matched else None;stream['distinct_matched_pairs']=len(matched)
        expected=2*700*candidate_sum+sum(r['metadata']['prefix_phases']*300-100 for r in saved_result['roots'])+1500+16
        assert expected==saved_result['science_classifier_updates']
        stream['science_classifier_updates']=expected;streams[saved_result['stream']]=stream
    verification=dict(passed=True,roots=len(root_rows),candidate_sets=cases,real_branch_executions=2*cases,integer_counts_exact=True,
                      reward_comparisons=all_comparisons,max_float_error=max_error,forward_or_optimizer_updates=0,
                      candidate_parent_budget_quota_overlap_checked=True,immutable_roles_and_nested_queries_checked=True,
                      not_reverified=['centroid geometry requires discarded feature tensors','internal prefix eviction history not fully reconstructible from saved root keys','old query local-view identity is runtime guarded but not archived in raw query keys','clinical patient independence unverified'])
    for name,value in [('VERIFICATION.json',verification),('SAVED_ANALYSIS.json',streams)]:
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    for name,rows in [('ROOT_ANALYSIS.csv',root_rows),('CLASS_QUERY_COUNTS.csv',class_rows),('MATCHED_RECONSTRUCTION.csv',matched_rows)]:
        with (output/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'verification':verification,'streams':streams},indent=2))

if __name__=='__main__':run(os.environ['Q_SAVED'],os.environ['Q_METADATA'],os.environ['Q_OUTPUT'])

