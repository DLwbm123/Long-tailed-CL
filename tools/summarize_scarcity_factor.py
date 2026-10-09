"""Anonymous readout and independent audit of the locked scarcity experiment."""
import json
import math
from pathlib import Path
import statistics as st
import sys
import time

KEYS=['zero','full_plus','full_minus','row_plus','row_minus','within_plus','within_minus']
FAMILIES={name:[0,KEYS.index(name+'_plus'),KEYS.index(name+'_minus')] for name in ('full','row','within')}


def groups(r,classes):
    return dict(all=classes,old=[c for c in classes if c not in r['current']],
                current=[c for c in classes if c in r['current']],tail=[c for c in classes if c in r['tail']])


def select(r,meta,candidates):
    seen=r['seen'];current=r['current'];records={}
    for i in candidates:
        terms={}
        for g,classes in groups(r,seen).items():
            delta=[st.mean((meta[i][seen.index(c)]-meta[0][seen.index(c)]) if c in current else
                          b[i][seen.index(c)]-b[0][seen.index(c)] for c in classes) for b in r['gaussian_batches']]
            terms[g]=dict(mean_gain=st.mean(delta),integration_se=st.stdev(delta)/math.sqrt(len(delta)))
        resolution=max(1e-8,2*terms['all']['integration_se'])
        protected=all(v['mean_gain']>=-1e-12 for g,v in terms.items() if g!='all')
        records[KEYS[i]]=dict(groups=terms,numerical_resolution=resolution,protection_pass=protected,
                             eligible=protected and terms['all']['mean_gain']>resolution)
    eligible=[k for k,v in records.items() if v['eligible']]
    winner=min(eligible,key=lambda k:(-records[k]['groups']['all']['mean_gain'],KEYS.index(k))) if eligible else 'zero'
    return winner,records


def audit_private_deletions(r,identities):
    """Run beside private files on the server; return only aggregate checks."""
    seen=r['seen'];totals={};counts={};checks=0
    for c in r['current']:
        j=seen.index(c);rows=[x for x in identities if x['label']==j]
        counts[j]=sum(x['images'] for x in rows);totals[j]=[sum(x['correct'][i] for x in rows) for i in range(7)]
        assert len(rows)==r['meta_support'][str(c)]['identities']
        assert counts[j]==r['meta_support'][str(c)]['images']
        for i in range(7):assert abs(totals[j][i]/counts[j]-r['meta_accuracy'][i][j])<1e-12
    hist={f:{KEYS[i]:0 for i in ids} for f,ids in FAMILIES.items()};unsupported=0
    for row in identities:
        j=row['label'];remaining=counts[j]-row['images']
        if not remaining:unsupported+=1;continue
        meta=[v[:] for v in r['meta_accuracy']]
        for i in range(7):meta[i][j]=(totals[j][i]-row['correct'][i])/remaining
        for f,ids in FAMILIES.items():
            winner,_=select(r,meta,ids);hist[f][winner]+=1;checks+=1
    for f,counts in hist.items():
        saved=r['stability'][f]
        assert counts==saved['winner_counts'],(f,counts,saved['winner_counts'])
        assert unsupported==saved['unsupported_deletions']
        assert len(identities)==saved['identity_deletions']
        assert saved['same_winner']==counts[saved['primary_point']]
        expected=saved['primary_point'] if not unsupported and saved['same_winner']==len(identities) else 'zero'
        assert expected==saved['stable_choice']==r['selected'][f+'_stable']
    return dict(status='PASS',identity_deletions=len(identities),family_selection_checks=checks,
                private_identity_records_exported=False)


def development(r,split):
    table=r['evaluation'][split];assert set(table)==set(KEYS)
    classes=r['seen'] if split=='full' else r['covered_classes'];result={}
    zero=table['zero']['metrics']['per_class']
    for key,value in table.items():
        per=value['metrics']['per_class'];pairs=value['paired_to_strong'];assert set(per)=={str(c) for c in classes}
        delta={str(c):per[str(c)]['recall']-zero[str(c)]['recall'] for c in classes}
        for c in classes:
            p=pairs[str(c)];v=per[str(c)];assert p['n']==v['n'] and v['n']>0
            assert abs(v['recall']-v['correct']/v['n'])<1e-12
            assert abs(delta[str(c)]-(p['helped']-p['hurt'])/p['n'])<1e-12
        absolute={g:st.mean(per[str(c)]['recall'] for c in cs) for g,cs in groups(r,classes).items()}
        gain={g:st.mean(delta[str(c)] for c in cs) for g,cs in groups(r,classes).items()}
        assert abs(absolute['all']-value['metrics']['views']['all']['BA'])<1e-12
        result[key]=dict(BA=absolute,BA_gain=gain,per_class_recall_gain=delta,
            helped=sum(v['helped'] for v in pairs.values()),hurt=sum(v['hurt'] for v in pairs.values()),
            group_protection=all(v>=-1e-12 for g,v in gain.items() if g!='all'),
            per_class_protection=min(delta.values())>=-1e-12,
            per_sample_protection=all(v['hurt']==0 for v in pairs.values()))
    return result


def summarize(data,lock,repair):
    summary={};comparisons={};checks=0
    for job in lock['jobs']:
        name=job['name'];status=data['statuses'][name];expected=[s['name'] for s in job['states']]
        assert status['status']==data['suites'][name]['status']=='COMPLETE'
        assert status['prepared_states']==status['evaluated_states']==expected
        assert status['image_rows']==status['adapter_updates']==status['policy_updates']==0 and not status['test_accessed']
        assert status['maximum_solve_residual']<=lock['max_solve_residual']
        assert data['barriers'][name]['states']==expected and not data['barriers'][name]['development_metrics_evaluated']
        assert set(data['results'][name])==set(expected)
        for key,r in data['results'][name].items():
            assert r['development_evaluated_after_all_selections'] and not r['selection_used_development_labels']
            assert r['primary_method']==lock['primary_method'] and not r['independent_confirmation']
            assert data['deletion_audit']['states'][key]['status']=='PASS'
            assert max(r['existing_head_reconstruction_errors'].values())<=lock['reconstruction_atol']
            for family,ids in FAMILIES.items():
                selected,audit=select(r,r['meta_accuracy'],ids);assert selected==r['selected'][family+'_point']
                for k,v in audit.items():
                    saved=r['choice_audits'][family][k]
                    assert v['eligible']==saved['eligible'] and v['protection_pass']==saved['protection_pass']
                    for g,terms in v['groups'].items():assert all(abs(x-saved['groups'][g][t])<1e-12 for t,x in terms.items())
                    checks+=1
            for k,v in r['allocations'].items():
                assert abs(sum(v['row_mass'])-sum(v['base_row_mass']))<1e-8 and v['minimum_base_ratio']>=.5-1e-12
                if k.startswith('within'):assert max(abs(x-y) for x,y in zip(v['row_mass'],v['base_row_mass']))<1e-10
            comparisons[key]={split:development(r,split) for split in ('full','A','B')};parts={}
            for split,rows in comparisons[key].items():
                selected={name:dict(key=k,**rows[k]) for name,k in r['selected'].items()}
                primary=r['selected']['within_stable']
                versus_fixed={k:{g:rows[primary]['BA'][g]-rows[k]['BA'][g] for g in rows[k]['BA']} for k in KEYS}
                oracle={}
                for family,ids in FAMILIES.items():
                    oracle[family]={}
                    for tier in ('unconstrained','group','per_class','per_sample'):
                        eligible=[KEYS[i] for i in ids if tier=='unconstrained' or rows[KEYS[i]][tier+'_protection']]
                        best=min(eligible,key=lambda k:(-rows[k]['BA_gain']['all'],KEYS.index(k)))
                        oracle[family][tier]=dict(key=best,**rows[best])
                interaction={sign:{g:rows['full_'+sign]['BA_gain'][g]-rows['row_'+sign]['BA_gain'][g]-rows['within_'+sign]['BA_gain'][g]
                                  for g in rows['zero']['BA_gain']} for sign in ('plus','minus')}
                parts[split]=dict(selected=selected,primary_vs_fixed=versus_fixed,oracles=oracle,BA_interaction=interaction)
            for k in KEYS:
                for c in r['covered_classes']:
                    for field in ('n','correct'):
                        get=lambda split:r['evaluation'][split][k]['metrics']['per_class'][str(c)][field]
                        assert get('full')==get('A')+get('B')
                    for field in ('n','helped','hurt','prediction_changed'):
                        get=lambda split:r['evaluation'][split][k]['paired_to_strong'][str(c)][field]
                        assert get('full')==get('A')+get('B')
            summary[key]=dict(dataset=r['dataset'],task=r['task'],cohort=r['cohort'],selected=r['selected'],
                excluded_classes=r['excluded_classes'],stability=r['stability'],partitions=parts)
    assert len(summary)==5 and checks==45
    assert sum(s['head_readouts'] for s in data['statuses'].values())==lock['head_readouts']==105
    primary=[(k,r) for k,r in summary.items() if r['cohort']=='existing_full_two_epoch_static_trajectory'];assert len(primary)==3
    nonnegative=all(min(r['partitions']['full']['selected']['within_stable']['BA_gain'].values())>=-1e-12 for _,r in primary)
    positive={};beats={}
    for dataset in ('ISIC','HK'):
        parts=[r['partitions']['full']['selected'] for _,r in primary if r['dataset']==dataset]
        positive[dataset]=any(p['within_stable']['BA_gain']['all']>1e-8 for p in parts)
        beats[dataset]=any(p['within_stable']['BA_gain']['all']-p['full_stable']['BA_gain']['all']>1e-8 for p in parts)
    signal=dict(passed=nonnegative and all(positive.values()) and all(beats.values()),all_groups_nonnegative=nonnegative,
                positive_per_dataset=positive,beats_matched_full_stable=beats,independent_confirmation=False)
    costs={k:v['elapsed_seconds'] for k,v in data['suites'].items()};failed=repair['failed_total_suite_seconds']
    for k,v in costs.items():assert v+repair['failed_attempt']['suites'][k]['elapsed_seconds']<=lock['suite_wall_seconds']+1
    total=sum(costs.values())+failed
    return dict(summary=summary,comparisons=comparisons,signal=signal,
        audit=dict(status='PASS',states=5,head_readouts=105,point_selector_candidate_checks=checks,
            independent_deletion_family_checks=sum(v['family_selection_checks'] for v in data['deletion_audit']['states'].values()),
            maximum_solve_residual=max(v['maximum_solve_residual'] for v in data['statuses'].values()),
            image_rows=0,adapter_updates=0,policy_updates=0,test_accessed=False,independent_confirmation=False),
        budget=dict(prior_closed_gpu_seconds=lock['prior_closed_gpu_seconds'],failed_attempt_suite_seconds=failed,
            successful_suite_seconds=costs,round_gpu_seconds=total,cumulative_gpu_seconds=lock['prior_closed_gpu_seconds']+total,
            worker_cpu_seconds={k:v['cpu_process_seconds'] for k,v in data['statuses'].items()},
            initial_cpu_selfcheck_seconds=1.2263397190000003,repair_cpu_selfcheck_seconds=repair['repair_selfcheck']['cpu_seconds'],
            deletion_audit_cpu_seconds=data['deletion_audit']['cpu_seconds'],original_deadline=lock['original_deadline'],
            note='Failed and successful suite residence counted once; CPU listed separately. No budget reset.'))


def table(result):
    lines=['# SCARCITY-FACTOR1 全部固定头与选择结果','','所有增量单位为百分点；A/B为固定覆盖类别，full包括全部已见类。','']
    for name,r in result['summary'].items():
        lines+=['## '+name,'','A/B排除类别：'+str(r['excluded_classes']),'']
        for split,rows in result['comparisons'][name].items():
            lines+=['### '+split,'','| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |','|---|---:|---:|---:|---:|---:|---:|---:|']
            for k,v in rows.items():lines.append('| '+k+' | '+f"{v['BA']['all']*100:.4f}"+' | '+' | '.join(f'{x*100:.4f}' for x in v['BA_gain'].values())+f" | {v['helped']} | {v['hurt']} |")
            lines+=['','| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |','|---|---|---:|---:|---:|---:|']
            for k,v in r['partitions'][split]['selected'].items():lines.append('| '+k+' | '+v['key']+' | '+' | '.join(f'{x*100:.4f}' for x in v['BA_gain'].values())+' |')
        lines+=['','删除稳定性（仅当前meta身份）：',json.dumps(r['stability'],ensure_ascii=False),'']
    return '\n'.join(lines).rstrip()+'\n'


def self_check():
    r=dict(seen=[0,1],current=[1],tail=[0],gaussian_batches=[[[.5,.5] for _ in KEYS] for _ in range(8)],
           meta_accuracy=[[0.,.5] for _ in KEYS],meta_support={'1':dict(identities=2,images=2)},stability={},selected={})
    r['meta_accuracy'][5][1]=1.
    for f,ids in FAMILIES.items():
        point,_=select(r,r['meta_accuracy'],ids);assert point==('within_plus' if f=='within' else 'zero')
        hist={KEYS[i]:0 for i in ids};hist['zero']=1 if f=='within' else 2
        if f=='within':hist['within_plus']=1
        r['stability'][f]=dict(winner_counts=hist,unsupported_deletions=0,identity_deletions=2,
            same_winner=hist[point],primary_point=point,stable_choice='zero')
        r['selected'][f+'_stable']='zero'
    rows=[dict(label=1,images=1,correct=[0,0,0,0,0,1,0]),dict(label=1,images=1,correct=[1]*7)]
    assert audit_private_deletions(r,rows)['family_selection_checks']==6


if __name__=='__main__':
    start=time.process_time();self_check();config=json.load(sys.stdin)
    data=json.loads(Path(config['input']).read_text());lock=json.loads(Path(config['lock']).read_text());repair=json.loads(Path(config['repair']).read_text())
    result=summarize(data,lock,repair);result['budget']['readout_cpu_seconds']=time.process_time()-start
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    for k,v in result.items():(out/(k.upper()+'.json')).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    for suite in data['suites'].values():
        for field in ('pid','child_pid'):suite.pop(field,None)
    (out/'RESULTS.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (out/'TABLE_ZH.md').write_text(table(result));print(json.dumps(result['audit']))
