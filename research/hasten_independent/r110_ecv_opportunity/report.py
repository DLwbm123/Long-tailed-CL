"""Saved-only prospective selection and aggregate reporting; AUDIT never selects."""
from pathlib import Path
from .common import write
import json,csv

def average(values):return sum(values)/len(values) if values else None

def root_summary(root):
    if 'skipped' in root:return {'stage':root['stage'],'scenario':root['scenario'],'skipped':True,'reason':root['skipped']}
    tapes=root['tapes'];n=len(root['candidates']);out={'stage':root['stage'],'scenario':root['scenario'],'skipped':False,'candidate_count':n,
             'pressure':root['metadata']['pressure'],'prefix_classes':root['metadata']['prefix_classes'],'supported_classes':root['metadata']['supported_shadow_classes']}
    for horizon in ['H1','H3']:
        means={name:[average([t[name][a][horizon]['R'] for t in tapes]) for a in range(n)] for name in ['FIT_small','FIT_large','AUDIT_small','AUDIT_large']}
        for fit in ['small','large']:
            choice=max(range(n),key=lambda a:(means['FIT_'+fit][a],-a))
            out[horizon+'_'+fit+'_FIT_choice']=choice
            for audit in ['small','large']:out[horizon+'_'+fit+'_FIT_to_'+audit+'_AUDIT_R']=means['AUDIT_'+audit][choice]
        out[horizon+'_AUDIT_self_oracle_descriptive_only']=max(means['AUDIT_large'])
        out[horizon+'_nonreference_positive_AUDIT_candidates']=sum(r>0 for r in means['AUDIT_large'][1:])
    choice=out['H3_large_FIT_choice'];tails=[]
    for tape in tapes:
        for endpoint in tape['AUDIT_large'][choice]['endpoints']:
            if endpoint['tail'] is not None:tails.append(endpoint['tail'])
    out['oldtail_evaluable']=len(tails)==6;out['oldtail_mean_delta']=average(tails)
    return out


def run(folder):
    folder=Path(folder);stream_results={};all_rows=[]
    for name in ['a','b']:
        saved=json.loads((folder/name/'RESULTS.json').read_text());roots=[root_summary(r) for r in saved['roots']]
        valid=[r for r in roots if not r['skipped']];gain=average([r['H3_large_FIT_to_large_AUDIT_R'] for r in valid]);risk=[r['oldtail_mean_delta'] for r in valid if r['oldtail_evaluable']]
        positives=sum(r['H3_large_FIT_to_large_AUDIT_R']>0 for r in valid)
        gate=len(valid)==16 and gain>=.005 and positives>=8 and len(risk)>=8 and average(risk)>=-.005
        stream_results[saved['stream']]={'completed_scenarios':len(roots),'evaluable_scenarios':len(valid),'large_FIT_to_large_AUDIT_mean_R':gain,
               'small_FIT_to_same_large_AUDIT_mean_R':average([r['H3_small_FIT_to_large_AUDIT_R'] for r in valid]),'positive_scenarios':positives,
               'oldtail_evaluable_scenarios':len(risk),'mean_oldtail_delta':average(risk),'P2_development_evidence_gate':gate,
               'AUDIT_self_oracle_mean_R_descriptive_only':average([r['H3_AUDIT_self_oracle_descriptive_only'] for r in valid]),'counts':saved['counts']}
        all_rows.extend([dict(stream=saved['stream'],**r) for r in roots])
    output={'phase':'COMPLETE_P1_SAVED_ANALYSIS','streams':stream_results,'gate_both_streams':all(s['P2_development_evidence_gate'] for s in stream_results.values()),
            'root_scenarios_not_independent_patients':True,'clinical_patient_independence_verified':False,'actor_verifier_updates':0,'VAL_TEST':0,
            'automatic_P2_or_RL_started':False,'mainline_performance_claim':False}
    write(folder/'RESULT_SUMMARY.json',output);write(folder/'ROOT_SUMMARY.json',all_rows)
    keys=sorted(set().union(*(r.keys() for r in all_rows)))
    with (folder/'ROOT_SUMMARY.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(all_rows)
    (folder/'REPORT_ZH.md').write_text('# ECV P1：动作机会与查询迁移诊断\n\n'+json.dumps(output,ensure_ascii=False,indent=2)+'\n\n32 个诊断状态复用有限的身份组件和八个 donor 阶段快照，不能当作 32 名独立患者。H=3 是已到达 TRAIN 内的三次影子学习/记忆决策，不是真实未来三阶段性能。AUDIT 自选最优只作描述上界。无 VAL/TEST，无控制器优化。详细逐根与原始整数分母已保存；不得以门通过代替方法或 RL 独立增益。\n')
    return output

