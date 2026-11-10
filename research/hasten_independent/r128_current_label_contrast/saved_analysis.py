"""Independent aggregate arithmetic and bounded successor admission decision."""
import csv
import json
import time
from pathlib import Path
from .core import write


def run(cfg):
    r=Path(cfg['output']);load=lambda n:json.loads((r/n).read_text())
    assert load('FINAL_VERIFICATION.json')['passed']
    rows=list(csv.DictReader((r/'CLASS_RESULTS.csv').open()));groups=list(csv.DictReader((r/'GROUP_DIAGNOSTICS.csv').open()))
    arms=list(dict.fromkeys(x['arm'] for x in rows));main='CONTRAST_TARS_RAPO';base='BASE_TARS_RAPO'
    by={(x['arm'],int(x['stage']),int(x['class'])):x for x in rows};counts={int(x['class']):int(x['TRAIN_count']) for x in rows}
    assert len(rows)==len(by)==288 and len(groups)==72
    def recall(a,s,c):
        x=by[a,s,c];return int(x['correct'])/int(x['denominator'])
    def mean(a,s,classes):return sum(recall(a,s,c) for c in classes)/len(classes)
    seen={s:sorted(c for a,stage,c in by if a==base and stage==s) for s in range(5)}
    means={a:{'macro':sum(mean(a,s,seen[s]) for s in range(1,5))/4,'tail':sum(mean(a,s,[c for c in seen[s] if counts[c]<250]) for s in range(1,5))/4} for a in arms}
    forgetting={a:sum(max(recall(a,s,c) for s in range(5) if c in seen[s])-recall(a,4,c) for c in seen[4])/23 for a in arms}
    head=[c for c in seen[0] if counts[c]>=250];change=mean(main,4,head)-mean(main,0,head)
    worst=min(mean(main,s,[c for c in seen[s-1] if counts[c]<250])-mean(base,s,[c for c in seen[s-1] if counts[c]<250]) for s in range(1,5))
    gates={'post_macro_over_BASE_ge_1pp':means[main]['macro']-means[base]['macro']>=.01,'post_tail_over_BASE_ge_2pp':means[main]['tail']-means[base]['tail']>=.02,'initial_task_head_drop_ge_minus1pp':change>=-.01,'worst_old_tail_over_BASE_ge_minus1pp':worst>=-.01}
    for a in ['CE_AUX_TARS_RAPO','POS_KD_TARS_RAPO']:
        for m in ['macro','tail']:gates[f'post_{m}_over_{a}_positive']=means[main][m]>means[a][m]
    gates['classwise_forgetting_less_than_BASE']=forgetting[main]<forgetting[base]
    reported=load('FINAL_RESULTS.json');assert gates==reported['gates']
    for a in arms:
        for m in ['macro','tail']:assert abs(means[a][m]-reported['post_stage_means'][a][m])<1e-12
        assert abs(forgetting[a]-reported['mean_classwise_forgetting'][a])<1e-12
        for s in range(1,5):
            seq=[x for x in groups if x['arm']==a and int(x['stage'])==s]
            assert sum(int(x['queries']) for x in seq)==1024
            for x in seq:assert int(x['all_wrong'])+int(x['all_correct'])+int(x['mixed'])==int(x['queries']) and int(x['all_wrong_queries_with_valid_teacher_pair'])<=int(x['all_wrong'])
    paired=[]
    for a in arms:
        if a==main:continue
        for s in range(1,5):
            tails=[c for c in seen[s] if counts[c]<250]
            for c in seen[s]:
                delta=recall(main,s,c)-recall(a,s,c);paired.append({'control':a,'stage':s,'class':c,'name':by[main,s,c]['name'],'denominator':int(by[main,s,c]['denominator']),'main_correct':int(by[main,s,c]['correct']),'control_correct':int(by[a,s,c]['correct']),'macro_contribution_pp':100*delta/len(seen[s]),'tail_contribution_pp':100*delta/len(tails) if c in tails else 0})
        for m in ['macro','tail']:assert abs(sum(x[m+'_contribution_pp'] for x in paired if x['control']==a)/4-100*(means[main][m]-means[a][m]))<1e-10
    with (r/'PAIRED_CLASS_CONTRIBUTIONS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    summary={'passed':True,'scope':'second_implementation_from_aggregate_integer_class_counts_not_new_model_calls','class_rows':288,'paired_rows':len(paired),'gates':gates,'gate_count':sum(gates.values()),'mean_post':means,'classwise_forgetting':forgetting,'first_task_head_change':change,'first_task_head_initial':mean(main,0,head),'first_task_head_final':mean(main,4,head),'worst_old_tail_over_BASE':worst,'new_model_calls':0,'new_optimizer_updates':0}
    write(r/'INDEPENDENT_AGGREGATE_AUDIT.json',summary)
    text=['# R128 完整证据与机制边界','','原九门3/9；主对BASE四post平均macro '+f"{100*(means[main]['macro']-means[base]['macro']):+.4f}"+'pp/tail '+f"{100*(means[main]['tail']-means[base]['tail']):+.4f}"+'pp；classwise遗忘主 '+f'{100*forgetting[main]:.4f}'+'pp、BASE '+f'{100*forgetting[base]:.4f}'+'pp。原firsttask head损 '+f'{-100*change:.4f}'+'pp，仍含菜单变化，不归为纯参数遗忘或换门。',
      '主对CE_AUX宏 '+f"{100*(means[main]['macro']-means['CE_AUX_TARS_RAPO']['macro']):+.4f}"+'pp/尾 '+f"{100*(means[main]['tail']-means['CE_AUX_TARS_RAPO']['tail']):+.4f}"+'pp；对POS_KD宏 '+f"{100*(means[main]['macro']-means['POS_KD_TARS_RAPO']['macro']):+.4f}"+'pp/尾 '+f"{100*(means[main]['tail']-means['POS_KD_TARS_RAPO']['tail']):+.4f}"+'pp。宏优势不能替代尾/保留联合门。',
      '4臂各512实际非零更新。主科学positive/negative teacher各26112行，POS_KD正教师32768行；有效teacher机会不是缺少分支执行。但未保存原始teacherlogp/逐类梯度，不能证明符号投影压制、错误强化或方差是失败原因。主相对BASE正确TRAIN响应多335也不建立VAL增益。',
      '最终主tiny0，BASE tiny5%仅UC grade2-3的1/4，其余四tiny0；CE_AUX和POS_KD最终tiny均0。阶段3 POS_KD的20%来自UC grade0-1的3/5且最终该类0/5，不能把阶段峰值当稳定保留。',
      'PAIRED_CLASS_CONTRIBUTIONS按全部到达类和全部四post完整分解，不据此删类/改门/定向调权；重复查询不是独立患者，单顺序旧seed共享前缀是开发条件。',
      '只确认当前完整标签对比辅助监督配方未建立对骨干的增益；不否定全部RL、原SIPO或所有上下文方法。关闭本配方参数/teacher周期/seed/提示网格，不保留无效模块包装创新。']
    (r/'MECHANISM_ANALYSIS_ZH.md').write_text('\n\n'.join(text)+'\n')
    now=time.time();old_start=1794271927.5872638;old_deadline=1794343927.5872638;remaining=old_deadline-now
    route={'decision':'reject_immediate_GPU_relaunch_continue_bounded_mechanism_entry_screen','decided_epoch':now,'R128_closed_frozen_auxiliary_family':True,'R126_original_start':old_start,'R126_original_deadline':old_deadline,'R126_remaining_seconds_at_decision':remaining,'R126_original_full_matrix_forecast_seconds':73527.32652,'R126_same_candidate_clock_not_reset':True,'successor_entry_first_preparation_epoch_upper_bound':1794310608.4689498,'successor_candidate_total_deadline_upper_bound':1794397008.4689498,'old_R126_candidate_earlier_clock_takes_precedence':True,'new_experiment_frozen':False,'new_native_or_science_started':False,'next_entry_question':'feedback_and_objective_compatibility_for_semantic_actions_under_cumulative_schema_change','next_screen_max_new_primary_sources':3,'no_user_approval_pending':True,'research_authorization_continues':True}
    write(r/'ROUTE_DECISION.json',route)
    report=['# 后继入口决策：拒绝无依据的立即GPU重启','','R128已完成并负于同骨干；关闭当前标签对比辅助监督配方的κ/δ/CE/KD/K10/seed扫描。R117查询分配、DAPO混合组、旧投影/保护器、R113记忆actor及R122/R123取证器均有旧边界，不能换名复活。',
      f'R126有界真实旧奖励问题仍未受科学检验，但原20h钟目前只余{remaining:.3f}秒，冻结完整矩阵forecast73527.327秒；没有新吞吐/容量证据满足原钟。不删OLD_PROXY/控制、不缩G/update、不放宽1.5margin、不改名重置旧candidate。R128实际耗时来自不同任务/teacher路径，不自动授权重跑R126赌更快。',
      '自主选定下一项有界新机制入口筛选：累计类别schema变化下，旧行为稳定目标与当前语义答案奖励能否兼容。目标是生成式持续RL主线的反馈/目标设计，非再改菜单/类别顺序/输出格式/长度或补SFT；先找最多3个此前未审primary，核实际可获得信息、动作、历史访问、可否证机制与同信息强CE/精确控制。它不是已选定的新算法或创新声明。',
      '具体缺口：R127只给一条固定权重菜单路径归因，不能识别交互；R128当前完整答案teacher只给当前类监督，不能证明旧分布风险；无旧图时旧菜单对新类图的回答可能强制落旧类，不能直接当正确保留反馈。若使用旧图/标签，必须明示有界buffer、同图同反馈控制与出生规则，并与R126旧任务奖励假设区别，不能另开同候选重置钟。',
      '入口须提出此前未测的可获得反馈或真正不同可否证机制；已有简单蒸馏、动态采样、类均衡或模块拼接不能单独当投稿贡献。若无法满足则记录具体否决原因，不能用toy/oracle或GPU忙碌替代证据。',
      '本次没有新实验科学冻结、远端root、native或启动，不伪造准备完成。不等待人审，下一轮继续该有界来源/可否证性入口工作；若形成真实候选，其最早准备保守采用1794310608.4689498或更早（包括本次入口分析、来源/规格），总截止不得晚于1794397008.4689498；若等价R126仍用更早原钟，全部费用/原钟/累计10GB继续。R128旧PID不再poll，不重复已通过finalizer。']
    (r/'ROUTE_DECISION_ZH.md').write_text('\n\n'.join(report)+'\n')
