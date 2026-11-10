"""Saved public aggregates only: paired contributions and reward opportunity audit."""
import csv
import json
import time
from pathlib import Path


def run(cfg):
    cpu=time.process_time();wall=time.time();root=Path(cfg['output'])
    load=lambda n:json.loads((root/n).read_text())
    assert load('FINAL_VERIFICATION.json')['passed']
    final=load('FINAL_RESULTS.json');classes=list(csv.DictReader((root/'CLASS_RESULTS.csv').open()))
    groups=list(csv.DictReader((root/'GROUP_DIAGNOSTICS.csv').open()))
    by={(int(r['stage']),r['arm'],int(r['class'])):r for r in classes};assert len(by)==360
    contributions=[];differences={}
    for control in ['GRPO','TARS','RAPO','TARS_CE']:
        total_macro=total_tail=0.
        for stage in range(1,5):
            current=[r for r in classes if int(r['stage'])==stage and r['arm']=='TARS_RAPO']
            ntail=sum(int(r['TRAIN_count'])<250 for r in current)
            for row in current:
                c=int(row['class']);other=by[stage,control,c]
                delta=float(row['recall_percent'])-float(other['recall_percent'])
                mc=delta/len(current)/4;tc=delta/ntail/4 if int(row['TRAIN_count'])<250 else 0.
                total_macro+=mc;total_tail+=tc
                contributions.append({'control':control,'stage':stage,'class':c,'name':row['name'],
                    'main_correct':int(row['correct']),'control_correct':int(other['correct']),
                    'VAL_denominator':int(row['denominator']),'delta_recall_pp':delta,
                    'post_mean_macro_contribution_pp':mc,'post_mean_tail_contribution_pp':tc})
        expected={k:100*(final['post_stage_means']['TARS_RAPO'][k]-final['post_stage_means'][control][k]) for k in ['macro','tail']}
        assert abs(total_macro-expected['macro'])<1e-10 and abs(total_tail-expected['tail'])<1e-10
        differences[control]=expected
    tiny_ids={int(r['class']) for r in classes if int(r['TRAIN_count'])<25}
    opportunities={}
    for arm in ['GRPO','TARS','RAPO','TARS_RAPO']:
        seq=[r for r in groups if r['arm']==arm and int(r['class']) in tiny_ids]
        totals={k:sum(int(r[k]) for r in seq) for k in ['queries','all_wrong','all_correct','mixed','correct_responses']}
        totals['per_class']=[{k:int(r[k]) for k in ['stage','class','queries','all_wrong','all_correct','mixed','correct_responses']} for r in seq]
        assert totals['queries']==totals['all_wrong']+totals['all_correct']+totals['mixed']
        opportunities[arm]=totals
    controls={a:{k:100*v for k,v in row.items()} for a,row in final['post_stage_means'].items()}
    final_rows=[r for r in final['stage_readouts'] if r['stage']==4]
    first_ids=range(5)
    initial_old_macro=sum(float(by[0,'TARS_RAPO',c]['recall_percent']) for c in first_ids)/5
    final_old_macro=sum(float(by[4,'TARS_RAPO',c]['recall_percent']) for c in first_ids)/5
    summary={'scope':'saved_public_aggregate_analysis_no_new_queries_or_updates','new_model_queries_updates':0,
        'post_stage_mean_percent':controls,'main_over_controls_pp':differences,
        'original_gates_passed':sum(final['gates'].values()),'original_gates_total':9,
        'original_initial_task_head_change_pp':100*final['initial_task_head_change'],
        'initial_first_task_macro_percent':initial_old_macro,'final_first_task_macro_percent':final_old_macro,
        'mean_classwise_forgetting_percent':{a:100*v for a,v in final['mean_classwise_forgetting'].items()},
        'tiny_TRAIN_reward_opportunities':opportunities,
        'main_minus_RAPO_nonzero_tail_contributions':[r for r in contributions if r['control']=='RAPO' and abs(r['post_mean_tail_contribution_pp'])>1e-12],
        'final_metrics_percent':[{k:(100*r[k] if k in ['macro','head','tail','tiny','micro'] else r[k]) for k in ['arm','macro','head','tail','tiny','micro']} for r in final_rows],
        'all_original_metrics_and_failed_gates_preserved':True,
        'CPU_analysis_body_seconds':time.process_time()-cpu,'wall_analysis_body_seconds':time.time()-wall}
    (root/'MECHANISM_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (root/'PAIRED_CLASS_CONTRIBUTIONS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(contributions[0]));w.writeheader();w.writerows(contributions)
    report=['# R125 完整机制结论：RL能启动，保留有增益，TARS叠加未获联合收益','',
        '**全部5真实任务/5臂完成并独立CPU保存验证；单order重复开发VAL，非独立患者或创新证明。监督和接口贡献不记RL。**','',
        '|arm|4个post阶段平均macro %|平均tail %|平均classwise遗忘 pp|','|---|---:|---:|---:|']
    for arm in controls:
        report.append(f"|{arm}|{controls[arm]['macro']:.4f}|{controls[arm]['tail']:.4f}|{100*final['mean_classwise_forgetting'][arm]:.4f}|")
    report+=['','主对TARS：macro+5.9665pp、tail+1.8561pp，遗忘减少19.9381pp；有真实保留增益，非R124零更新。原tail门+2pp未达，首task head仍下降41.4883pp。',
        '主对单独RAPO：平均macro−.0278pp、tail−1.1905pp；组合不支持净增益。最终单点macro主16.7269% vsRAPO16.4724%不能替代事先冻结的跨阶段平均。',
        '对匹配CE：平均macro+8.1993pp、tail−.7066pp。CE无G8/KL且计算更少/反馈形式不同，不能用macro优势证明RL目标必要性。原九门4/9，失败门完整保留。',
        '全部4RL各512实际非零参数步、已获得正确与mixed奖励；本轮解决的是接口条件下的启动资格，而不是完整持续学习成功。接口也改了共同首task SFT，历史R124对比不能单独归因于post提示。',
        '## 精确贡献与小分母','',
        'PAIRED_CLASS_CONTRIBUTIONS按固定原指标精确分解，贡献总和核对原macro/tail差值，未删类别、改分母或重设门。主对RAPO的平均tail差异全部来自stage1 impacted-stool的4个净正确差（主3/21，RAPO7/21）；其余阶段tail相同。不据此事后给该类调权或改变原tail结论，单seed小样本不代表稳定显著优劣。',
        '最终RaPO和主tiny均5%，都只是UC-grade-2-3的1/4；其他四tiny类仍0。不是五类普遍改善或独立临床验证。完整逐类分母与TRAIN奖励机会在JSON/CSV。',
        '## 可支持与不可支持的机制解释','',
        '相对无RaPO的TARS/GRPO，当前任务候选级保留反馈与持续sigma的配方明显减少旧类遗忘；这是匹配配方结果，尚未分离retention奖励、CTAN和交互的各自因果。',
        'TARS加到RAPO后没有原平均macro/tail净增量。仅重分配loss权重不能自动产生稀缺类正确回答；tiny类查询、全错/混合组和成功数必须一起看。RaPO在全错准确率组仍可能有不同retention奖励，不能把accuracy-mixed=0当PG必为0。',
        '保留代理只看当前TRAIN与旧policy，不观测真实旧图像风险；即使相对保留更好，也没有保证首task head保护。classwise peak-to-final遗忘也可能受学习新类程度影响，必须同时报old/new/峰值/初始首task头类，不能独立当安全证据。',
        '现有记录没有逐token轨迹/每类梯度，不能从组份额、权重或混合数证明梯度方差、错误retention强化的因果，也不追加生成补造。',
        '## 收口与后继边界','',
        '关闭本次词表资格变体；不扫提示/格式/长度/seed/SFT/分辨率或旧保护阈值，不把+1.8561调到2救门。原R117等旧失败保持。RL仍是主线，下一步只能围绕真实旧风险可获得反馈与尾类学习机会的新可否证机制，先排除R67/R105/R113旧闭合路线。',
        '本分析0新模型调用/更新，无新GPU实验启动；若下一机制只是静态采样、记忆选择器或已知简单组合换名，明确否决，不以持续授权制造盲跑。当前不是用户暂停或权限/存储阻塞。']
    (root/'MECHANISM_ANALYSIS_ZH.md').write_text('\n'.join(report)+'\n')
    return summary
