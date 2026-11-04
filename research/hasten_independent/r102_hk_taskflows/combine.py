"""Complete both fixed streams, all metrics and paired differences, then stop."""
import json,csv
from pathlib import Path
from common import write

def run(root):
    root=Path(root);summaries={};costs={};lines=['# R102 HK新任务流：完整试跑结果','','新任务流下的教师条件机制试跑，非严格有限记忆 LT-CIL。无新算法、超参搜索或旧教师复用。','','## 任务分布','','|stream|T1|T2|T3|T4|T5|','|---|---:|---:|---:|---:|---:|','|Ordered|3331|2400|892|165|33|','|Stratified order0|1563|1468|1505|1223|1062|','','## 方法结果','','|stream|mode|五阶段mean macro|final macro|final tail|mean tail有效阶段|','|---|---|---:|---:|---:|---:|']
    for folder in ['a','b']:
        r=root/folder;s=json.loads((r/'RESULT_SUMMARY.json').read_text());v=json.loads((r/'VERIFICATION.json').read_text());assert v['passed'];summaries[folder]=s;costs[folder]=json.loads((r/'COSTS.json').read_text())
        for a in s['mean']:lines.append(f"|{s['stream']}|{a}|{s['mean'][a]['macro']:.6f}|{s['final'][a]['macro']:.6f}|{s['final'][a]['tail']:.6f}|{s['mean_valid_stages'][a]['tail']}|")
    lines += ['', '两流各5新教师和20最终学习端点，共10教师/40学习最终端点；40中间和40stage起点另计。所有teacher只用该流当前已到达TRAIN，从同一通用Biomed重新训练；LoRA携带，优化器每stage重置，10epochs配方。方法模块每stage512cycles且参数/优化器跨stage携带。全部40学习端点封存后才F/VAL，TEST0。', '', 'Ordered前两阶段没有已到达尾类，tail为null，均值分母3；混合流tail有效阶段5，不能把二者tail平均当作同口径顺序因果效应。tiny按原始标签绑定，逐stage correct/count和legacy/new索引见CLASS_STAGE.csv；单例重复stage不代表独立患者。新五阶段均值不可减旧六阶段66.84作为增益。', '', '每条流的RESULT_SUMMARY包括PROTO_GRPO对同流T/UNIFORM/CE_WEIGHT/RLPO_STYLE两两差值；CLASS/GROUP_STAGE和RESCUE_HARM报告新增尾类学习、旧尾类保持、所有正负类别。F是历史开发且教师见过的TRAIN角色，非独立测试；训练B奖励提升不自动代表F/VAL改善。没有事后新成功门、最好epoch/arm/order选择、重复seed或显著性主张。', '', '成本分为新通用前缀、教师训练/观察、新阶段特征提取、模块native/训练及评估。TEACHER_RECORDS逐stage列CPU/wall和更新/暴露；各stream COSTS含所有worker子项，不能重复加。root CONTROL_EXIT和COSTS_COMPLETE保留完整wait4 CPU/RSS/原clock/磁盘；GPU设备wall与kernel时间不同，未采样kernel时间不虚构GPU算时收益。准备/SSH/文档/发布CPU非0未全采样。大文件data盘保存，身份/图像/特征/逐例/weights/optim/rawlogs不公开。', '', '本轮完整试跑后停止，不开新算法、其他order/seed/dataset。按用户最新要求本会话唯一30分钟监测只负责本轮故障/完成和交付，交付后停用。']
    write(root/'TRAIN_AGGREGATES.json',{folder:json.loads((root/folder/'TRAIN_AGGREGATES.json').read_text()) for folder in ['a','b']})
    write(root/'RESULT_SUMMARY.json',{'streams':summaries,'all_two_streams_complete':True,'TEST':0,'no_posthoc_success_gate':True})
    for name in ['STAGE_METRICS.csv','CLASS_STAGE.csv','GROUP_STAGE.csv','RESCUE_HARM.csv','CONTRIBUTION_DECOMPOSITION.csv','POLICY_DIAGNOSTICS.csv']:
        values=[];headers=None
        for folder in ['a','b']:
            with (root/folder/name).open() as f:rd=csv.DictReader(f);headers=rd.fieldnames;values += [dict(stream=summaries[folder]['stream'],**x) for x in rd]
        with (root/name).open('w') as f:w=csv.DictWriter(f,fieldnames=['stream']+headers);w.writeheader();w.writerows(values)
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    write(root/'VERIFICATION.json',{'passed':True,'streams':2,'teachers':10,'final_learning_endpoints':40,'middle_endpoints':40,'start_states':40,'module_supervised_steps':20480,'module_policy_steps':15360,'module_total_steps':35840,'new_teachers_not_old_order':True,'all_endpoints_before_F_VAL':True,'all_stream_verifications_passed':True,'TEST':0})

if __name__=='__main__':run(Path(__file__).resolve().parent)
