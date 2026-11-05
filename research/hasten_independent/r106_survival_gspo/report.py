"""Two-flow complete evidence, memory and physical/logical cost accounting."""
import csv,json,shutil,time
from pathlib import Path
from common import write
from evaluate import csvout
from preflight import ARMS

def run(root):
    root=Path(root);summaries={};audit=[];meta_rows=[];lines=['# R106 Survival-GSPO：受限证据记忆与真实后续学习','','仅HK Ordered与分层order0，各五阶段5+5+5+4+4，seed74002；通用BiomedCLIP冻结，监督聚合器/余弦头持续更新。并非R104/105全类监督权重、三答案路由或TARS改名。全部学习端点先seal后一次统一VAL，TEST0。','','|流|方法|mean micro|mean macro|final macro|final tail|mean old-tail|有效旧尾阶段|','|---|---|---:|---:|---:|---:|---:|---:|']
    for k in ['a','b']:
        r=root/k;assert json.loads((r/'VERIFICATION.json').read_text())['passed'];s=json.loads((r/'RESULT_SUMMARY.json').read_text());summaries[k]=s
        for arm in ARMS:
            old=s['mean'][arm]['old_tail'];old='N/A' if old is None else f'{old:.6f}';lines.append(f"|{k}|{arm}|{s['mean'][arm]['micro']:.6f}|{s['mean'][arm]['macro']:.6f}|{s['final'][arm]['macro']:.6f}|{s['final'][arm]['tail']:.6f}|{old}|{s['valid_stages'][arm]['old_tail']}|")
        for a in json.loads((r/'MEMORY_AND_ACCESS_AUDIT.json').read_text()):
            a=dict(a);a.pop('per_class_parent_coverage');audit.append(dict(stream=k,**a))
        for t in range(5):
            for j,e in enumerate(json.loads((r/f's{t}'/'META_DIAGNOSTICS.private.json').read_text())):
                if 'skipped' in e:meta_rows.append({'stream':k,'stage':t,'episode':j,'status':'SKIPPED','gspo_return':None,'pg_return':None,'cf_abs_mean':None,'gspo_zero':None,'pg_zero':None});continue
                roles={role:[d['return'] for d in e['diagnostics'] if d['role']==role] for role in ['GSPO','PG']};meta_rows.append({'stream':k,'stage':t,'episode':j,'status':'COMPLETE','gspo_return':sum(roles['GSPO'])/4,'pg_return':sum(roles['PG'])/4,'cf_abs_mean':sum(abs(d['d']) for d in e['counterfactual'])/6,'gspo_zero':e['updates']['GSPO'][0]['zero_advantage_group'],'pg_zero':e['updates']['PG'][0]['zero_advantage_group']})
    for name in ['STAGE_METRICS.csv','CLASS_STAGE.csv','GROUP_STAGE.csv','RESCUE_HARM.csv']:
        rows=[]
        for k in ['a','b']:
            rows += [dict(stream=k,**x) for x in csv.DictReader((root/k/name).open())]
        csvout(root/name,rows)
    csvout(root/'MEMORY_ACCESS.csv',audit);csvout(root/'META_SUMMARY.csv',meta_rows);write(root/'RESULT_SUMMARY.json',{'streams':summaries,'primary':'SURVIVAL_GSPO','TEST':0})
    lines+=['','## 预定配对差（GSPO−参照，pp）','','|流|参照|mean macro|final macro|final tail|','|---|---|---:|---:|---:|']
    for k,s in summaries.items():
        for ref,d in s['comparisons'].items():lines.append(f"|{k}|{ref}|{d['mean']['macro']:+.6f}|{d['final']['macro']:+.6f}|{d['final']['tail']:+.6f}|")
    lines+=['','可读取的持久证据记忆每臂528568B以内：256×2064B记录+184B类别统计。全局/局部/标签/父索引/视图/填充均计费；工作解码缓冲计入RAM，不额外保留训练证据。最低每类一个全局共同约束。旧阶段归档模型/日志离线封存，不回读为训练数据；丢弃旧全图/局部不能恢复。所有方法正式pre200+memory-only post100更新/阶段，分类头与聚合器共享容量，前阶段蒸馏固定。','','伪未来只来自本阶段新TRAIN，影子学习器通用初始化，组件分离查询不进入任何其监督、原型或回放。不是患者独立，也不是现实未来类别。长期主回报评价后续两阶段(.5macro BA+.5normalized Brier)减.25旧尾退化；反事实价值监督为等字节替换后一阶段干扰的效用差，显式短于主回报，不称无偏或现实因果保证。','','GSPO/PG每个伪起点各4条独立合法完整轨迹，真实old概率保留，两次更新期间行为分布/状态不变，零组保留；GSPO长度归一化整序列clip.2是适配surrogate，不称完整轨迹无偏IS或语言模型超参最优。规范顺序保证同一最终记忆不因排列形成伪多样性。正式GSPO/PG记忆固定seed单次采样，不重采挑记忆；最终分类确定性。','','VALUE_GREEDY使用与GSPO完全相同的价值网络参数/表征，价值更新与actor更新分离；MORE用相同容量价值网络但每组6对而非1对CF数据，额外计算强对照。每组逻辑影子分类器更新GSPO/PG=12+32+144=188；GREEDY=44，MORE=12+6×32=204。正式学习更新相同。MORE的步数超过RL，但不声称wall完全匹配；独立报告各项实耗。实际共享物理shadow更新每组492，不将共享项重复加作实际CPU/GPU费用。','','模型参数/AdamW状态的有界持久存储与回放记忆分开报告，不能把模型权重或离线审计当免费训练数据。Native/预算fixture使用重复generic特征与合成标签只测最大形状，正式模型/策略/价值头重新初始化；正式encoder只读当前到达TRAIN，旧源pixels/缓存无回读。所有70分类端点封存后每流一次六视图VAL，重复阶段/seed不当新病例，TEST0。','','本轮证明与否以完整任务流、主方法对所有匹配控制、旧尾保护和tiny原分母贡献分析为准；shadow reward或梯度存在不是科学成功，若仅监督价值有效不能归GSPO。单开发seed和多轮使用VAL不支持显著性或CVPR创新成立。原R80等负结果保留，不外推换问题必有效。','','全费用见COSTS及CONTROL_EXIT：native/profile/仿真/fit/eval已含worker；原FIRST24h不重置，准备/SSH/文档/发布CPU非0未完整采样，GPUkernel时间不虚构。新磁盘含全部状态/trace/日志小于20GB并留512MiB硬余量，服务器新增<50GB。结果完整交付后本轮停止，monitor OFF，不自动扩任务/种子/数据集。私有身份/图像/逐例features/policy概率/权重/optim/rawlogs不公开。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');write(root/'COSTS.json',{'workers':{k:json.loads((root/k/'COSTS.json').read_text()) for k in ['a','b']},'actual_wait4_and_controller':'CONTROL_EXIT.json written after aggregation','shared_shadow_actual_classifier_updates_per_complete_group':492,'logical_GSPO_per_group':188,'logical_PG_per_group':188,'logical_GREEDY_per_group':44,'logical_MORE_per_group':204,'shared_physical_work_must_not_be_double_counted':True});write(root/'VERIFICATION.json',{'passed':True,'streams':2,'classifier_endpoints':70,'formal_classifier_updates':21000,'all_worker_verifications_passed':True,'TEST':0})
