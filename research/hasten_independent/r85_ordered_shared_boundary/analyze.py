import os,json
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);rows=json.loads((r/'RECORDS.private.json').read_text())
assert json.loads((r/'VERIFICATION.json').read_text())['passed']
counts=cfg['domains']['HK']['train_counts_ordered'];controls=['unstructured','degree_only','family_mean'];arms=['NCM','natural']+controls+['ordered']
def mean(v):return sum(v)/len(v) if v else None
def metric(q,tail=False,subset=None):
 cc=[i for i,z in enumerate(q['counts']) if z and (not tail or counts[i]<250) and (subset is None or i in subset)]
 return None if not cc else 100*mean([q['correct'][i]/q['counts'][i] for i in cc])
table=[];groups=[]
for tag in ['full','Qa','Qb']:
 for stage in range(6):
  cells=[v for v in rows if v['stage']==stage];values={a:{m:mean([metric(v['query'][a][tag],m=='tail') for v in cells]) for m in ['macro','tail']} for a in arms}
  gains={a:{m:values['ordered'][m]-values[a][m] for m in ['macro','tail']} for a in controls+['NCM']}
  table.append(dict(stage=stage,query=tag,arms=values,gains=gains,six_positive=all(gains[a][m]>0 for a in controls for m in ['macro','tail'])))
  old=0 if stage==0 else sum(map(len,cfg['domains']['HK']['stages'][:stage]));n=cells[0]['seen']
  for group in ['old_head','old_tail','new_head','new_tail']:
   cc=[c for c in range(n) if (c<old)==group.startswith('old') and (counts[c]<250)==group.endswith('tail')]
   out={}
   for a in arms:
    vals=[metric(v['query'][a][tag],subset=cc) for v in cells];out[a]=mean([v for v in vals if v is not None])
   groups.append(dict(stage=stage,query=tag,group=group,arms=out))
classes=[]
for c in range(23):
 cc=[v for v in rows if c<v['seen']];contrast={}
 for a in ['NCM']+controls:
  d=[100*(v['query']['ordered']['full']['correct'][c]-v['query'][a]['full']['correct'][c])/v['query'][a]['full']['counts'][c] for v in cc]
  contrast[a]=dict(mean_recall_change_pp=mean(d),improved=sum(x>0 for x in d),harmed=sum(x<0 for x in d),same=sum(x==0 for x in d))
 classes.append(dict(class_index=c,cells=len(cc),ordered_minus=contrast))
full=[v for v in table if v['query']=='full'];global_gain={a:{m:mean([v['gains'][a][m] for v in full]) for m in ['macro','tail']} for a in controls+['NCM']}
tiny=[classes[c]['ordered_minus']['NCM']['mean_recall_change_pp'] for c in [1,2,12]];tiny_gate=all(x>=0 for x in tiny) and sum(x>0 for x in tiny)>=2 and classes[12]['ordered_minus']['unstructured']['mean_recall_change_pp']>0
stages=sum(v['six_positive'] for v in full);last=full[-1]['gains']['NCM'];gate=all(global_gain[a][m]>0 for a in controls for m in ['macro','tail']) and stages>=5 and all(global_gain['NCM'][m]>0 and last[m]>0 for m in ['macro','tail']) and tiny_gate
result=dict(global_gain_pp=global_gain,six_positive_stages=stages,tiny_class_gate=tiny_gate,final_NCM_gain_pp=last,necessary_gate=gate,not_GRPO_efficacy=True,new_VAL_TEST_reads=0)
for name,data in [('ANALYSIS',result),('STAGE_METRICS',table),('CLASS_RESPONSES',classes),('OLD_NEW_HEAD_TAIL',groups)]:write(r/f'{name}.json',data)
text='# R85：等级邻接信息的边界共享诊断\n\nCPU-only，原R84固定特征与S/Q；36新监督CPU头拟合，原NCM/unstructured/natural只读缓存，不重训或重评分；所有新头先seal后Q，无新视觉编码/视觉optimizer/controller/VAL/TEST。\n\n'
text+=f'事前必要门 **{gate}**，全六项同时正阶段 **{stages}/6**，极少类门 **{tiny_gate}**。\n\n| ordered 减参照 | 宏 pp | 尾 pp |\n|---|---:|---:|\n'
for a,v in global_gain.items():text+=f"| {a} | {v['macro']:+.6f} | {v['tail']:+.6f} |\n"
text+='\n| stage | 宏 vs unstructured | 尾 vs unstructured | 宏 vs degree | 尾 vs degree | 宏 vs family_mean | 尾 vs family_mean |\n|---|---:|---:|---:|---:|---:|---:|\n'
for v in full:text+='| '+str(v['stage'])+' | '+' | '.join(f"{v['gains'][a][m]:+.6f}" for a in controls for m in ['macro','tail'])+' |\n'
text+='\n固定单位图先验，不搜索graph/强度/类别顺序。degree_only匹配逐类对角收缩；family_mean匹配总边质量但逐类degree不同，此限制保留。标签顺序仅建模先验，不代表线性临床严重度或重新标注。全部23类与Qa/Qb分解保留；class1fold1无双component覆盖，不补证据。以前源模型见过TRAIN，重复stage/fold非独立患者/新seed。监督信息共享即使有效也不能归GRPO；主门不足不改系数救援，所有R80-R84负证据保持。\n'
(r/'RESULTS_ZH.md').write_text(text)
