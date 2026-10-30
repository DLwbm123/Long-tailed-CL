import os,json,statistics
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);rows=json.loads((r/'RECORDS.private.json').read_text())
assert json.loads((r/'VERIFICATION.json').read_text())['passed']
def mean(v):return sum(v)/len(v) if v else None
counts=cfg['domains']['HK']['train_counts_ordered'];table=[];groups=[];classes=[]
def metric(q,tail=False):
 c=[i for i,n in enumerate(q['counts']) if n and (not tail or counts[i]<250)]
 return 100*mean([q['correct'][i]/q['counts'][i] for i in c])
for tag in ['full','Qa','Qb']:
 for stage in range(6):
  cells=[v for v in rows if v['stage']==stage];entry=dict(stage=stage,query=tag,arms={})
  for a in ['NCM','natural','balanced']:
   entry['arms'][a]=dict(macro=mean([metric(v['query'][a][tag]) for v in cells]),tail=mean([metric(v['query'][a][tag],True) for v in cells]))
  entry['gains']={a:{m:entry['arms']['balanced'][m]-entry['arms'][a][m] for m in ['macro','tail']} for a in ['NCM','natural']}
  entry['four_positive']=all(z>0 for d in entry['gains'].values() for z in d.values());table.append(entry)
  for group in ['old_head','old_tail','new_head','new_tail']:
   old=0 if stage==0 else sum(map(len,cfg['domains']['HK']['stages'][:stage]));ids=[c for c in range(cells[0]['seen']) if (c<old)==group.startswith('old') and (counts[c]<250)==group.endswith('tail')]
   values={}
   for a in ['NCM','natural','balanced']:
    vv=[]
    for v in cells:
     q=v['query'][a][tag];cc=[c for c in ids if q['counts'][c]]
     if cc:vv.append(100*mean([q['correct'][c]/q['counts'][c] for c in cc]))
    values[a]=mean(vv)
   groups.append(dict(stage=stage,query=tag,group=group,arms=values))
for c in range(23):
 vals=[];positive=negative=same=0;covered=0
 for v in rows:
  if c>=v['seen']:continue
  base=v['query']['NCM']['full'];new=v['query']['balanced']['full'];assert base['counts'][c]>0
  d=(new['correct'][c]-base['correct'][c])/base['counts'][c];vals.append(d);positive+=d>0;negative+=d<0;same+=d==0;covered+=1
 classes.append(dict(class_index=c,cells=covered,mean_recall_change_pp=100*mean(vals),improved=positive,harmed=negative,same=same))
full=[v for v in table if v['query']=='full'];global_gain={a:{m:mean([v['gains'][a][m] for v in full]) for m in ['macro','tail']} for a in ['NCM','natural']}
tiny=[v for v in classes if v['class_index'] in [1,2,12]];stage_positive=sum(v['four_positive'] for v in full)
tiny_gate=all(v['mean_recall_change_pp']>=0 for v in tiny) and sum(v['mean_recall_change_pp']>0 for v in tiny)>=2
gate=all(x>0 for v in global_gain.values() for x in v.values()) and stage_positive>=5 and tiny_gate
summary=dict(global_gain_pp=global_gain,four_positive_stages=stage_positive,tiny_class_gate=tiny_gate,
 necessary_boundary_gate=gate,not_GRPO_efficacy=True,new_VAL_TEST_reads=0)
for name,value in [('ANALYSIS',summary),('STAGE_METRICS',table),('CLASS_RESPONSES',classes),('OLD_NEW_HEAD_TAIL',groups)]:write(r/(name+'.json'),value)
text='# R84：冻结表示下的独立判别边界\n\n仅 TRAIN 固定六阶段、两 component folds；24 个监督 CPU 判别头拟合，0 视觉优化/控制器/VAL/TEST。所有 head 先封存，Q 仅评估。不是 GRPO 收益证据。\n\n'
text+='必要门：**'+str(gate)+'**；四项同时正阶段 '+str(stage_positive)+'/6；极少类正确数门 '+str(tiny_gate)+'。\n\n'
text+='| balanced 减参照 | 宏 pp | 尾 pp |\n|---|---:|---:|\n'
for a in global_gain:text+=f"| {a} | {global_gain[a]['macro']:+.6f} | {global_gain[a]['tail']:+.6f} |\n"
text+='\n全部阶段、类别方向、旧头旧尾新头新尾及 Qa/Qb 见聚合文件。class1 fold1 缺双查询 component，保留 fullQ，不能外推该类跨查询可靠性。固定 R81 源模型以前见过 TRAIN；这不是新 seed、患者独立验证或 VAL 确认。\n\n'
text+='自然/平衡头同一 S、总数据质量 n、相同单位二次锚与确定性求解，不用 Q 选参数。不论支持拟合多好，不能替代查询识别门。若失败，不据 Q 扫锚定强度/头结构或追加选择器；若成功，只支持后续匹配策略实验，不能归因为 RL。R80/R81/R82/R83 所有历史负证据保持。\n'
(r/'RESULTS_ZH.md').write_text(text)
