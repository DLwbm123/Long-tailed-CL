import os,json
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);rows=json.loads((r/'RECORDS.private.json').read_text())
assert json.loads((r/'VERIFICATION.json').read_text())['passed']
counts=cfg['domains']['HK']['train_counts_ordered'];controls=['duplicate','CLS'];arms=['NCM','natural']+controls+['patch']
def mean(v):return sum(v)/len(v) if v else None
def metric(q,tail=False,subset=None):
 cc=[i for i,z in enumerate(q['counts']) if z and (not tail or counts[i]<250) and (subset is None or i in subset)]
 return None if not cc else 100*mean([q['correct'][i]/q['counts'][i] for i in cc])
table=[];groups=[]
for tag in ['full','Qa','Qb']:
 for stage in range(6):
  cells=[v for v in rows if v['stage']==stage];values={a:{m:mean([metric(v['query'][a][tag],m=='tail') for v in cells]) for m in ['macro','tail']} for a in arms}
  gains={a:{m:values['patch'][m]-values[a][m] for m in ['macro','tail']} for a in controls+['NCM']}
  table.append(dict(stage=stage,query=tag,arms=values,gains=gains,four_positive=all(gains[a][m]>0 for a in controls for m in ['macro','tail'])))
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
  d=[100*(v['query']['patch']['full']['correct'][c]-v['query'][a]['full']['correct'][c])/v['query'][a]['full']['counts'][c] for v in cc]
  contrast[a]=dict(mean_recall_change_pp=mean(d),improved=sum(x>0 for x in d),harmed=sum(x<0 for x in d),same=sum(x==0 for x in d))
 classes.append(dict(class_index=c,cells=len(cc),patch_minus=contrast))
full=[v for v in table if v['query']=='full'];global_gain={a:{m:mean([v['gains'][a][m] for v in full]) for m in ['macro','tail']} for a in controls+['NCM']}
tiny=[classes[c]['patch_minus']['NCM']['mean_recall_change_pp'] for c in [1,2,12]];tiny_gate=all(x>=0 for x in tiny) and sum(x>0 for x in tiny)>=2
stages=sum(v['four_positive'] for v in full);last=full[-1]['gains']['NCM'];gate=all(global_gain[a][m]>0 for a in controls for m in ['macro','tail']) and stages>=5 and all(last[m]>0 for m in ['macro','tail']) and tiny_gate
result=dict(global_gain_pp=global_gain,four_positive_stages=stages,tiny_class_gate=tiny_gate,final_NCM_gain_pp=last,necessary_gate=gate,not_GRPO_efficacy=True,new_VAL_TEST_reads=0)
for name,data in [('ANALYSIS',result),('STAGE_METRICS',table),('CLASS_RESPONSES',classes),('OLD_NEW_HEAD_TAIL',groups)]:write(r/f'{name}.json',data)
text='# R86：固定 patch 信息与双份 CLS 对照\n\n同一冻结R81表示状态，原S/Q，全六阶段24个监督CPU头；0视觉optimizer/controller/VALTEST。新通道为CLS+固定patch均值，对照双份CLS，同1024维和同分类头；全部头先seal后Q。\n\n'
text+=f'事前必要门 **{gate}**，全四项同时正阶段 **{stages}/6**，极少类门 **{tiny_gate}**。\n\n| patch 减参照 | 宏 pp | 尾 pp |\n|---|---:|---:|\n'
for a,v in global_gain.items():text+=f"| {a} | {v['macro']:+.6f} | {v['tail']:+.6f} |\n"
text+='\n| stage | 宏 vs duplicate | 尾 vs duplicate | 宏 vs CLS | 尾 vs CLS |\n|---|---:|---:|---:|---:|\n'
for v in full:text+='| '+str(v['stage'])+' | '+' | '.join(f"{v['gains'][a][m]:+.6f}" for a in controls for m in ['macro','tail'])+' |\n'
text+='\n固定全部196非CLS token均值，无patch选择/混合比例/温度搜索；双份CLS数学等距控制由软件检验，实际数值差另行披露。原CLS/NCM/natural参照只读缓存，0重训0重评分。完整23类及Qa/Qb、旧头旧尾新头新尾均保留。class1fold1仍无双component查询覆盖，不补可靠性证据；单HK74002反复TRAIN开发，不是独立患者/新seed/VAL确认。监督通道信息即使有效也不能归为GRPO。保留R80-R85原失败；本次授权只解除固定诊断，其他分支不恢复。\n'
(r/'RESULTS_ZH.md').write_text(text)
