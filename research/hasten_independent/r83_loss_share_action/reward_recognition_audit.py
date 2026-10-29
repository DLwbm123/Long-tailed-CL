"""Descriptive decomposition of the already frozen selector; no alternative selection policy."""
import os,json,time
from pathlib import Path
from common import write
r=Path(json.loads(Path(os.environ['RUN_CONFIG']).read_text())['output']);rows=json.loads((r/'HK/RECORDS.private.json').read_text());out=[]
for z in rows:
 actions=[b for b in z['branches'] if not b['duplicate'] and not b['fixed_mean']];item=dict(stage=z['stage'],fold=z['fold'],repeat=z['repeat'])
 for source,target,tag in [('Qa','Qb','forward'),('Qb','Qa','reverse')]:
  top=max(b['stats'][source]['utility'] for b in actions);best=[b for b in actions if top-b['stats'][source]['utility']<=1e-7]
  def mean(bs,key):return sum(b['stats'][target][key] for b in bs)/len(bs)
  du=mean(best,'utility')-mean(actions,'utility');dd=mean(best,'damage')-mean(actions,'damage');mb=lambda b:sum(v for v,c in zip(b['query'][target]['brier'],b['query'][target]['counts']) if c)/sum(c>0 for c in b['query'][target]['counts']);dm=sum(mb(b) for b in best)/len(best)-sum(mb(b) for b in actions)/len(actions);assert abs(du+dm+dd)<1e-12
  item[tag]=dict(utility_gain_vs_uniform=du,macro_Brier_change=dm,protected_damage_change=dd,BA_gain_vs_uniform=mean(best,'BA')-mean(actions,'BA'),tail_BA_gain_vs_uniform=mean(best,'tail_BA')-mean(actions,'tail_BA'))
 out.append(item)
summary={tag:dict(mean_utility_gain=sum(x[tag]['utility_gain_vs_uniform'] for x in out)/36,mean_macro_Brier_change=sum(x[tag]['macro_Brier_change'] for x in out)/36,mean_damage_change=sum(x[tag]['protected_damage_change'] for x in out)/36,positive_utility_blocks=sum(x[tag]['utility_gain_vs_uniform']>1e-7 for x in out),positive_utility_BA_negative_blocks=sum(x[tag]['utility_gain_vs_uniform']>1e-7 and x[tag]['BA_gain_vs_uniform']<0 for x in out),positive_utility_tail_negative_blocks=sum(x[tag]['utility_gain_vs_uniform']>1e-7 and x[tag]['tail_BA_gain_vs_uniform']<0 for x in out)) for tag in ['forward','reverse']};summary.update(scope='descriptive decomposition of frozen R83 selector, no alternate selector or new model scores',new_model_scores=0,new_optimizer_steps=0);write(r/'results/REWARD_RECOGNITION_DECOMPOSITION.json',dict(summary=summary,blocks=out));print(json.dumps(summary))
