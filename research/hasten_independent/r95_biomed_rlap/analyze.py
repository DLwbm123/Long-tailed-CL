import json
from pathlib import Path
from common import write

def run(cfg):
 r=Path(cfg['output']);v=json.loads((r/'STAGE_METRICS.json').read_text());assert len(v)==6;zero=1e-8
 comparisons={}
 for ref in ['UNIFORM','CE_WEIGHT','NCM','NCM_ALL']:
  dd=[{k:z['arms']['FULL'][k]-z['arms'][ref][k] for k in ['macro','tail','micro']} for z in v];comparisons[ref]={'stage_gains_pp':dd,'mean_stage_gain_pp':{k:sum(x[k] for x in dd)/6 for k in dd[0]}}
 all4=[all(comparisons[a]['stage_gains_pp'][s][k]>zero for a in ['UNIFORM','CE_WEIGHT'] for k in ['macro','tail']) for s in range(6)];global_ok=all(comparisons[a]['mean_stage_gain_pp'][k]>zero for a in ['UNIFORM','CE_WEIGHT'] for k in ['macro','tail']);last=v[-1]['arms'];tiny={str(c):last['FULL']['recall'][c]-last['NCM']['recall'][c] for c in [1,2,12]};tinyok=all(x>=-zero for x in tiny.values()) and sum(x>zero for x in tiny.values())>=2;final=all(comparisons['NCM']['stage_gains_pp'][-1][k]>zero for k in ['macro','tail']);strongfinal=all(comparisons['NCM_ALL']['stage_gains_pp'][-1][k]>zero for k in ['macro','tail'])
 result={'comparisons':comparisons,'stage_all4positive':all4,'global_RLPO_loss_increment':global_ok,'stage_gate':sum(all4)>=5,'tiny_final_gain_pp':tiny,'tiny_gate':tinyok,'final_vs_bounded_NCM':final,'final_vs_all_TRAIN_NCM':strongfinal,'necessary_gate':global_ok and sum(all4)>=5 and tinyok and final,'claims':'Paper reconstruction only; not official code, standard GRPO, independent patients or multiple seeds. NCM memory and all-TRAIN references both disclosed.'};write(r/'ANALYSIS.json',result);return result
