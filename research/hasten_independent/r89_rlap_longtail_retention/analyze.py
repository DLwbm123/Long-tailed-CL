import json
from pathlib import Path
from common import write

def run(cfg):
 r=Path(cfg['output']);v=json.loads((r/'STAGE_METRICS.json').read_text());assert len(v)==6;keys=['micro','macro','tail'];zero=1e-8
 effects={}
 for ref in ['BASE','MACRO','RETENTION','NCM','NCM_ALL']:
  d=[{k:z['arms']['JOINT'][k]-z['arms'][ref][k] for k in keys} for z in v];effects[ref]={'stage_gains_pp':d,'mean_stage_gains_pp':{k:sum(x[k] for x in d)/6 for k in keys}}
 stage=[all(effects[a]['stage_gains_pp'][s][k]>zero for a in ['BASE','MACRO','RETENTION'] for k in ['macro','tail']) for s in range(6)]
 global_ok=all(effects[a]['mean_stage_gains_pp'][k]>zero for a in ['BASE','MACRO','RETENTION'] for k in ['macro','tail']);last=v[-1]['arms'];tiny={str(c):last['JOINT']['recall'][c]-last['NCM']['recall'][c] for c in [1,2,12]};tiny_ok=all(x>=-zero for x in tiny.values()) and sum(x>zero for x in tiny.values())>=2;final=all(effects['NCM']['stage_gains_pp'][-1][k]>zero for k in ['macro','tail'])
 interaction=[{k:z['arms']['JOINT'][k]-z['arms']['RETENTION'][k]-z['arms']['MACRO'][k]+z['arms']['BASE'][k] for k in keys} for z in v]
 result={'effects':effects,'interaction_pp':interaction,'stage_all6positive':stage,'global_gate':global_ok,'stage_gate':sum(stage)>=5,'tiny_gain_pp':tiny,'tiny_gate':tiny_ok,'final_vs_bounded_NCM':final,'necessary_gate':global_ok and sum(stage)>=5 and tiny_ok and final,'claims':'2x2 mechanism diagnosis, not final novelty, official RLAP, standard GRPO, multiple seeds or independent patients.'};write(r/'ANALYSIS.json',result);return result
