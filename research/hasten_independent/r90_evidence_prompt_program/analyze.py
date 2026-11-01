import json
from pathlib import Path
from common import write

def run(cfg):
 r=Path(cfg['output']);v=json.loads((r/'STAGE_METRICS.json').read_text());assert len(v)==6;result={};zero=1e-8
 for primary,refs in [('E',['C','U','WG']),('G',['C','U','WG','E','X','H'])]:
  effects={ref:[{k:z['arms'][primary][k]-z['arms'][ref][k] for k in ['micro','macro','tail']} for z in v] for ref in refs+['NOP','NCM_MEMORY']};mean={ref:{k:sum(x[k] for x in dd)/6 for k in ['micro','macro','tail']} for ref,dd in effects.items()};stage=[all(effects[ref][s][k]>zero for ref in refs for k in ['macro','tail']) for s in range(6)];global_ok=all(mean[ref][k]>zero for ref in refs for k in ['macro','tail']);last=v[-1]['arms'];final=all(last[primary][view][k]>last[ref][view][k]+zero for ref in ['NOP','NCM_MEMORY'] for view in ['modal'] for k in ['macro','tail']) and all(effects[ref][-1][k]>zero for ref in ['NOP','NCM_MEMORY'] for k in ['macro','tail']);tiny={str(c):last[primary]['recall'][c]-last['NCM_MEMORY']['recall'][c] for c in [1,2,12]};tiny_ok=all(x>=-zero for x in tiny.values()) and sum(x>zero for x in tiny.values())>=2
  result[primary]={'effects_pp':effects,'mean_stage_gains_pp':mean,'global_gate':global_ok,'stage_all_positive':stage,'stage_gate':sum(stage)>=5,'final_expected_and_modal_gate':final,'tiny_expected_gain_pp':tiny,'tiny_gate':tiny_ok,'necessary_gate':global_ok and sum(stage)>=5 and final and tiny_ok}
 result['scope']='Mechanism bridge only. Shared canonical dictionary, one HK seed, not original visual-GRPO improvement, complete RLAP, independent patients or final novelty.';write(r/'ANALYSIS.json',result)
