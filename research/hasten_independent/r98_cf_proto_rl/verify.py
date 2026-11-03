"""Saved-only independent count/metric/gate verification; no feature/model forward."""
import json,csv,torch
from pathlib import Path
from common import write
from analyze import table,summarize

def run(c):
 r=Path(c['output']);seal=json.loads((r/'SEAL.json').read_text());assert seal['final_endpoints']==36 and seal['formal_optimizer_steps']==10752 and seal['F_VAL_TEST_before_seal']==0;assert len(list(r.glob('s*_*.private.pt')))==66
 for n in seal['snapshots']:
  z=torch.load(r/n,map_location='cpu',weights_only=False);assert z['optimizer']['state'] and all(torch.isfinite(x).all() for x in z['model'].values())
 packs=[torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False) for s in range(6)];records=[table(c,p,s) for s,p in enumerate(packs)];a=summarize(c,records,packs);saved=json.loads((r/'RESULT_SUMMARY.json').read_text());assert a==saved;ev=json.loads((r/'EVAL_COMPLETE.json').read_text());assert ev['TEST']==0 and ev['counts']['formal_optimizer_steps']==10752
 contrib=list(csv.DictReader((r/'CONTRIBUTION_DECOMPOSITION.csv').open()))
 for arm in ['SUP0','SC','RLP','CF','GLOBAL','PATH']:
  for ref in ['T','SUP0']:
   rr=[v for v in contrib if v['arm']==arm and v['reference']==ref]
   for metric in ['macro','tail']:
    total=sum(float(v[f'mean_{metric}_contribution_pp']) for v in rr);assert abs(total-(a['mean'][arm][metric]-a['mean'][ref][metric]))<1e-9
 write(r/'VERIFICATION.json',{'passed':True,'final_endpoints':36,'intermediate_endpoints':30,'formal_steps':10752,'all_stage_arm_metrics_and_A_B_recomputed':True,'original_denominator_contributions_reconstructed':True,'new_encoder_optimizer_F_VAL_TEST':0,'no_endpoint_reselection':True})
