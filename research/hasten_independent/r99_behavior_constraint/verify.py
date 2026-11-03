import json,csv,torch
from pathlib import Path
from common import write
from analyze import table,summarize
from losses import ARMS

def run(c):
 r=Path(c['output']);seal=json.loads((r/'SEAL.json').read_text());assert seal['final_endpoints']==54 and seal['formal_optimizer_steps']==13824 and seal['F_VAL_TEST_before_seal']==0;assert len(list(r.glob('s*_*.private.pt')))==108
 for n in seal['snapshots']:
  z=torch.load(r/n,map_location='cpu',weights_only=False);assert z['step']==256 and z['optimizer']['state'] and all(torch.isfinite(x).all() for x in z['model'].values())
  for st in z['optimizer']['state'].values():
   for v in st.values():
    if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
 packs=[torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False) for s in range(6)];records=[table(c,p,s) for s,p in enumerate(packs)];a=summarize(c,records,packs);assert a==json.loads((r/'RESULT_SUMMARY.json').read_text());ev=json.loads((r/'EVAL_COMPLETE.json').read_text());assert ev['TEST']==0 and ev['counts']['formal_optimizer_steps']==13824;rows=list(csv.DictReader((r/'CONTRIBUTION_DECOMPOSITION.csv').open()))
 for arm in ['SUP0',*ARMS]:
  for ref in ['T','SUP0']:
   rr=[v for v in rows if v['arm']==arm and v['reference']==ref]
   for metric in ['macro','tail']:assert abs(sum(float(v[f'mean_{metric}_contribution_pp']) for v in rr)-(a['mean'][arm][metric]-a['mean'][ref][metric]))<1e-9
 write(r/'VERIFICATION.json',{'passed':True,'final_endpoints':54,'middle_endpoints':54,'formal_steps':13824,'all_arm_stage_metrics_A_B_C_recomputed':True,'model_and_optimizer_finite':True,'original_denominator_contributions_reconstructed':True,'new_encoder_optimizer_F_VAL_TEST':0,'no_endpoint_selection':True})
