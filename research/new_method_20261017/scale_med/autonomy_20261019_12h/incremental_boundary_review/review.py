"""Arithmetic of previously published aggregates; no private assets or fitting."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
a=json.loads((root/'initial_bias/results/fit_aggregate.json').read_text())
names=['G_Matched','P_Matched','C_Matched','G_Bias','P_Bias','C_Bias']
rows={n:[a['summary'][n][str(s)]['recall'][1]*100 for s in range(3)] for n in names}
r=dict(source='initial_bias/results/fit_aggregate.json (public aggregate)',class1_recalls_pct=rows,
       class1_stage2_minus_stage0_pp={n:v[2]-v[0] for n,v in rows.items()},
       class1_C_Matched_minus_G_Matched_pp=[rows['C_Matched'][s]-rows['G_Matched'][s] for s in range(3)],
       fixed_alpha=.01,class_counts=[4,5,6],unnormalized_A_effective_penalty=[k*.01 for k in [4,5,6]],
       no_new_test_access=True,new_fit_units=0,new_scores=0,method_success=False,
       point_arithmetic_is_not_new_statistical_test=True)
Path(__file__).with_name('aggregate.json').write_text(json.dumps(r,indent=2)+'\n')
