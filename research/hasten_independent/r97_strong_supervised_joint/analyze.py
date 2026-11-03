import json,csv
from pathlib import Path
import torch
from common import write

def run(c):
 r=Path(c['output']);z=json.loads((r/'STAGE_METRICS.json').read_text());seal=json.loads((r/'ALL_STATES_SEALED.json').read_text());summary={'RL_trained':seal['RL_trained'],'mean':{},'final':{},'comparisons_to_teacher':{},'main_gate':None,'historical_strong_SFT_reference':{'mean_macro':66.840041,'final_macro':64.876782,'final_tail':44.924938},'attribution_controls_deferred':True};modes=list(z[0]['arms'])
 for mode in modes:
  summary['mean'][mode]={k:sum(s['arms'][mode][k] for s in z)/6 for k in ['micro','macro','tail']};summary['final'][mode]={k:z[-1]['arms'][mode][k] for k in ['micro','macro','tail']}
 for mode in modes[1:]:
  d={k:[s['arms'][mode][k]-s['arms']['TEACHER'][k] for s in z] for k in ['micro','macro','tail']};mean={k:sum(v)/6 for k,v in d.items()};final={k:v[-1] for k,v in d.items()};both=sum(a>=-1e-8 and b>=-1e-8 for a,b in zip(d['macro'],d['tail']));tiny=[]
  for cl in [1,2,12]:
   harm=rescue=0
   for s in range(6):
    v=torch.load(r/f'VAL_scores_s{s}.private.pt',map_location='cpu',weights_only=False);yy=v['labels'];mask=yy==cl;t=v['scores']['TEACHER'].argmax(1)==yy;p=v['scores'][mode].argmax(1)==yy;harm+=int((mask&t&~p).sum());rescue+=int((mask&~t&p).sum())
   tiny.append({'class':cl,'final_count':z[-1]['arms']['TEACHER']['counts'][cl],'teacher_final_correct':z[-1]['arms']['TEACHER']['correct'][cl],'method_final_correct':z[-1]['arms'][mode]['correct'][cl],'harm_stage_exposures':harm,'rescue_stage_exposures':rescue})
  gate=mean['macro']>=.5 and final['macro']>=.5 and final['tail']>=1 and mean['tail']>=-1e-8 and both>=4 and all(t['harm_stage_exposures']==0 for t in tiny);summary['comparisons_to_teacher'][mode]={'stage_delta_pp':d,'mean_delta_pp':mean,'final_delta_pp':final,'stages_macro_tail_nonnegative':both,'tiny':tiny,'performance_gate':gate}
  if mode=='JOINT':summary['main_gate']=gate
 if seal['RL_trained']:
  summary['JOINT_minus_SUP']={'mean_delta_pp':{k:summary['mean']['JOINT'][k]-summary['mean']['SUP'][k] for k in ['micro','macro','tail']},'final_delta_pp':{k:summary['final']['JOINT'][k]-summary['final']['SUP'][k] for k in ['micro','macro','tail']},'not_an_independent_GRPO_attribution_control':True}
 summary['conclusion']='COMPLETE_MAIN_GATE_POSITIVE_PILOT_ONLY' if summary['main_gate'] else ('COMPLETE_MAIN_GATE_NEGATIVE' if seal['RL_trained'] else 'COMPLETE_SUPERVISION_GATE_NEGATIVE');write(r/'ANALYSIS.json',summary)
 with (r/'CLASS_STAGE.csv').open('w') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['stage','arm','class','VAL_count','correct','recall'])
  for s in z:
   for mode,a in s['arms'].items():
    for cl in range(s['classes']):w.writerow([s['stage'],mode,cl,a['counts'][cl],a['correct'][cl],a['recall'][cl]])
 with (r/'GROUP_STAGE.csv').open('w') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['stage','arm','group','macro'])
  for s in z:
   for mode,a in s['arms'].items():
    for group,value in a['groups'].items():w.writerow([s['stage'],mode,group,value])
 lines=['# R97 frozen strong-SFT and gated TARS–RLAP pilot','',summary['conclusion'],'','Saved aggregates; TEST0. Teacher is borrowed full-history SFT, not a bounded-memory baseline or newly fitted arm. Main attribution controls deferred. All stage results and negative outcomes retained.','', '|mode|mean macro|final macro|final tail|','|---|---:|---:|---:|']
 for mode in modes:lines.append(f"|{mode}|{summary['mean'][mode]['macro']:.6f}|{summary['final'][mode]['macro']:.6f}|{summary['final'][mode]['tail']:.6f}|")
 lines+=['','If RL was not trained, no GRPO efficacy is reported. Historical teacher reference64.876782 final macro is not asserted bitwise identical on this hardware. Exact initial teacher preservation does not guarantee preservation after learned correction. Full class/stage/oldnew headtail, tiny harm denominators and cost receipts remain available. Native/profile costs are part of worker cost; no double counting. Final source/aggregate delivery follows local proxy and privacy rules; no automated monitor was created.'];(r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
