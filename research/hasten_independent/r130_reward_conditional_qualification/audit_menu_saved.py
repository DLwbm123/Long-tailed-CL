"""Same-component 2x2 arithmetic and prediction-denominator audit only."""
import csv,json,time
from pathlib import Path
import torch
from .core import write

def run(cfg):
 p=Path(cfg['output']);begin=time.time();cpu=time.process_time();rows=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()));y=[int(r['original_label']) for r in rows];ids=[i for i,c in enumerate(y) if c in range(5)];assert len(ids)==624
 saved=json.loads((p/'MENU_INTERACTION.json').read_text());by={(r['parameter'],r['menu']):r for r in saved['readouts']};assert len(by)==4;verified=[]
 for parameter in [0,1]:
  for menu in [0,1]:
   z=torch.load(p/f'MENU_{parameter}_{menu}.private.pt',map_location='cpu',weights_only=True);assert z['ids']==ids and len(z['predictions'])==624;correct=[0]*23;den=[0]*23
   for i,pred in zip(ids,z['predictions']):den[y[i]]+=1;correct[y[i]]+=int(pred==y[i])
   r=by[parameter,menu];assert correct==r['correct'] and den==r['denominator'];v=sum(correct[c]/den[c] for c in range(5))/5;assert abs(v-r['old_macro'])<1e-12;verified.append({'parameter':parameter,'menu':menu,'old_macro':v,'correct':correct[:5],'denominator':den[:5]})
 d={(r['parameter'],r['menu']):r['old_macro'] for r in verified};interaction=d[1,1]-d[1,0]-d[0,1]+d[0,0];assert abs(interaction-saved['interaction'])<1e-12
 proof={'passed':True,'readouts':verified,'interaction':interaction,'menu_at_initial_parameter':d[0,1]-d[0,0],'parameter_at_final_menu':d[1,1]-d[0,1],'total_path':d[1,1]-d[0,0],'responses_verified':2496,'new_model_generation_encoder_optimizer_calls':0,'CPU_body_seconds':time.process_time()-cpu,'wall_body_seconds':time.time()-begin,'retrospective_diagnostic_not_unique_causal_fraction_or_prompt_selection':True};write(p/'MENU_SAVED_VERIFICATION.json',proof)
