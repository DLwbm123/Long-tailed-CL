"""All saved group coefficients; never interpret scalar shares as gradient effects."""
import json,time
from pathlib import Path
import torch
from .core import ARMS,write

def run(cfg):
 p=Path(cfg['output']);start=time.time();cpu=time.process_time();rows=[]
 for arm in ARMS:
  r={'arm':arm,'PG_component_actually_used':'a' if arm=='RAPO' else 'none' if arm=='CE_OLD_COND' else 'ac','queries':0,'responses':0,'correct_responses':0,'all_wrong_queries':0,'all_correct_queries':0,'mixed_queries':0,'all_wrong_queries_with_nonconstant_retention':0,'wrong_responses_with_positive_total_advantage':0,'correct_responses_with_negative_total_advantage':0,'wrong_responses_with_positive_actual_PG_advantage':0,'total_advantage_L1':0.,'accuracy_advantage_L1':0.,'retention_advantage_L1':0.,'maximum_decomposition_residual':0.}
  for k in range(32):
   z=torch.load(p/f'ROLLOUT_{arm}_{k:03d}.private.pt',map_location='cpu',weights_only=True);b=z['b'];a=z['a'];ac=z['ac'];ar=z['ar'];h=z['h'];n=b.sum(-1);r['queries']+=len(b);r['responses']+=b.numel();r['correct_responses']+=int(b.sum());r['all_wrong_queries']+=int((n==0).sum());r['all_correct_queries']+=int((n==8).sum());r['mixed_queries']+=int(((n>0)&(n<8)).sum());r['all_wrong_queries_with_nonconstant_retention']+=int(((n==0)&((h.max(-1).values-h.min(-1).values)>0)).sum());r['wrong_responses_with_positive_total_advantage']+=int(((b==0)&(a>1e-7)).sum());r['correct_responses_with_negative_total_advantage']+=int(((b==1)&(a<-1e-7)).sum());actual=a if arm=='RAPO' else torch.zeros_like(a) if arm=='CE_OLD_COND' else ac;r['wrong_responses_with_positive_actual_PG_advantage']+=int(((b==0)&(actual>1e-7)).sum());r['total_advantage_L1']+=float(a.abs().sum());r['accuracy_advantage_L1']+=float(ac.abs().sum());r['retention_advantage_L1']+=float(ar.abs().sum());r['maximum_decomposition_residual']=max(r['maximum_decomposition_residual'],float((a-ac-ar).abs().max()))
  assert r['queries']==256 and r['responses']==2048 and r['all_wrong_queries']+r['all_correct_queries']+r['mixed_queries']==256;rows.append(r)
 write(p/'SAVED_REWARD_SUMMARY.json',{'rows':rows,'new_model_calls':0,'new_optimizer_updates':0,'total_scientific_queries':1280,'total_scientific_responses':10240,'threshold_for_advantage_sign_numerical_reporting':1e-7,'scalar_L1_or_group_fractions_are_not_parameter_gradients_or_VAL_effects':True,'total_advantage_in_B_C_D_E_is_diagnostic_not_its_PG_objective':True,'CPU_body_seconds':time.process_time()-cpu,'wall_body_seconds':time.time()-start})
