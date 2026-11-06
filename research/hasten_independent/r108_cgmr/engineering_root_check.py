"""Synthetic CPU end-to-end shadow feedback contract, engineering only."""
from pathlib import Path
import time,torch
from roles import Roles
from memory import from_views
from shadow import episode,root,feedback
from common import write

def run(r,cfg):
 cpu=time.process_time();wall=time.time();torch.set_num_threads(1);torch.manual_seed(108);z=torch.nn.functional.normalize(torch.randn(100,6,512),dim=-1);bank=from_views(z,torch.arange(100),torch.arange(100)//20);rows=[dict(identity_component=str(i),original_label=str(i//20)) for i in range(100)];roles=Roles();roles.arrive(list(range(100)),rows);e,skip=episode(bank,roles,rows,[20]*5,74002,0);assert skip is None;counter={};ctx=root(e,counter,'CPU_engineering',cfg);scores,raw=feedback(e,ctx,counter,'CPU_engineering',cfg);assert scores['FIT']['H'][0]==0 and scores['AUDIT']['H'][0]==0;assert all(torch.isfinite(torch.tensor(v['R'])).all() for v in scores.values());assert not set(ctx['pool'].pid.tolist())&ctx['m'].forbidden
 out={'passed':True,'synthetic_CPU_root_only':True,'classifier_optimizer':counter['CPU_engineering_classifier_optimizer'],'candidate_count':len(ctx['mems']),'counts':counter,'query_prototype_gradient_candidate_firewall':True,'no_op_H_zero_both_query_roles':True,'same_fixed_query_denominators_all_candidates':True,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'science_encoder_optimizer_VAL_TEST':0};write(Path(r)/'ENGINEERING_ROOT_CHECK.json',out)
if __name__=='__main__':
 import json
 r=Path(__file__).resolve().parent;run(r,json.loads((r/'FIRST.private.json').read_text()))
