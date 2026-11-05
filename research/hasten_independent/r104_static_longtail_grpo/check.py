"""Static-task semantic checks; established R103 objective tests are not repeated."""
import json
from pathlib import Path
import torch
from router import Router,features
from objectives import objective
from grpo_reference import old_tail_penalty
from common import write

def run(root):
    torch.set_num_threads(1);torch.manual_seed(74002)
    prior=json.loads((Path(root)/'REFERENCE_SOFTWARE_CHECK.json').read_text());assert prior['status']=='PASS'
    B,C=6,23;ss=torch.randn(B,3,C);z=torch.nn.functional.normalize(torch.randn(B,3,512),dim=-1)
    birth=torch.zeros(C,dtype=torch.long);phi=features(ss,z,torch.arange(1,C+1),birth,0)
    assert torch.equal(phi[:,:,[9,12]],torch.ones(B,3,2)) and torch.equal(phi[:,:,[10,13]],torch.zeros(B,3,2))
    assert phi.shape==(B,3,19);rm=Router();assert sum(p.numel() for p in rm.parameters())==672
    correct=torch.tensor([[1,0,1],[0,1,0],[1,1,0],[0,1,1],[1,0,0],[0,0,0]])
    protected=torch.zeros(B,dtype=torch.bool);logits=rm(phi);old=logits.double().softmax(-1).detach();actions=torch.tensor([[0,1,2,0,1,2,0,1]]*B)
    assert float(old_tail_penalty(logits,correct,protected,eligible_class_count=C,protected_class_count=0))==0
    loss=objective('GRPO_CF',logits,old,correct,protected,C,0,actions);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in rm.parameters())
    out={'status':'PASS','all_23_classes_visible_at_once':True,'new_age_features_constant':True,'empty_old_tail_penalty_zero':True,'same_19D_672_parameter_router':True,'static_objective_finite_gradient':True,'established_reference_tests_reused_from':'R103 source4379501, SOFTWARE_CHECK passed; no numerical algorithm change','formal_encoder_optimizer_VAL_TEST':0,'CPU_seconds':__import__('time').process_time()}
    write(Path(root)/'SOFTWARE_CHECK.json',out);return out
if __name__=='__main__':run(Path(__file__).resolve().parent)
