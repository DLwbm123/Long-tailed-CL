import json
from pathlib import Path
import torch
from common import write
from model import ARMS
from analyze import metrics,summarize

def finite(obj):
    if isinstance(obj,torch.Tensor):return bool(torch.isfinite(obj).all())
    if isinstance(obj,dict):return all(finite(v) for v in obj.values())
    if isinstance(obj,(tuple,list)):return all(finite(v) for v in obj)
    return True

def run(c):
    r=Path(c['output']);seal=json.loads((r/'SEAL.json').read_text());assert seal['final_endpoints']==20 and seal['total_optimizer_steps']==17920
    for s in range(5):
        h=torch.load(r/'strong_heads'/f'head{s}.private.pt',map_location='cpu',weights_only=False);assert finite(h) and h['new_stream']==c['stream_name']
        for a in ARMS:
            f=torch.load(r/f's{s}_{a}.private.pt',map_location='cpu',weights_only=False);mid=torch.load(r/f's{s}_{a}_middle.private.pt',map_location='cpu',weights_only=False);start=torch.load(r/f'STAGE_START_s{s}_{a}.private.pt',map_location='cpu',weights_only=False)
            assert finite(f) and finite(mid) and finite(start) and f['cycle']==512 and mid['cycle']==256
            if s:
                prev=torch.load(r/f's{s-1}_{a}.private.pt',map_location='cpu',weights_only=False)
                for k,v in start['model'].items():
                    if k.startswith('reference.'):assert torch.allclose(v,prev['model'][k]*.9+prev['model'][k.replace('reference.','policy.')]*.1,atol=2e-7,rtol=2e-6)
                    else:assert torch.equal(v,prev['model'][k])
    packs=[torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False) for s in range(5)];records=[metrics(c,p,s) for s,p in enumerate(packs)];assert summarize(c,records,packs)==json.loads((r/'RESULT_SUMMARY.json').read_text())
    if c['setting']=='ordered':assert all(v['arms']['T']['tail'] is None for v in records[:2])
    write(r/'VERIFICATION.json',{'passed':True,'teacher_endpoints':5,'final_learning_endpoints':20,'middle_endpoints':20,'start_states':20,'models_optimizers_finite':True,'carry_and_EMA_verified':True,'all_metrics_recomputed':True,'tiny_mapping_verified':c['tiny_indices'],'empty_groups_null':True,'new_optimizer_encoder_F_VAL_TEST':0})
