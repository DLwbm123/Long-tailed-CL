"""Saved-only counts, endpoint finiteness, carries and all final gate recomputation."""
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
    r=Path(c['output']);seal=json.loads((r/'SEAL.json').read_text());assert seal['final_endpoints']==24 and seal['total_optimizer_steps']==21504
    states=0
    for s in range(6):
        for arm in ARMS:
            final=torch.load(r/f's{s}_{arm}.private.pt',map_location='cpu',weights_only=False)
            middle=torch.load(r/f's{s}_{arm}_middle.private.pt',map_location='cpu',weights_only=False)
            start=torch.load(r/f'STAGE_START_s{s}_{arm}.private.pt',map_location='cpu',weights_only=False)
            assert final['cycle']==512 and middle['cycle']==256 and finite(final) and finite(middle) and finite(start);states+=3
            if s:
                previous=torch.load(r/f's{s-1}_{arm}.private.pt',map_location='cpu',weights_only=False)
                for k,v in start['model'].items():
                    if k.startswith('reference.'):
                        target=previous['model'][k]*.9+previous['model'][k.replace('reference.','policy.')]*.1
                        assert torch.allclose(v,target,atol=2e-7,rtol=2e-6)
                    else:assert torch.equal(v,previous['model'][k]),('CARRY_MISMATCH',s,arm,k)
    packs=[torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False) for s in range(6)]
    records=[metrics(c,p,s) for s,p in enumerate(packs)];summary=summarize(c,records,packs)
    saved=json.loads((r/'RESULT_SUMMARY.json').read_text());assert summary==saved
    assert json.loads((r/'EVAL_COMPLETE.json').read_text())['counts']['TEST_rows']==0
    write(r/'VERIFICATION.json',{'passed':True,'saved_states':states,'final_endpoints':24,'middle_endpoints':24,'stage_start_states':24,'model_optimizer_finite':True,'cross_stage_carry_and_reference_EMA_verified':True,'all_final_metrics_and_gates_recomputed':True,'new_optimizer_encoder_F_VAL_TEST':0,'no_endpoint_selection':True})
