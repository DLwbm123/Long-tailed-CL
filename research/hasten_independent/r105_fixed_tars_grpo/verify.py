"""Saved endpoint, prediction metrics and frozen-source verification, no new forward."""
import json
from pathlib import Path
import torch
from router import ARMS
from common import write
from binding import unchanged

def finite(x):
    if torch.is_tensor(x):assert torch.isfinite(x).all()
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite(v)

def run(c):
    r=Path(c['output']);seal=json.loads((r/'LEARNING_SEAL.json').read_text());assert seal['router_endpoints']==8 and seal['router_updates']==4096;unchanged(c)
    for arm in ARMS:
        for step in [0,256,512]:finite(torch.load(r/f'{arm}_step{step}.private.pt',map_location='cpu',weights_only=False))
        trace=torch.load(r/f'{arm}_POLICY_TRACE.private.pt',map_location='cpu',weights_only=False);assert len(trace)==128;finite(trace)
        assert all(torch.allclose(x['old_probabilities'].sum(-1),torch.ones(128,dtype=torch.double)) for x in trace)
        assert json.loads((r/f'{arm}_TRAIN_DIAGNOSTICS.json').read_text())['strict_restore']
    pack=torch.load(r/'VAL_predictions_s0.private.pt',map_location='cpu',weights_only=False);finite(pack);assert pack['global_seal_before_read'];y=pack['labels'];den=torch.bincount(y,minlength=23);tail=torch.tensor(c['train_counts'])<250;record=json.loads((r/'RESULT_SUMMARY.json').read_text())['final'];worst=0
    assert len(y)==1698 and len(pack['predictions'])==12
    for mode,pred in pack['predictions'].items():
        cc=torch.bincount(y[y==pred],minlength=23);rec=cc.double()/den;values={'micro':float((y==pred).double().mean()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100)}
        for k,v in values.items():worst=max(worst,abs(v-record[mode][k]));assert worst<1e-10
    write(r/'VERIFICATION.json',{'passed':True,'router_final_endpoints':8,'all_step0_256_512_states_finite':True,'private_true_old_traces_saved':True,'frozen_producers_unchanged':True,'all_12_micro_macro_tail_recomputed_from_saved_predictions':True,'max_error_pp':worst,'VAL_after_global_seal':True,'TEST':0,'new_encoder_optimizer_rows':0})
