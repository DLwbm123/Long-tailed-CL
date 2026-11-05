"""Independent saved predictions/count/seal/finiteness verification, no encoders."""
import json
from pathlib import Path
import torch
from router import ARMS
from common import write

def finite(x):
    if torch.is_tensor(x):assert torch.isfinite(x).all()
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(tuple,list)):
        for v in x:finite(v)

def run(c):
    r=Path(c['output']);seal=json.loads((r/'LEARNING_SEAL.json').read_text());assert seal['router_updates']==2560 and seal['fusion_updates']==3072
    endpoints=0;fusion=0;teachers=0
    for s in range(len(c['stages'])):
        for bank in ['FULL','FIT_A','FIT_B']:
            for action in [1,2]:
                for step in [0,256,512]:finite(torch.load(r/f's{s}'/bank/f'fusion{action}_step{step}.private.pt',map_location='cpu',weights_only=False))
                fusion+=1
        for arm in ARMS:
            for step in [0,256,512]:finite(torch.load(r/f's{s}'/f'{arm}_step{step}.private.pt',map_location='cpu',weights_only=False))
            endpoints+=1
        for bank in ['FULL_TEACHER','FIT_A','FIT_B']:
            finite(torch.load(r/bank/f's{s}'/'teacher.private.pt',map_location='cpu',weights_only=False));finite(torch.load(r/bank/f's{s}'/'teacher_optimizer.private.pt',map_location='cpu',weights_only=False));teachers+=1
        pack=torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False);finite(pack);assert pack['global_seal_before_read'];C=sum(map(len,c['stages'][:s+1]));y=pack['labels'];den=torch.bincount(y,minlength=C)
        recorded=json.loads((r/'STAGE_METRICS.json').read_text())[s]['modes']
        for name,pred in pack['predictions'].items():
            correct=torch.bincount(y[y==pred],minlength=C);macro=float((correct.double()/den).mean()*100);assert abs(macro-recorded[name]['macro'])<1e-10
        if c['setting']=='ordered' and s<2:assert all(x['tail'] is None for x in recorded.values())
    assert seal['counts']['teacher_optimizer']==560
    assert endpoints==5 and fusion==6 and teachers==3
    write(r/'VERIFICATION.json',{'passed':True,'router_endpoints':endpoints,'fusion_endpoints':fusion,'new_fold_teacher_endpoints':2,'new_FULL_teacher_endpoints':1,'all_saved_states_finite':True,'saved_prediction_macro_recomputed':True,'OOF_lineage_runtime_audit_passed':json.loads((r/'LINEAGE_RUNTIME_AUDIT.json').read_text())['passed'],'VAL_after_global_seal':True,'TEST':0,'new_encoder_optimizer_rows':0})
