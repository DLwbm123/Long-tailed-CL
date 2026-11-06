"""Saved-only endpoint, byte memory, access audit and metric reconstruction."""
import json
from pathlib import Path
import torch,numpy as np
from memory import DTYPE,BUDGET_BYTES,STATS_BYTES
from common import write
from preflight import ARMS

def finite(x):
    if torch.is_tensor(x):assert torch.isfinite(x).all()
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite(v)

def run(c):
    r=Path(c['output']);s=json.loads((r/'LEARNING_SEAL.json').read_text());assert s['classifier_endpoints']==35 and s['formal_classifier_updates']==10500
    audit=json.loads((r/'MEMORY_AND_ACCESS_AUDIT.json').read_text());assert len(audit)==35 and all(a['old_views_from_own_previous_memory_only'] and a['strict_restore'] and a['memory_bytes']<=BUDGET_BYTES for a in audit)
    metrics=json.loads((r/'STAGE_METRICS.json').read_text());worst=0
    previous={}
    for t in range(5):
        C=sum(map(len,c['stages'][:t+1]));finite(torch.load(r/f's{t}'/'POLICY_AND_VALUE.private.pt',map_location='cpu',weights_only=False))
        for arm in ARMS:
            finite(torch.load(r/f's{t}'/f'{arm}_MODEL.private.pt',map_location='cpu',weights_only=False));p=r/f's{t}'/f'{arm}_MEMORY.private.bin';a=np.fromfile(p,dtype=DTYPE);assert p.stat().st_size+Path(str(p)+'.stats').stat().st_size<=BUDGET_BYTES;assert np.isfinite(a['feature']).all();assert len(set(a['label'].tolist()))==C
            assert set(a['parent'][a['view']>0])<=set(a['parent'][a['view']==0]);assert ((a['label']>=0)&(a['label']<C)).all()
            old=C-len(c['stages'][t]);oldkeys=set(zip(a['parent'][a['label']<old].tolist(),a['view'][a['label']<old].tolist()));assert oldkeys<=previous.get(arm,set());previous[arm]=set(zip(a['parent'].tolist(),a['view'].tolist()))
            stats=np.fromfile(str(p)+'.stats',dtype=[('count','<u4'),('birth','<i2'),('padding','<u2')]);assert (stats['count'][C:]==0).all() and (stats['birth'][C:]==-1).all();assert stats['count'][:C].tolist()==c['train_counts'][:C]

        pack=torch.load(r/f'VAL_s{t}.private.pt',map_location='cpu',weights_only=False);assert pack['global_seal_before_read'];y=pack['labels'];den=torch.bincount(y,minlength=C);tail=torch.tensor(c['train_counts'][:C])<250
        for arm,pred in pack['predictions'].items():
            rec=torch.bincount(y[y==pred],minlength=C).double()/den;values={'micro':float((y==pred).double().mean()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100) if tail.any() else None}
            for k,v in values.items():
                if v is not None:worst=max(worst,abs(v-metrics[t]['modes'][arm][k]));assert worst<1e-10
    write(r/'VERIFICATION.json',{'passed':True,'classifier_endpoints':35,'policy_value_endpoints':5,'all_saved_states_finite':True,'actual_memory_files_budget_and_parent_constraints_verified':True,'per_arm_access_and_restore_audit_passed':True,'saved_micro_macro_tail_reconstructed':True,'max_error_pp':worst,'VAL_after_global_seal':True,'TEST':0,'new_forward_optimizer':0,'saved_old_tokens_subset_own_previous':True,'unseen_statistics_absent':True})
