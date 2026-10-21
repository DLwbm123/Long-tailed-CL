"""One paired-context diagnostic after scientific parity failure; never repairs results."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from common import settings,write

def main():
    cfg,out=settings();start=time.time();encodes=0
    assert json.loads((out/'VERIFICATION.json').read_text())['scientific_acceptance'] is False
    original=json.loads(Path(cfg['source_input']).read_text());sys.path.insert(0,str(Path(cfg['source'])/'tools'));sys.path.insert(1,str(Path(original['legacy_repo'])/'third_party/APART'))
    from run_multilabel import ApartFeatures
    from run_medical_v2 import transform
    from run_prototype_single import Images,manifests
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    query=np.load(out/'query.private.npz');w=np.load(out/'heads.private.npz')['stage1'].astype(np.float32);seen=cfg['order'][:10]
    x=query['x'];labels=query['labels'];ids=np.flatnonzero(np.isin(labels,seen));prefix=x[ids];scores=prefix@w;col=seen.index(6)
    margins=scores[:,col]-np.max(np.delete(scores,col,axis=1),axis=1);candidates=np.flatnonzero((labels[ids]==6)&(scores.argmax(1)==col));j=int(candidates[np.argmin(margins[candidates])]);target=int(ids[j])
    _,rows=manifests(original);oldrows=[r for r in rows if r['label'] in seen]
    assert rows[target]['relative_path']==oldrows[j]['relative_path']
    state=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False);model=ApartFeatures(original['legacy_repo'],original['weight'],20,'cuda:0',cfg['seed']);model.load_state_dict(state['model'],strict=True);model.requires_grad_(False).eval();del state
    records=[];vectors=[]
    try:
        for name,source,index,task in [('original_Task1',oldrows,j,1),('full300',rows,target,6)]:
            assert time.time()<cfg['hard_deadline_epoch']-60
            start_index=index//64*64;batch=source[start_index:start_index+64];dataset=Images(batch,cfg['images'],transform(False),cfg['seed']+task*100003)
            images=torch.stack([dataset[i][0] for i in range(len(dataset))]);assert encodes+len(images)<=128
            with torch.no_grad():z=model(images.cuda()).cpu().numpy()
            encodes+=len(z);v=z[index-start_index];vectors.append(v)
            changed=prefix.copy();changed[j]=v;sc=changed@w;value=sc[j];margin=float(value[col]-np.max(np.delete(value,col)))
            records.append(dict(context=name,batch_n=len(batch),candidate_prediction=int(seen[int(value.argmax())]),candidate_true_margin=margin,candidate_feature_delta_from_saved=float(abs(v-x[target]).max())))
        write(out/'BOUNDARY_AUDIT.json',dict(status='COMPLETE_DIAGNOSTIC_ONLY',records=records,paired_candidate_feature_max_error=float(abs(vectors[0]-vectors[1]).max()),saved_candidate_true_margin=float(margins[j]),diagnostic_val_encodes=encodes,formal_parity_still_failed=True,scientific_results_overwritten=False,worker_seconds=time.time()-start,GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved()))
    finally:write(out/'BOUNDARY_COSTS.json',dict(diagnostic_val_encodes=encodes,training_encodes=0,optimizer_updates=0,worker_seconds=time.time()-start))
if __name__=='__main__':main()
