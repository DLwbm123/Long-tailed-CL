"""Before dispatch, recognize existing witnesses rather than re-solving them."""
import os,json,time,sys
from pathlib import Path
import numpy as np
import torch
from verify import verify
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);old=Path(cfg['reference_run']);start=time.time()
assert not (root/'DISPATCH_SEAL.json').exists() and not (root/'KNOWN_WITNESSES.json').exists()
seal=json.loads((root/'QUOTA_SEAL.json').read_text());k=seal['quotas'];y=np.load(root/'labels.private.npy')
trans=torch.load(old/'transforms.private.pt',weights_only=False);heads=torch.load(old/'2.heads.private.pt',weights_only=False)
G=np.load(root/'G.features.private.npy');C=np.load(root/'C.features.private.npy');U=np.load(root/'U.features.private.npy');WG=heads['G']['0.01'].numpy();WC=heads['C']['0.01'].numpy();d=1318
WGu=np.zeros((U.shape[1],6));WGu[:d]=(float(trans['C']['out_norm'][1])/float(trans['G']['out_norm'][1]))*WG[:d];WGu[C.shape[1]-1:-1]=WG[d:-1]
WGu[-1]=WG[-1]+((trans['C']['out_norm'][0][:d]-trans['G']['out_norm'][0][:d])/trans['G']['out_norm'][1]).numpy()@WG[:d]
WCu=np.zeros_like(WGu);WCu[:C.shape[1]-1]=WC[:-1];WCu[-1]=WC[-1]
assert np.array_equal((U@WGu).argmax(1),(G@WG).argmax(1)) and np.array_equal((U@WCu).argmax(1),(C@WC).argmax(1))
known={};checks={}
for a,X,W,origin in [('G',G,WG,'FROZEN_REFERENCE_G'),('C',C,WC,'FROZEN_REFERENCE_C'),('U',U,WCu,'EMBEDDED_FROZEN_REFERENCE_C')]:
    tick=time.time();check=verify(X,y,W,k);checks[a]=check
    if check['meets_all']:
        np.save(root/(a+'.reference.witness.private.npy'),W)
        known[a]=dict(space=a,classification='FEASIBLE_WITNESS',witness_origin=origin,verification=check,solver_not_run=True,solver_seconds=0,wall_seconds=time.time()-tick,full_family_infeasible_proven=False)
(root/'KNOWN_WITNESSES.json').write_text(json.dumps(known,indent=2,allow_nan=False))
(root/'PRE_DISPATCH_AUDIT.json').write_text(json.dumps(dict(reference_checks=checks,union_reference_argmax_matches=True,verification_wall_seconds=time.time()-start,test_used=False),indent=2,allow_nan=False))
print(json.dumps(dict(known_spaces=list(known),reference_checks=checks),indent=2),flush=True)
