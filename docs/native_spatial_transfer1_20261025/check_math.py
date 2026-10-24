"""One synthetic CPU spatial pooling, packed symmetry and score check."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from core import ARMS,groups,affine,numpy_affine,pack,unpack_numpy

def main(path):
    torch.set_num_threads(2);torch.manual_seed(74008);tick=time.time();z=torch.randn(20,197,7,dtype=torch.float64);gates=[]
    def check(name,error,tolerance):
        gates.append(dict(check=name,error=error,tolerance=tolerance,PASS=error<=tolerance));Path(path).write_text(json.dumps(dict(state='RUNNING',gates=gates),indent=2));assert error<=tolerance,name
    for arm in ARMS:
        f=affine(z,arm);n=numpy_affine(z.numpy(),arm);check(arm+'_operator',float(abs(f.numpy()-n).max()),1e-12);m=f.T@f;check(arm+'_pack',float(abs(unpack_numpy(pack(m).numpy(),len(m))-m.numpy()).max()),1e-12);rhs=f[:3].T;matrix=m+1e-4*torch.eye(len(m));h=torch.linalg.solve(matrix,rhs);ref=np.linalg.solve(matrix.numpy(),rhs.numpy());check(arm+'_score',float(abs(n@ref-(f@h).numpy()).max()),1e-9)
    q=groups(ARMS[-1]);r=groups(ARMS[2]);assert all(len(v)==49 for v in q+r) and sorted(sum(q,[]))==sorted(sum(r,[]))==list(range(196));out=dict(state='PASS',gates=gates,CPU_synthetic_token_rows=20,NIH_images=0,GPU_calls=0,optimizer_steps=0,real_native_models=0,wall_seconds=time.time()-tick);Path(path).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='gates'}))
if __name__=='__main__':main(sys.argv[1])
