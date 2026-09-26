"""Production/reference parity including resume, CV and a final three-class task."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from tools.rfvila_reference import projections,normalize_rows,features,MomentBank,batch_ridge,select_lambda_cv
from tools.rfvila_math import tensor,transform,solve,select_cv,append,GRID

class ProductionChecks(unittest.TestCase):
    def test_production_parity_and_resume(self):
        rng=np.random.default_rng(781);y=np.repeat([6,2,8,1,4],[7,11,3,8,5]);a=normalize_rows(rng.normal(size=(len(y),5)));u=normalize_rows(rng.normal(size=(len(y),3)));ra,ru=projections(5,3,12,67101)
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for kind,ref,branch in [('linear','F.LIN',None),('linear','J.LIN',u),('relu_single','F.RF',None),('relu_joint','J.RF',u),('relu_split','J.SPLIT',u),('random_linear','J.RPLINEAR',u)]:
                x=transform(tensor(a,device),None if branch is None else tensor(u,device),tensor(ra,device),tensor(ru,device),kind)
                np.testing.assert_allclose(x.cpu(),features(a,u,ra,ru,ref),atol=1e-13)
                S=torch.zeros((x.shape[1],x.shape[1]),dtype=torch.float64,device=device);Q=torch.empty((x.shape[1],0),dtype=torch.float64,device=device);seen=[];counts=[]
                for c in [6,2]:S,Q=append(S,Q,seen,counts,c,x[tensor(y==c,device).bool()])
                with tempfile.TemporaryDirectory() as td:
                    p=Path(td)/'state.npz';np.savez(p,S=S.cpu(),Q=Q.cpu(),ids=seen,counts=counts)
                    with np.load(p) as z:S=tensor(z['S'],device);Q=tensor(z['Q'],device);seen=z['ids'].tolist();counts=z['counts'].tolist()
                for c in [8,1,4]:S,Q=append(S,Q,seen,counts,c,x[tensor(y==c,device).bool()])
                W,res=solve(S,Q,.001)
                np.testing.assert_allclose(W.cpu(),batch_ridge(x.cpu().numpy(),y,seen,.001),atol=1e-10)
                with self.assertRaises(ValueError):append(S,Q,seen,counts,6,x[:1])
                sel=y==6;sel|=y==2;xx=x[tensor(sel,device).bool()];yy=y[sel];com=np.array([str(i) for i in range(len(yy))])
                lam,log=select_cv(xx,yy,com,[6,2]);expected,elog=select_lambda_cv(xx.cpu().numpy(),yy,com,[6,2],GRID)
                self.assertEqual(lam,expected);np.testing.assert_allclose(log['losses'],elog['losses'],atol=1e-10)
                self.assertEqual(log['spectral_decompositions'],3)
                self.assertEqual(select_cv(xx,yy,np.array(['one' if v==6 else 'two' for v in yy]),[6,2])[0],.001)

if __name__=='__main__':unittest.main()
