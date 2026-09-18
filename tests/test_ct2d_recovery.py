"""Retained score bytes must survive recovery; changed cached features must fail."""
import sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from resume_ct2d import Recovery,base

def run():
    with tempfile.TemporaryDirectory() as d:
        r=object.__new__(Recovery);r.private=Path(d);(r.private/'scores').mkdir()
        p=dict(dataset='ISIC',seed=1994,stream='U',order=list(range(8)))
        raw=np.random.default_rng(46001).normal(size=(5,2,768)).astype('float32')
        W=np.random.default_rng(46002).normal(size=(1536,8));val=dict(raw=raw,y=np.arange(5),rows=[dict(sample_id=str(i)) for i in range(5)])
        target=r.private/'scores/ISIC_1994_U_Q00.npz'
        np.savez(target,raw=base.joint(raw)@W,W=W,ids=np.array([str(i) for i in range(5)]),order=np.arange(8),y=np.arange(5))
        digest=base.sha(target);r.save_score(p,'Q00',W,val);assert base.sha(target)==digest
        raw[0,0,0]+=1
        try:r.save_score(p,'Q00',W,val)
        except AssertionError:pass
        else:raise AssertionError('Changed cache accepted')
        assert base.sha(target)==digest
    print('PASS: exact retained-score reuse; changed cache blocked; existing bytes preserved')
if __name__=='__main__':run()
