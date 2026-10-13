"""Full 1536x23 synthetic CPU preparation and frozen/shared equivalence."""
import copy,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
import prototype_coherent as native
import module_additions as modules
import core1_competition as core
import intervention_gain as method
from intervention_gain_production import fit_all
from check_intervention_gain import run as mathematical_check

def run(protocol):
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(74002)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    p=json.loads(Path(protocol).read_text());method.validate_design(p['design_parameters'])
    scratch=Path(os.environ['TMPDIR'])/'design_check.json';scratch.write_text(json.dumps(p['design_parameters']))
    oracle=mathematical_check(scratch);scratch.unlink()
    results=[]
    for d,k,old in [(8,5,3),(1536,23,21)]:
        counts=[3 if i==1 else 48 for i in range(k)];y=torch.cat([torch.full((n,),c) for c,n in enumerate(counts)])
        features=torch.randn(len(y),d,dtype=torch.float64)*.03+F.one_hot(y,d);weights=features.new_tensor([1/counts[int(c)] for c in y]);groups=2*y+torch.arange(len(y))%2
        bank=native.append(native.empty(d),features,y,groups,weights)
        if d==1536:
            mu=torch.eye(d,dtype=torch.float64)[:k]*.2;mu[:,-1]=1.
            covariance=torch.eye(d,dtype=mu.dtype)*.05;counts=[2048]*k;counts[1]=3
            moments=[];components=[]
            for c in range(k):
                S=covariance.clone()
                if c==1:S.zero_();S[0,0]=S[1,1]=.05
                moments.append(torch.outer(mu[c],mu[c])+S);components.append(dict(label=c,center=mu[c],count=counts[c],mass=mu.new_tensor(1.),radius=S.trace()))
            bank=dict(mu=mu,Q=torch.stack(moments),n=counts,components=components)
        before=copy.deepcopy(bank);initial,_=native.head(bank);pairs=core.competition(bank,initial)
        for control in ('base','local_transport'):
            R,_=modules.metric(bank,control,'HK',list(range(k)),None if control=='base' else torch.randn(k,4,d,dtype=features.dtype));W,qa=core.bank_head(bank,pairs,regularizer=R)
            output=fit_all(bank,pairs,R,W,old,W.float());heads={}
            for v,(delta,audit) in output.items():
                bound=1e-8+torch.finfo(torch.float32).eps*delta[:,:old].abs().sum(1)
                assert torch.isfinite(delta).all() and not delta[:,old:].any() and not (delta[:,:old].sum(1).abs()>bound).any() and audit['relative_residual']<=1e-8
                meta=dict(dataset='HK',control=control,control_source='synthetic',source='synthetic',order=list(range(k)),seen=list(range(k)),task=6,seed=74002)
                state=method.pack(delta,old,meta,v);assert torch.equal(method.restore(state,old,meta,delta.shape,v),delta.float())
                if d==8:
                    frozen,a=method.fit(bank,pairs,R,W,old,v,W.float());torch.testing.assert_close(delta,frozen,atol=1e-12,rtol=1e-12)
                if d==1536 and v in ('GAIN','GAIN_NO_BORROW'):assert audit['proposal_norm']>1e-10 and audit['iterations']>0 and audit['preselection_delta_norm']>0
                heads[v]=dict(nonzero=bool(delta.any()),chosen_scale=audit.get('chosen_scale'),relative_residual=audit['relative_residual'],checkpoint_roundtrip=True,proposal_norm=audit.get('proposal_norm'),direction_iterations=audit.get('iterations'))
            first=fit_all(bank,pairs,R,W,0,W.float());assert all(not z[0].any() for z in first.values())
            assert torch.equal(bank['Q'],before['Q']) and torch.equal(bank['mu'],before['mu']) and bank['n']==before['n']
            results.append(dict(dimension=d,classes=k,old_classes=old,control=control,native_residual=qa['relative_residual'],heads=heads,shared_math_equals_frozen_helper=d==8,native_statistics_unchanged=True))
            (Path(os.environ['TMPDIR'])/'CPU_PROGRESS.private.json').write_text(json.dumps(results,indent=2))
            del R,W,output,first
        del features,bank,before,initial,pairs
    return dict(status='PASS',synthetic_only=True,full_high_dimensional_cpu=True,dimensions=[1536,23],medical_images_read=0,medical_checkpoint_reads=0,medical_val_reads=0,GPU_processes=0,formal_optimizer_updates=0,mathematical_check=oracle,cases=results,pipeline_seconds=time.monotonic()-started)

if __name__=='__main__':print(json.dumps(run(os.environ['Q128_PROTOCOL']),indent=2,allow_nan=False))
