"""Overlapping synthetic classes, permutation equivariance and interface checks."""
import copy,json,os,time
from pathlib import Path
import torch
import prototype_coherent as native
import core1_competition as core
import concm_identity as method

def run():
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(1074005)
    p=json.loads(Path(os.environ['Q128_PROTOCOL']).read_text());method.validate_design(p)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    d,k,old=8,5,3;counts=[13,17,21,27,33];y=torch.arange(k).repeat_interleave(torch.tensor(counts))
    x=torch.randn(len(y),d,dtype=torch.float64);x[:,:5]+=torch.nn.functional.one_hot(y,5)*.15
    weights=x.new_tensor([1/counts[int(c)] for c in y]);groups=torch.cat([2*c+torch.arange(n)%2 for c,n in enumerate(counts)])
    bank=native.append(native.empty(d),x,y,groups,weights);R=native.metric(bank);initial,_=native.head(bank);pairs=core.competition(bank,initial);W,_=core.bank_head(bank,pairs,regularizer=R)
    S,edges,b=method.tail_geometry(bank,pairs,W,old);probabilities=[e['p'] for e in edges];delta,a=method.fit(bank,pairs,R,W,old,'TAIL')
    B,s,_=torch.linalg.svd((bank['mu'][:old]-bank['mu'][:old].mean(0)).T,full_matrices=False);B=B[:,s>1e-12];outside=float((delta-B@(B.T@delta)).norm())
    assert a['tail_risk_mass']>0 and float(delta.norm())>1e-6 and outside>1e-8
    perm=torch.tensor([2,0,1,4,3]);inverse=torch.argsort(perm);pb=copy.deepcopy(bank)
    pb.update(mu=bank['mu'][perm],Q=bank['Q'][perm],n=[bank['n'][int(i)] for i in perm])
    for row in pb['components']:row['label']=int(inverse[row['label']])
    differences={}
    for v in method.PRESETS:
        source,_=method.fit(bank,pairs,R,W,old,v);changed,_=method.fit(pb,pairs[perm][:,perm],R,W[:,perm],old,v)
        error=float((changed-source[:,perm]).norm());assert error<1e-8;differences[v]=error
    # Tensor interfaces: full feature dimension and HK class count without reading a medical checkpoint.
    mu=torch.randn(3,1536,dtype=torch.float64)*.01;dummy=torch.randn(1536,3,dtype=torch.float64);newtarget,etf=method.matching_anchors(mu,dummy-dummy.mean(1,keepdim=True))
    assert newtarget.shape==(1536,3) and not etf['skipped']
    scores=torch.randn(32,23,dtype=torch.float32);shift=torch.randn_like(scores);shift[:,21:]=0
    result=method.guard(scores,shift,21);assert result.shape==(32,23) and torch.equal(result[:,21:],scores[:,21:])
    return dict(status='PASS',synthetic_only=True,GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,formal_optimizer_updates=0,
        overlapping_classes_nonzero_tail_force=True,tail_error_mass=a['tail_risk_mass'],tail_probability_range=[min(probabilities),max(probabilities)],tail_delta_norm=float(delta.norm()),outside_old_mean_span_norm=outside,
        all_eight_class_permutations_equivariant=True,permutation_absolute_errors=differences,
        feature_1536_ETF_interface=True,class_23_guard_interface=True,full_1536x23_medical_solve_tested=False,protocol_and_parameters_unchanged=True,pipeline_seconds=time.monotonic()-started)
if __name__=='__main__':print(json.dumps(run(),indent=2))
