"""Local synthetic CPU tests, no Torch, medical manifests or checkpoints."""
import json
import math
import time
import numpy as np
import gain_calibration_plan as plan


def reject(f):
    try:f()
    except ValueError:return
    raise AssertionError('Invalid scope accepted')


def run():
    start = time.monotonic(); rng = np.random.default_rng(74002)
    E = 3656448; alpha = .01
    for n in (3,6,23,25,34,85,736,3263):
        q = plan.feasible_support(n, n, alpha, E, .01, 16.)
        req = q['minimum_independent_n']
        assert plan.feasible_support(n, req, alpha, E, .01, 16.)['optimistic_sample_feasible']
        assert not plan.feasible_support(n, req-1, alpha, E, .01, 16.)['optimistic_sample_feasible']
    small = plan.feasible_support(25,25,alpha,E,.01,16.)
    assert not small['optimistic_sample_feasible'] and small['minimum_independent_n'] == 3338
    empty = {key:set() for key in ('encoder','head','bank','ray','policy')}
    good = plan.audit_independence({'hold1','hold2'},empty,'train_only',True); assert good['independent']
    for stage in empty:
        bad = dict(empty, **{stage:{'hold1'}})
        assert not plan.audit_independence({'hold1','hold2'},bad,'train_only',True)['independent']
    assert not plan.audit_independence({'hold1'},empty,'train_only',False)['independent']
    for split in ('val','test','reserved'):
        reject(lambda split=split:plan.audit_independence({'x'},empty,split,True))
    reject(lambda:plan.audit_independence({'x'},{'head':set()},'train_only',True))
    rows = [dict(group=f'{c}_{i}',label=c,split='train_only') for c in range(3) for i in range(15)]
    fit,held = plan.prospective_split(rows,74002); assert len(held)==9 and len(fit)==36
    assert plan.prospective_split(rows,74002)==(fit,held)
    exposure = {key:{r['group'] for r in fit} for key in empty}
    assert plan.audit_independence({r['group'] for r in held},exposure,'train_only',True)['independent']
    # Task-end inclusion of meta in the bank/head destroys calibration independence.
    exposure['bank'] = {r['group'] for r in rows}
    assert not plan.audit_independence({r['group'] for r in held},exposure,'train_only',True)['independent']
    reject(lambda:plan.prospective_split([dict(group='same',label=0,split='train_only')],74002))
    reject(lambda:plan.prospective_split([dict(group='a',label=0,split='val')],74002))
    # Identical separate moments do not identify the missing before/after cross moment.
    x = np.array([[-1.,0.],[1.,0.]])
    for y in (x,-x):
        np.testing.assert_array_equal(y.mean(0),x.mean(0));np.testing.assert_array_equal(y.T@y/2,x.T@x/2)
    stable = plan.paired_score_moments(x,x,np.eye(2),np.zeros(2))
    moved = plan.paired_score_moments(x,-x,np.eye(2),np.zeros(2))
    assert stable['second'].trace()==0. and moved['second'].trace()==4.
    # Full-dimensional paired projection and all four original quantized steps.
    d,k,old,n = 1536,23,21,32
    before = rng.normal(size=(n,d)); after = before+rng.normal(0,.01,size=(n,d)); shift=(after-before).mean(0)
    W = rng.normal(0,.02,size=(d,k)).astype(np.float32); ray = np.zeros_like(W)
    ray[:,:old]=rng.normal(0,.01,size=(d,old)).astype(np.float32);ray[:,:old]-=ray[:,:old].mean(1,keepdims=True)
    P = np.concatenate((W,ray),1).astype(np.float64)
    drift = plan.paired_score_moments(before,after,P,shift)
    assert drift['second'].shape==(46,46) and np.isfinite(drift['second']).all()
    assert np.linalg.eigvalsh(drift['centered_covariance']).min() > -1e-12
    changes = {step:after@(step*ray).astype(np.float32).astype(np.float64) for step in plan.STEPS}
    off = after@W; labels=rng.integers(0,k,size=n)
    def guard(scores, delta):
        z=scores+delta;raw=z[:,:old];rawmax=raw.max(1,keepdims=True);oldmax=scores[:,:old].max(1,keepdims=True)
        z[:,:old]=np.where(rawmax>oldmax,oldmax-(rawmax-raw),raw);return z
    counts = plan.candidate_counts(off,changes,labels,guard)
    assert all(sum(v['n'] for v in row.values())==n for row in counts.values())
    assert all(v['rescued']+v['harmed']<=v['n'] for row in counts.values() for v in row.values())
    assert all(v['harmed']==0 for row in counts.values() for c,v in row.items() if c>=old)
    for step in plan.STEPS:np.testing.assert_array_equal(changes[step],after@(step*ray).astype(np.float32).astype(np.float64))
    reject(lambda:plan.candidate_counts(off,{.1:changes[.125]},labels,guard))
    # Repeated image exposures never inflate independent units.
    repeated = ['personA']*50+['personB']*50
    assert plan.audit_independence(repeated,empty,'train_only',True)['calibration_units']==2
    folds=plan.diagnostic_folds(repeated,74002)
    assert len(folds)==2 and set(folds.values())=={0,1}
    assert plan.diagnostic_folds(repeated,74002)==folds
    leaked=dict(empty,encoder=set(repeated))
    assert not plan.audit_independence(repeated,leaked,'train_only',True)['independent']
    reject(lambda:plan.diagnostic_folds(['same']*20,74002))
    return dict(status='PASS',synthetic_only=True,GPU_processes=0,medical_images_read=0,
                medical_checkpoint_reads=0,medical_val_reads=0,optimizer_updates=0,
                policy_updates=0,test_accessed=False,unchanged_direction_and_steps=True,
                complete_exposure_lineage_checked=True,encoder_leakage_refused=True,
                task_end_meta_inclusion_refused=True,val_test_reserved_refused=True,
                external_train_only_valid_fixture=True,prospective_group_split_checked=True,
                repeated_exposures_not_independent_units=True,moment_nonidentifiability_oracle=True,
                diagnostic_two_folds_do_not_claim_classifier_independence=True,
                native_dimension=1536,classes=23,joint_score_dimension=46,
                paired_score_moments_finite_PSD=True,candidates_quantization_exact=True,
                all_candidate_paired_counts_checked=True,new_class_guard_checked=True,
                minimum_support_threshold_oracle=True,original_risk_cap_unmodified=True,
                medical_independent_calibration_ready=False,pipeline_seconds=time.monotonic()-start)


if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
