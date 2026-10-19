"""Recount saved partial OOF evidence after authorized continuation failure; no solver, fit, or query access."""
import json, os, time
from collections import Counter
from pathlib import Path
import numpy as np

start = time.time()
root = Path(os.environ['RUN_OUTPUT'])
cfg = json.loads((root/'config.private.json').read_text())
status = json.loads((root/'status.private.json').read_text())
terminal = json.loads((root.parent/'resume_terminal.private.json').read_text())
audits = json.loads((root/'TRAINING_AUDIT.json').read_text())
assert status['state'] == 'FAILED' and terminal['returncode'] != 0
assert not (root/'FIT_SEAL.json').exists()
assert not list(root.glob('stage_*.private.pt')) and not (root/'results.json').exists()
index = np.load(root/'train_index.private.npz')
labels, groups, folds, weights, bounds = [index[k] for k in ['labels','groups','folds','weights','bounds']]
assert len(labels) == len(groups) == len(folds) == len(weights) == 68058
mapping = {}
for g, f in zip(groups.tolist(), folds.tolist()):
    assert mapping.setdefault(g, f) == f
for ci in range(20):
    m = labels == ci
    counts = Counter(groups[m].tolist())
    expected = np.array([1/(len(counts)*counts[g]) for g in groups[m]])
    assert np.max(np.abs(expected-weights[m])) < 1e-12
    assert abs(weights[m].sum()-1) < 1e-10

stages = []
for stage, saved in enumerate(audits):
    k = sum(cfg['task_sizes'][:stage+1]); n = bounds[k]
    o = np.load(root/f'OOF_{stage}.private.npz')
    p = np.load(root/f'problem_{stage}.private.npz')
    assert np.array_equal(o['labels'], labels[:n]) and np.array_equal(o['folds'], folds[:n])
    assert np.array_equal(o['groups'], groups[:n]) and np.array_equal(o['weights'], weights[:n])
    raw = o['CLS']; joint = o['joint']; w = weights[:n]; ys = labels[:n]
    assert raw.shape == joint.shape == (n,k) and np.isfinite(raw).all() and np.isfinite(joint).all()
    c = raw-raw.mean(1,keepdims=True)
    d = joint-raw; d -= d.mean(1,keepdims=True)
    target = np.eye(k)[ys]; target -= target.mean(1,keepdims=True)
    def risk(alpha, m=None):
        if m is None: m = np.ones(n,dtype=bool)
        f = c[m]+d[m]*alpha; f -= f.mean(1,keepdims=True)
        return float((((f-target[m])**2)*w[m,None]).sum()/(w[m].sum()*k))
    zero = risk(0.)
    alpha = np.array(saved['risk_limited_alpha']); scalar = saved['uniform_alpha']
    e = c-target; den = (d*d*w[:,None]).sum()
    u = float(np.clip(-(e*d*w[:,None]).sum()/den,0,1)) if den > 1e-14*w.sum()*k else 0.
    assert abs(u-scalar)<1e-9 and alpha.min()>=-1e-10 and alpha.max()<=1+1e-10
    err = max(abs(z-saved[key]) for z,key in [(zero,'OOF_CLS_risk'),(risk(u),'OOF_uniform_risk'),(risk(alpha),'OOF_limited_risk')])
    assert err<1e-10
    h = np.eye(k)-np.ones((k,k))/k
    q = ((d*w[:,None]).T@d)*h/(w.sum()*k)
    b = (e*d*w[:,None]).sum(0)/(w.sum()*k)
    assert np.max(np.abs(q-p['Q']))<1e-12 and np.max(np.abs(b-p['b']))<1e-12
    correct = raw.argmax(1)==ys; expected = []; skips = []; violations = []
    for fold in range(3):
        for ci in range(k):
            base = (folds[:n]==fold)&(ys==ci)
            for kind,m in [('class',base),('CLS_correct',base&correct)]:
                meta = dict(fold=fold,class_id=cfg['order'][ci],kind=kind,images=int(m.sum()),patients=len(np.unique(groups[:n][m])))
                if not m.any() or (kind=='CLS_correct' and meta['patients']<2): skips.append(meta); continue
                r0 = risk(0.,m); delta = risk(alpha,m)-r0; norm = delta/max(r0,1e-8)
                expected.append((meta,r0,delta,norm)); violations.append(norm)
    assert skips==saved['skipped_constraints'] and len(expected)==len(saved['constraints'])==len(p['Qs'])
    for (meta,r0,delta,norm),a in zip(expected,saved['constraints']):
        assert all(a[key]==val for key,val in meta.items())
        assert abs(r0-a['baseline_risk'])<1e-10 and abs(delta-a['risk_delta'])<1e-10 and abs(norm-a['normalized_delta'])<1e-7
    violation = max(0.,max(violations))
    assert abs(violation-saved['solver']['max_normalized_violation'])<1e-7
    accepted = bool(saved['solver']['success'] and violation<=cfg['feasibility_tolerance'])
    assert accepted == (stage<len(audits)-1)
    assert np.isfinite(p['Qs']).all() and np.isfinite(p['bs']).all()
    eigmin = min(float(np.linalg.eigvalsh(z).min()) for z in p['Qs'])
    zero_delta = float(np.max(np.einsum('i,gij,j->g',np.zeros(k),p['Qs'],np.zeros(k))+2*p['bs']@np.zeros(k)))
    assert zero_delta==0. and eigmin>=-1e-12
    stages.append(dict(stage=stage,classes=k,OOF_images=int(n),evidence_recount_passed=True,accepted_by_frozen_rule=accepted,uniform_alpha=u,limited_alpha_min=float(alpha.min()),limited_alpha_max=float(alpha.max()),OOF_CLS_risk=zero,OOF_uniform_risk=risk(u),OOF_limited_risk=risk(alpha),max_risk_recount_error=err,max_normalized_violation=violation,constraints=len(expected),skipped_constraints=len(skips),constraint_min_eigenvalue=eigmin,zero_is_feasible=True,solver=saved['solver']))

cost = status['costs']
assert cost['attempts']==2 and cost['new_training_images']==68058 and cost['new_validation_images']==cost['backbone_optimizer_updates']==0
assert cost['fold_head_attempts']==cost['fold_head_completed']==18 and cost['shrink_solver_calls']==4
assert terminal['pid']==878966 and str(terminal['start_ticks'])=='441194217'
assert not Path('/proc/'+str(terminal['pid'])).exists() and not Path('/proc/878965').exists()
assert terminal['elapsed_seconds']+terminal['prior_and_repair_wall_seconds']<=cfg['max_wall_seconds']
assert status['RSS_peak_bytes']<=cfg['host_RSS_budget_bytes'] and status['GPU_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes']
size = sum(p.stat().st_size for directory in [root,root.parent/'solver_deps'] for p in directory.rglob('*') if p.is_file())
assert size<=cfg['output_budget_bytes']
assert cost['continuation_new_training_images']==cost['continuation_new_validation_images']==0
assert audits[-1]['solver']['status']=='AlmostSolved' and not audits[-1]['solver']['success']
assert audits[-1]['solver']['absolute_gap']>cfg['solver_ftol']
assert json.loads((root/'failure_attempt1/status.private.json').read_text())['costs']['attempts']==1
result = dict(first_failure_preserved=True,continuation_new_encodes=0,status='PARTIAL_FAILURE_EVIDENCE_VERIFIED',scientific_experiment_complete=False,validation_comparison_available=False,training_patient_fold_consistency_and_weights_verified=True,stages=stages,budget_checks_passed=True,output_bytes_before_verification=size,feature_cache_bytes=(root/'train768.private.npy').stat().st_size,guardian_wall_seconds=terminal['elapsed_seconds'],cumulative_main_clock_seconds=status['wall_seconds'],continuation_main_wall_seconds=status['wall_seconds']-terminal['prior_and_repair_wall_seconds'],cumulative_guardian_and_repair_wall_seconds=terminal['prior_and_repair_wall_seconds']+terminal['elapsed_seconds'],main_peak_RSS_bytes=max(status['RSS_peak_bytes'],json.loads((root/'failure_attempt1/status.private.json').read_text())['RSS_peak_bytes']),main_peak_GPU_reserved_bytes=status['GPU_peak_reserved_bytes'],costs=cost,worker_and_guardian_retired=True,cache_retained=True,verification_wall_seconds=time.time()-start)
(root/'RESUME_FAILURE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
