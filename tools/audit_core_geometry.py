"""Read saved statistical tensors on CPU; export scalar geometry, never model inference."""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F


def cosine(a, b):
    den = float(a.norm()*b.norm())
    return None if den <= 1e-20 else float(a.dot(b)/den)


def run(config):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU statistics only')
    torch.set_num_threads(2)
    begun = time.monotonic()
    protocol = config['protocol']
    rows, stages = [], []
    for job in config['jobs']:
        name, dataset, control = job['job_name'], job['name'], job['control']
        d = protocol['data'][dataset]
        parent = Path(job['parent_root'])
        pc = json.loads((parent/'CONFIG.private.json').read_text())
        receipt = json.loads((parent/'DELIVERY.private.json').read_text())
        provenance = protocol['controls'][control]
        assert parent.name == provenance['round'] and pc['source_commit'] == provenance['source']
        assert receipt['commit'] == provenance['delivery'] and receipt['nas_saved']
        original = next(x for x in pc['jobs'] if x['job_name'] == name)
        assert original['method'] == 'static_pc' and original['module'] == control and original['seed'] == 74002
        assert original['order'] == d['order'] and original['task_sizes'] == d['task_sizes']
        assert Path(original['output']).resolve() == parent.resolve()/'runs'/name
        previous_mu = previous_w = previous_regions = None
        for task in range(1, len(d['task_sizes'])+1):
            path = Path(original['output'])/f'stage_{task}.pt'
            frozen = config['fingerprints'][name][task-1]
            stat = path.stat()
            assert dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns) == frozen
            st = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
            seen = d['order'][:sum(d['task_sizes'][:task])]
            old = sum(d['task_sizes'][:task-1])
            assert st['seen'] == seen and st['steps'] == d['cumulative_steps'][task-1] and st['module'] == control
            bank = st['bank']; mu, Q = bank['mu'].double(), bank['Q'].double()
            W = st['head'].double(); k, dim = mu.shape
            assert dim == 1536 and W.shape == (dim, k) and len(bank['n']) == k
            assert bank['n'] == [d['train_counts'][str(c)] for c in seen]
            assert torch.isfinite(mu).all() and torch.isfinite(Q).all() and torch.isfinite(W).all()
            P = torch.stack([F.normalize(c['center'].double(), dim=0)*math.sqrt(float(c['mass'])) for c in bank['components']], 1)
            regions = st['module_memory']
            assert (regions is None) == (control == 'base')
            M = None if regions is None else F.normalize(regions.double(), dim=-1).reshape(-1, dim).T/2
            E = P if M is None else torch.cat([P, M], 1)
            eye = torch.eye(dim, dtype=torch.float64)
            R = eye-E@torch.linalg.solve(torch.eye(E.shape[1], dtype=torch.float64)+E.T@E, E.T)
            R = (R+R.T)/2
            G = Q.sum(0)/k+.001*R
            factor = torch.linalg.cholesky(G)
            native = torch.cholesky_solve(mu.T/k, factor)
            wc = W-W.mean(1, keepdim=True)
            mc = mu-mu.mean(0)
            nc = native-native.mean(1, keepdim=True)
            U = None
            if M is not None:
                U, singular, _ = torch.linalg.svd(M, full_matrices=False)
                U = U[:, singular > torch.finfo(M.dtype).eps*max(M.shape)*float(singular.max())]
            # This cached evidence was produced before val; no encoder or manifest is opened here.
            evidence = torch.load(Path(config['evidence_root'])/'statistics'/name/f'EVIDENCE_{task}.private.pt', map_location='cpu', weights_only=False)
            assert evidence['metadata'] == dict(source_commit=protocol['source_for_analysis'], job=name, task=task, sidecar_source=protocol['previous_sidecar_source'])
            for i, label in enumerate(seen):
                n = bank['n'][i]
                variance = max(0., float(Q[i].diagonal().sum()-mu[i].square().sum()))
                norm = float(mu[i].norm())
                current = evidence['distributions'][i]
                assert current['label'] == label and current['n'] == n
                actual_scores = torch.as_tensor(current['mean'][:k], dtype=torch.float64)
                bank_scores = mu[i]@W
                residual = float((actual_scores-bank_scores).norm())
                others = [j for j in range(k) if j != i]
                training = evidence['counts']['C0125'][i]
                row = dict(job=name, dataset=dataset, control=control, task=task, label=label,
                           role='old' if i < old else 'new', train_n=n, few_shot_descriptive=n < 20,
                           prototype_is_transported_estimate=i < old, raw_weight_norm=float(W[:, i].norm()),
                           discriminative_weight_norm=float(wc[:, i].norm()), prototype_norm=norm,
                           discriminative_prototype_norm=float(mc[i].norm()), raw_weight_prototype_cosine=cosine(W[:, i], mu[i]),
                           discriminative_weight_prototype_cosine=cosine(wc[:, i], mc[i]),
                           discriminative_weight_native_ridge_cosine=cosine(wc[:, i], nc[:, i]),
                           mean_variance_trace_proxy=variance/n if n > 1 else None,
                           variance_estimation_identified=n > 1, mean_relative_uncertainty_proxy=variance/n/max(norm*norm, 1e-30) if n > 1 else None,
                           current_train_bank_score_residual_norm=residual,
                           current_train_bank_score_relative_residual=residual/max(float(actual_scores.norm()), 1e-30),
                           current_train_mean_score_margin=float(actual_scores[i]-actual_scores[others].max()),
                           transported_bank_mean_score_margin=float(bank_scores[i]-bank_scores[others].max()),
                           retrospective_train_correct=training['correct_off'], retrospective_train_n=training['n'],
                           previous_stage_prototype_cosine=None if i >= old else cosine(previous_mu[i], mu[i]),
                           previous_stage_weight_cosine=None if i >= old else cosine(previous_w[:, i], W[:, i]),
                           region_center_mean_norm=None, region_weight_span_fraction=None, region_prototype_span_fraction=None,
                           previous_region_cosine_mean=None)
                if regions is not None:
                    row.update(region_center_mean_norm=float(regions[i].norm(dim=1).mean()),
                               region_weight_span_fraction=float((U.T@wc[:, i]).square().sum()/wc[:, i].square().sum().clamp_min(1e-30)),
                               region_prototype_span_fraction=float((U.T@mc[i]).square().sum()/mc[i].square().sum().clamp_min(1e-30)))
                    if i < old:
                        row['previous_region_cosine_mean'] = sum(cosine(a, b) for a,b in zip(previous_regions[i], regions[i]))/4
                rows.append(row)
            stages.append(dict(job=name, task=task, old_classes=old, new_classes=k-old,
                               common_weight_norm=float(W.mean(1).norm()*math.sqrt(k)), full_weight_norm=float(W.norm()),
                               discriminative_weight_norm=float(wc.norm()),
                               region_span_rank=None if U is None else U.shape[1],
                               noncompetition_to_competition_relative_difference=float((native-W).norm()/W.norm().clamp_min(1e-30))))
            previous_mu, previous_w = mu.clone(), W.clone()
            previous_regions = None if regions is None else regions.clone()
            assert dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns) == frozen
            del st, evidence, Q, P, E, R, G, factor, native, wc, nc, M, U
    assert len(stages) == 18 and len(rows) == 252
    return dict(status='SAVED_STATISTICS_AUDIT_COMPLETE', class_rows=rows, stages=stages,
                saved_checkpoint_reads=18, saved_training_evidence_reads=18,
                encoder_forward_calls=0, medical_image_reads=0, val_manifest_reads=0, test_accessed=False,
                optimizer_updates=0, GPU_processes=0, model_or_statistical_matrices_exported=False,
                epoch_geometry_reconstructable=False, old_actual_feature_direction_available=False,
                pipeline_seconds=time.monotonic()-begun)


if __name__ == '__main__':
    print(json.dumps(run(json.load(sys.stdin)), allow_nan=False))
