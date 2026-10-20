"""One new signed-p3 candidate; reuse all historical control scores."""
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from core import signed_pool, concentration, self_check


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main(cfg):
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(cfg['output'])
    root.mkdir(exist_ok=False)
    base = Path(cfg['remote_base'])
    common = module(base / 'matched_heads1_20261018/core.py', 'moment_helpers')
    data = module(base / 'case_hierarchical1_20261018/data.py', 'data_helpers')
    df = module(base / 'joint_controls1_20261019/core.py', 'df_helpers')
    prior = Path(cfg['prior_run'])
    control = Path(cfg['control_run'])
    start = cfg['original_start_epoch']
    nt, nv, nq = 0, 0, 0

    def status(phase, **kw):
        write(root / 'status.private.json', dict(state='RUNNING', phase=phase,
              updated=time.time(), started=start, new_training_images=nt,
              new_validation_images=nv, qualification_training_images=nq, **kw))
        assert time.time() < cfg['hard_deadline_epoch'] - 60
        assert nt <= cfg['train_count'] and nv <= cfg['val_count']
        assert nt + nq <= cfg['new_training_encode_cap']
        if torch.cuda.is_initialized():
            assert torch.cuda.max_memory_reserved() <= cfg['gpu_reserved_budget_bytes']

    status('data_and_algebra_qualification')
    self_check()
    train, val = data.read_rows(cfg, 'train'), data.read_rows(cfg, 'val')
    assert len(train) == 68058 and len(val) == 300
    assert not ({r['group'] for r in train} & {r['group'] for r in val})
    old_cfg = json.loads((prior / 'config.private.json').read_text())
    for k in ['seed', 'order', 'task_sizes', 'manifest', 'images', 'weight', 'author_labels']:
        assert old_cfg[k] == cfg[k]
    assert json.loads((control / 'status.private.json').read_text())['state'] == 'COMPLETE'
    from safetensors.torch import load_file
    import timm
    encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    encoder.load_state_dict({k: v for k, v in load_file(cfg['weight']).items()
                             if not k.startswith('head.')}, strict=True)
    encoder.eval().requires_grad_(False).cuda()

    def extract(rows):
        for images, idx in DataLoader(data.Images(rows, cfg), batch_size=64,
                                      num_workers=0, shuffle=False):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                z = encoder.forward_features(images.cuda())
            assert z.shape[1:] == (197, 768) and torch.isfinite(z).all()
            yield z[:, 0].double(), z[:, 1:].float(), idx

    # Training-only, disjoint from validation; extra exposure is explicitly costed.
    qrows = [r for r in train if r['label'] == cfg['order'][0]][:32]
    qstart = time.time()
    for _, tokens, idx in extract(qrows):
        nq += len(idx)
        status('pixel_and_p1_qualification')
        error = float((signed_pool(tokens, 1) - tokens.mean(1)).abs().max())
        assert error < 2e-6
        assert torch.isfinite(signed_pool(tokens)).all()
        direct = tokens.double().clamp_min(0).pow(3).mean(1).pow(1/3)
        direct -= (-tokens.double()).clamp_min(0).pow(3).mean(1).pow(1/3)
        p3_error = float((signed_pool(tokens).double()-direct).abs().max())
        assert torch.allclose(signed_pool(tokens).double(), direct, atol=2e-5, rtol=2e-6)
    torch.cuda.synchronize()
    qualification_seconds = time.time()-qstart
    # Conservative small-window feasibility screen, not a stable-throughput claim.
    estimate = qualification_seconds / nq * cfg['train_count'] * 1.5 + 900
    assert estimate < cfg['max_wall_seconds']-300, 'Qualification cannot support the frozen 8h budget'
    write(root / 'QUALIFICATION.json', dict(state='PASS', training_images=nq,
          validation_images=0, optimizer_updates=0, p1_max_error=error,
          p3_float64_reference_max_error=p3_error,
          wall_seconds=qualification_seconds, conditional_projected_seconds=estimate,
          forecast_is_not_upper_bound=True))
    grams, targets, cg, parity, concentrations = [], [], [], {}, {}
    for c in cfg['order']:
        rows = [r for r in train if r['label'] == c]
        old = torch.load(prior / f'class_{c:02d}.private.pt', weights_only=True)
        weights = common.weights([r['group'] for r in rows], 'cuda')
        assert len({r['group'] for r in rows}) == old['groups']
        g = torch.zeros(1537, 1537, dtype=torch.float64, device='cuda')
        diag = {k: dict(valid_channels=0, effective_sum=0.) for k in ['positive', 'negative']}
        for cls, tokens, idx in extract(rows):
            nt += len(idx)
            x = common.augmented(torch.cat([cls, signed_pool(tokens).double()], 1))
            mass = weights[idx.cuda()]
            g += x.T @ (x * mass[:, None])
            for k, v in concentration(tokens).items():
                for field in v:
                    diag[k][field] += v[field]
            status('training_moments', class_id=c, class_images=int(idx[-1])+1)
        g = g.cpu()
        ids = list(range(768)) + [1536]
        err = float((g[ids][:, ids] - old['ridge_xx']).abs().max())
        assert err < cfg['feature_atol'], 'CLS marginal differs from reference'
        assert abs(float(g[-1, -1])-1) < 1e-9
        parity[str(c)] = err
        concentrations[str(c)] = diag
        t = g[:, -1].clone()
        torch.save(dict(gram=g, target=t, images=len(rows), groups=old['groups']),
                   root / f'class_{c:02d}.private.pt')
        grams.append(g)
        targets.append(t)
        cg.append(old['ridge_xx'])
        print(json.dumps(dict(event='class_complete', class_id=c, training_images=nt,
                              CLS_gram_error=err)), flush=True)
    assert nt == cfg['train_count']
    ec = sum(float(g[:-1, :-1].trace()) for g in cg[:10]) / 10
    ep = sum(float(g[768:1536, 768:1536].trace()) for g in grams[:10]) / 10
    scale = (ec / ep)**.5
    assert ec > 0 and ep > 0
    audits = []
    for stage in range(6):
        n = sum(cfg['task_sizes'][:stage+1])
        g, t, factors = df.scaled(torch.stack(grams[:n]).sum(0), torch.stack(targets[:n], 1), scale)
        eigen = df.spectrum(g)
        target_df = df.degrees(df.spectrum(torch.stack(cg[:n]).sum(0)), 1.)
        alpha = df.match_alpha(eigen, target_df)
        head = torch.linalg.solve(g + alpha * torch.eye(1537, dtype=torch.float64), t)
        torch.save(dict(head=head, factors=factors, alpha=alpha), root / f'heads_{stage}.private.pt')
        audits.append(dict(stage=stage, classes=n, alpha=alpha, target_df=target_df,
                           actual_df=df.degrees(eigen, alpha)))
        status('fit_candidate', stage=stage)
    write(root / 'FIT_SEAL.json', dict(query_opened=False, frozen_at=time.time(),
          p=3, patch_scale=scale, base_CLS_energy=ec, base_patch_energy=ep,
          stages=audits, training_CLS_parity=parity))
    # The first validation encoding happens only after all candidate heads are sealed.
    query = {}
    for c in cfg['order']:
        rows = [r for r in val if r['label'] == c]
        assert len(rows) == 15
        cls_parts, patch_parts = [], []
        for cls, tokens, idx in extract(rows):
            nv += len(idx)
            cls_parts.append(cls.cpu())
            patch_parts.append(signed_pool(tokens).double().cpu())
        cls, patch = torch.cat(cls_parts), torch.cat(patch_parts)
        ref = torch.load(prior / f'val_{c:02d}.private.pt', weights_only=True)[2].double()
        assert float((cls-ref).abs().max()) <= cfg['feature_atol']
        query[c] = torch.cat([cls, patch, torch.ones(15, 1, dtype=torch.float64)], 1)
        status('validation_features', class_id=c)
    torch.save(query, root / 'query.private.pt')
    write(root / 'CONCENTRATION.json', concentrations)
    write(root / 'costs.json', dict(wall_seconds=time.time()-start,
          new_training_images=nt, qualification_training_images=nq,
          total_training_encodes=nt+nq+cfg.get('prior_qualification_images', 0), new_validation_images=nv,
          prior_engineering_failure=cfg.get('prior_engineering_failure'),
          optimizer_updates=0, GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
          GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(), attempts=1+cfg.get('prior_attempts', 0),
          reference_creation_costs_excluded_and_retained_in_historical_reports=True))
    write(root / 'status.private.json', dict(state='FIT_COMPLETE_PENDING_VERIFICATION',
          updated=time.time(), started=start, new_training_images=nt,
          qualification_training_images=nq, new_validation_images=nv))


if __name__ == '__main__':
    cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:
        main(cfg)
    except BaseException:
        root = Path(cfg['output'])
        root.mkdir(exist_ok=True)
        p = root / 'status.private.json'
        state = json.loads(p.read_text()) if p.exists() else {}
        state.update(state='FAILED', updated=time.time(), error=traceback.format_exc())
        write(p, state)
        raise
