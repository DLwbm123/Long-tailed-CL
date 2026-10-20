"""Independent saved-statistic solve and metric verification; no image encoding."""
import json
import os
import time
from pathlib import Path
import torch
from run import module, write


def main(cfg):
    torch.set_num_threads(4)
    root, base = Path(cfg['output']), Path(cfg['remote_base'])
    start = time.time()
    data = module(base / 'case_hierarchical1_20261018/data.py', 'verification_data')
    df = module(base / 'joint_controls1_20261019/core.py', 'verification_df')
    diag = module(base / 'joint_origin1_20261019/diagnose.py', 'verification_diagnostics')
    train, val = data.read_rows(cfg, 'train'), data.read_rows(cfg, 'val')
    assert len(train) == 68058 and len(val) == 300
    assert not ({r['group'] for r in train} & {r['group'] for r in val})
    seal = json.loads((root / 'FIT_SEAL.json').read_text())
    assert seal['query_opened'] is False and seal['p'] == 3
    assert json.loads((root / 'QUALIFICATION.json').read_text())['state'] == 'PASS'
    costs = json.loads((root / 'costs.json').read_text())
    assert costs['total_training_encodes'] == 68090+cfg.get('prior_qualification_images', 0)
    assert costs['new_validation_images'] == 300
    assert costs['optimizer_updates'] == 0
    query = torch.load(root / 'query.private.pt', weights_only=True)
    stats = [torch.load(root / f'class_{c:02d}.private.pt', weights_only=True) for c in cfg['order']]
    for c, s in zip(cfg['order'], stats):
        rows = [r for r in train if r['label'] == c]
        assert s['images'] == len(rows) and s['groups'] == len({r['group'] for r in rows})
        assert torch.equal(s['target'], s['gram'][:, -1])
    ec = sum(float(s['gram'][:768, :768].trace()) for s in stats[:10]) / 10
    ep = sum(float(s['gram'][768:1536, 768:1536].trace()) for s in stats[:10]) / 10
    assert abs((ec/ep)**.5 - seal['patch_scale']) < 1e-6
    # Compare separate Cholesky reconstruction to the worker's general linear solve.
    records, diagnostics, previous = [], [], None
    for stage in range(6):
        assert time.time() < cfg['hard_deadline_epoch'] - 60
        old = sum(cfg['task_sizes'][:stage])
        n = old + cfg['task_sizes'][stage]
        seen = cfg['order'][:n]
        g, t, factors = df.scaled(torch.stack([s['gram'] for s in stats[:n]]).sum(0),
                                 torch.stack([s['target'] for s in stats[:n]], 1), seal['patch_scale'])
        head = torch.load(root / f'heads_{stage}.private.pt', weights_only=True)
        ids = list(range(768)) + [1536]
        cls_g = torch.stack([s['gram'][ids][:, ids] for s in stats[:n]]).sum(0)
        target_df = df.degrees(df.spectrum(cls_g), 1.)
        assert abs(df.degrees(df.spectrum(g), head['alpha']) - target_df) < 1e-6
        reconstructed = torch.cholesky_solve(t, torch.linalg.cholesky(g + head['alpha'] * torch.eye(1537, dtype=torch.float64)))
        assert torch.allclose(reconstructed, head['head'], atol=1e-8, rtol=1e-7)
        baseline = torch.load(Path(cfg['control_run']) / f'stage_{stage}.private.pt', weights_only=True)
        labels = baseline['labels']
        assert baseline['seen'] == seen
        expected = torch.tensor([r['label'] for r in val if r['label'] in seen])
        assert torch.equal(labels, expected)
        pos = {c: 0 for c in seen}
        xs = []
        for c in labels.tolist():
            xs.append(query[c][pos[c]])
            pos[c] += 1
        x = torch.stack(xs) * factors
        scores = x @ head['head']
        assert float((scores-x@reconstructed).abs().max()) < 1e-8
        all_scores = dict(baseline['scores'], signed_p3_df_joint=scores)
        predictions = {a: torch.tensor(seen)[s.argmax(1)] for a, s in all_scores.items()}
        metrics = {}
        for arm, pred in predictions.items():
            counts = [int((labels == c).sum()) for c in seen]
            correct = [int(((labels == c) & (pred == c)).sum()) for c in seen]
            assert counts == [15] * n
            recalls = [a/b for a, b in zip(correct, counts)]
            metrics[arm] = dict(macro=sum(recalls)/n, counts=counts, correct=correct, recall=recalls)
        contrast = {}
        for arm in ['df_matched_joint', 'scale_joint']:
            p, q = predictions['signed_p3_df_joint'], predictions[arm]
            contrast[arm] = dict(rescue=int(((p == labels) & (q != labels)).sum()),
                                  harm=int(((p != labels) & (q == labels)).sum()),
                                  changed=int((p != q).sum()),
                                  delta_BA=metrics['signed_p3_df_joint']['macro']-metrics[arm]['macro'])
        saved = dict(scores=all_scores, labels=labels, seen=seen)
        diagnostics.append(diag.stage_diagnostics(saved, previous))
        previous = saved
        records.append(dict(stage=stage, seen=seen, metrics=metrics, contrasts=contrast))
        torch.save(saved, root / f'stage_{stage}.private.pt')
    support = json.loads((Path(cfg['prior_run']) / 'support.json').read_text())
    groups = {}
    for criterion in ['images', 'patients']:
        ordered = sorted(cfg['order'], key=lambda c: (support[str(c)][criterion], c))
        bins = dict(tail7=ordered[:7], middle6=ordered[7:13], head7=ordered[13:],
                    initial10=cfg['order'][:10], later10=cfg['order'][10:])
        groups[criterion] = dict(bins=bins, correct={
            a: {k: sum(m['correct'][cfg['order'].index(c)] for c in members)
                for k, members in bins.items()} for a, m in records[-1]['metrics'].items()})
    averages = {a: sum(r['metrics'][a]['macro'] for r in records)/6 for a in records[-1]['metrics']}
    counts = groups['images']['correct']
    gates = dict(final_correct=sum(records[-1]['metrics']['signed_p3_df_joint']['correct']) >= 86,
                 mean_stage_gain=averages['signed_p3_df_joint']-averages['df_matched_joint'] >= .01,
                 tail7=counts['signed_p3_df_joint']['tail7'] >= 22,
                 later10=counts['signed_p3_df_joint']['later10'] >= counts['df_matched_joint']['later10'])
    write(root / 'results.json', records)
    write(root / 'DIAGNOSTICS.json', diagnostics)
    write(root / 'FREQUENCY.json', groups)
    write(root / 'DECISION.json', dict(gates=gates, proceed_to_B=all(gates.values()),
          mean_stage_BA=averages, thresholds_are_practical_not_significance=True))
    resources = json.loads((root.parent / 'guardian_resources.private.json').read_text())
    assert resources['new_output_bytes'] < cfg['output_budget_bytes']
    assert costs['GPU_peak_reserved_bytes'] <= cfg['gpu_reserved_budget_bytes']
    write(root / 'VERIFICATION.json', dict(state='PASS', stages=6, validation_images=300,
          new_image_encodes=0, wall_seconds=time.time()-start, metrics_independently_recomputed=True,
          heads_independently_solved=True, first_run_clock=cfg['original_start_epoch']))
    write(root / 'status.private.json', dict(state='COMPLETE', updated=time.time(),
          delivery_pending=True, decision=all(gates.values())))


if __name__ == '__main__':
    main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
