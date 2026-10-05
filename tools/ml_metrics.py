"""Masked multi-label metrics, with explicit denominators and undefined cases."""
import numpy as np


def ranking_metrics(true, score):
    """Threshold-grouped AP and Mann-Whitney AUROC, including tied scores."""
    order = np.argsort(score, kind='stable')
    true, score = true[order], score[order]
    if not len(true):
        return None, None
    starts = np.r_[0, np.flatnonzero(np.diff(score)) + 1]
    count = np.diff(np.r_[starts, len(true)])
    positive = np.add.reduceat(true, starts)
    negative = count - positive
    positives, negatives = positive.sum(), negative.sum()
    tp = np.cumsum(positive[::-1])
    selected = np.cumsum(count[::-1])
    ap = float(np.sum(positive[::-1] * tp / selected) / positives) if positives else None
    negatives_below = np.cumsum(negative) - negative
    auc = float(np.sum(positive * (negatives_below + .5 * negative)) / (positives * negatives)) if positives and negatives else None
    return ap, auc


def evaluate(logits, targets, observed, labels, threshold=.5):
    z, y, mask = np.asarray(logits), np.asarray(targets), np.asarray(observed)
    if z.ndim != 2 or z.shape != y.shape or y.shape != mask.shape or z.shape[1] != len(labels):
        raise ValueError('Expected aligned B x C logits/targets/mask')
    if mask.dtype != np.bool_ or not np.isfinite(z).all() or not np.isin(y[mask], [0, 1]).all():
        raise ValueError('Invalid scores, targets, or mask')
    thresholds = np.broadcast_to(np.asarray(threshold, dtype=float), (len(labels),))
    if not ((thresholds > 0) & (thresholds < 1)).all():
        raise ValueError('Thresholds must be fixed before evaluation and lie in (0,1)')
    prediction = z >= np.log(thresholds / (1 - thresholds))
    rows = []
    for c, name in enumerate(labels):
        true, pred, score = y[mask[:, c], c], prediction[mask[:, c], c], z[mask[:, c], c]
        positive, negative = int((true == 1).sum()), int((true == 0).sum())
        ap, auc = ranking_metrics(true, score)
        tp, fp, fn = int(((true == 1) & pred).sum()), int(((true == 0) & pred).sum()), int(((true == 1) & ~pred).sum())
        rows.append(dict(label=name, observed=len(true), positive=positive, negative=negative,
                         AP=ap, AUROC=auc,
                         F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                         precision=tp / (tp + fp) if tp + fp else None,
                         recall=tp / positive if positive else None,
                         zero_recall=bool(positive and tp == 0), threshold=float(thresholds[c]),
                         tp=tp, fp=fp, fn=fn))
    aggregate = {}
    for key, output in [('AP', 'mAP'), ('AUROC', 'macro_AUROC'), ('F1', 'macro_F1')]:
        values = [r[key] for r in rows if r[key] is not None]
        aggregate[output] = float(np.mean(values)) if values else None
        aggregate[output + '_valid_labels'] = len(values)
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
    aggregate['micro_F1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    aggregate['observed_pairs'] = int(mask.sum())
    return dict(aggregate=aggregate, per_label=rows)


def groups(report, label_groups):
    """Groups (old/current/head/tail) must be protocol inputs, not test-derived."""
    by_name = {r['label']: r for r in report['per_label']}
    result = {}
    for name, labels in label_groups.items():
        if len(labels) != len(set(labels)) or not set(labels) <= set(by_name):
            raise ValueError('Invalid metric group')
        result[name] = {}
        for metric in ('AP', 'AUROC', 'F1', 'recall'):
            values = [by_name[c][metric] for c in labels if by_name[c][metric] is not None]
            result[name][metric] = float(np.mean(values)) if values else None
            result[name][metric + '_valid_labels'] = len(values)
    return result
