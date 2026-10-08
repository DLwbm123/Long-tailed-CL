"""Frozen training-only admission gate; no fitting or validation-set selection."""
import itertools
import json
from pathlib import Path


def summarize(records, expected=20):
    if len(records) != expected or any(r.get('status') != 'COMPLETE' for r in records):
        return dict(status='INCOMPLETE', passed=False, completed=len(records), expected=expected)
    pairs = []; false_safe = 0; recall_regrets = []; information = 0
    for record in records:
        arms = record['arms']
        if len(arms) != 3 or record['actual_updates'] != record['expected_updates']:
            raise ValueError('Invalid calibration accounting')
        if any(record[k] for k in ('real_old_images_accessed','future_images_accessed','validation_images_accessed','test_accessed')):
            raise ValueError('Calibration data boundary violation')
        for a, b in itertools.combinations(arms, 2):
            predicted = a['proxy_reward']-b['proxy_reward']
            actual = a['oracle_reward']-b['oracle_reward']
            if abs(predicted) >= 1e-4 and abs(actual) >= 1e-4:
                pairs.append(predicted*actual > 0)
        for a in arms:
            predicted_gain = a['proxy_before'][0]-a['proxy_after'][0]
            recall_loss = a['oracle_before']['recall'][0]-a['oracle_after']['recall'][0]
            false_safe += predicted_gain >= 1e-4 and recall_loss >= .01
        recalls = [sum(a['oracle_after']['recall'])/2 for a in arms]
        selected = max(range(3), key=lambda i:(arms[i]['proxy_reward'], i == 1, -i))
        recall_regrets.append(max(recalls)-recalls[selected])
        information += max(recalls)-min(recalls) >= .01
    agreement = sum(pairs)/len(pairs) if pairs else None
    gates = dict(enough_informative_pairs=len(pairs) >= 20,
        pairwise_reward_order=agreement is not None and agreement >= 2/3,
        missed_old_recall_harm=false_safe/(3*expected) <= .1,
        mean_class_balanced_recall_regret=sum(recall_regrets)/expected <= .01,
        enough_action_effects=information >= 5)
    return dict(status='PASS' if all(gates.values()) else 'STOPPED_REWARD_GATE', passed=all(gates.values()),
        gates=gates, completed=len(records), informative_pairs=len(pairs), pairwise_agreement=agreement,
        false_safe_branches=false_safe, branch_count=3*expected, informative_episodes=information,
        mean_recall_regret=sum(recall_regrets)/expected, actual_updates=sum(r['actual_updates'] for r in records),
        independent_confirmation=False, scope='arrived T2 pseudo-incremental training diagnostic', test_accessed=False)


if __name__ == '__main__':
    import sys
    c=json.load(sys.stdin)
    result=summarize([json.loads(Path(p).read_text()) for p in c['records']],c.get('expected',20))
    from run_prototype_single import save
    save(Path(c['output']),result)
    print(json.dumps(result,indent=2))
