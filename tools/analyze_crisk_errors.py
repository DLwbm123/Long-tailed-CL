"""Decompose sealed CRISK aggregates only; no medical/model IO or selection changes."""
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path


ROUND = Path('docs/crisk1_20261014')
CANDIDATES = ('C0125', 'C025', 'C05', 'C1')
STEPS = (.125, .25, .5, 1.)


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run():
    begun = time.monotonic()
    source = json.loads((ROUND / 'results/RESULTS.json').read_text())
    assert source['program']['status'] == 'COMPLETE' and not source['program']['active']
    assert source['source_commit'] == '295a059a3fca03ad7006c6125e4bb64264da18b5'
    cases, classes, guards, edges, summaries = [], [], [], [], []
    for job, result in source['runs'].items():
        data = source['lock']['data'][job.split('_')[0]]
        counts, reasons = Counter(), Counter()
        for stage, risk in zip(result['stages'], source['risk_prepared'][job]['stages']):
            task = stage['task']
            assert stage['selections'] == risk['selections']
            assert risk['selections']['CRISK']['chosen_scale'] == 0.
            if task == 1:
                continue
            audit = {(a['candidate'], a['label']): a for a in risk['class_audit'] if a['mode'] == 'CRISK'}
            canonical = stage['heads'][CANDIDATES[0]]
            seen = canonical['metrics']['OFF']['seen']
            old = canonical['old_count']
            for edge in canonical['class_confusion']:
                if edge['label'] != edge['competitor'] and edge['off']:
                    edges.append(dict(job=job, task=task, label=edge['label'], competitor=edge['competitor'],
                                      true_is_old=edge['label'] in seen[:old], predicted_is_old=edge['competitor'] in seen[:old],
                                      errors=edge['off'], val_n=data['val_counts'][str(edge['label'])],
                                      train_n=data['train_counts'][str(edge['label'])],
                                      conditional_error_rate=edge['off']/data['val_counts'][str(edge['label'])]))
            for name, scale, choice in zip(CANDIDATES, STEPS, risk['selections']['CRISK']['candidates']):
                assert choice['scale'] == scale and choice['ready'] and not choice['admitted']
                on = stage['heads'][name]
                raw = stage['raw_heads'][name]
                offm, onm = on['metrics']['OFF'], on['metrics']['ON']
                utility_failed = choice['conditional_utility'] <= 1e-12
                failed_labels = [seen[i] for i in choice['edge_failed_classes']]
                assert all(a in seen[:old] for a in failed_labels)
                assert set(failed_labels) == {c for c in seen[:old] if audit[name, c]['edge_failed']}
                delta = 100*(onm['balanced_accuracy']-offm['balanced_accuracy'])
                sign = 'positive_BA' if delta > 1e-10 else 'negative_BA' if delta < -1e-10 else 'zero_BA'
                reason = 'utility_and_edge' if utility_failed and failed_labels else 'utility_only' if utility_failed else 'edge_only'
                counts[sign] += 1
                reasons[reason] += 1
                raw_predictive = sum(audit[name, c]['rescue_mean']-audit[name, c]['harm_mean'] for c in seen)/len(seen)
                train_macro = sum((audit[name, c]['training_counts']['rescued']-audit[name, c]['training_counts']['harmed'])/audit[name, c]['target_actual_n'] for c in seen)/len(seen)
                class_contributions = []
                for c in seen:
                    paired = on['paired'][str(c)]
                    ra = audit[name, c]
                    tr = ra['training_counts']
                    assert paired['n'] == data['val_counts'][str(c)] and tr['n'] == data['train_counts'][str(c)]
                    assert paired['net_correct'] == paired['rescued']-paired['harmed']
                    contribution = 100*paired['net_correct']/paired['n']/len(seen)
                    class_contributions.append(contribution)
                    classes.append(dict(job=job, task=task, candidate=name, label=c, is_old=c in seen[:old],
                                        train_n=tr['n'], train_correct_off=tr['correct_off'], train_rescued=tr['rescued'], train_harmed=tr['harmed'],
                                        val_n=paired['n'], correct_off=paired['correct_off'], correct_on=paired['correct_on'],
                                        rescued=paired['rescued'], harmed=paired['harmed'], net_correct=paired['net_correct'],
                                        wrong_to_different_wrong=paired['wrong_to_different_wrong'], macro_contribution_pp=contribution,
                                        raw_rescued=raw['paired'][str(c)]['rescued'], raw_harmed=raw['paired'][str(c)]['harmed'],
                                        edge_score=ra['relative_edge_score'], edge_budget=ra['edge_budget'], edge_failed=ra['edge_failed'],
                                        posterior_net_before_drift=ra['rescue_mean']-ra['harm_mean'], posterior_net_after_drift=ra['conditional_net_mean'],
                                        posterior_off_error=ra['off_error_mean'], empirical_off_error=ra['empirical_off_error'], model_off_error=ra['model_q'],
                                        rescue_conditional_support=ra['rescue_conditional_n'], harm_conditional_support=ra['harm_conditional_n'],
                                        model_prior_pseudocount=ra['model_prior_pseudocount'],
                                        donor_total_mixing_weight=sum(z['weight'] for z in ra['donor_audit']),
                                        drift_q=ra['drift_off_error_sensitivity'], drift_r=ra['drift_rescue_sensitivity'], drift_h=ra['drift_harm_sensitivity']))
                assert math.isclose(sum(class_contributions), delta, abs_tol=1e-9)
                assert on['paired']['new']['harmed'] == 0
                paired = on['paired']['all']
                cases.append(dict(job=job, task=task, candidate=name, scale=scale, final_stage=task == len(result['stages']),
                                  actual_sign=sign, rejection=reason, edge_failed_original_labels=json.dumps(failed_labels),
                                  delta_BA_pp=delta, delta_tail_pp=100*(onm['tail_recall']-offm['tail_recall']),
                                  delta_Acc_pp=100*(onm['accuracy']-offm['accuracy']), delta_F1_pp=100*(onm['macro_F1']-offm['macro_F1']),
                                  rescued=paired['rescued'], harmed=paired['harmed'], net_correct=paired['net_correct'],
                                  old_rescued=on['paired']['old']['rescued'], old_harmed=on['paired']['old']['harmed'],
                                  new_rescued=on['paired']['new']['rescued'], new_harmed=on['paired']['new']['harmed'],
                                  train_macro_net_pp=100*train_macro, posterior_net_before_drift_pp=100*raw_predictive,
                                  posterior_mean_after_drift_pp=100*choice['predictive_net_mean'], utility_q10_pp=100*choice['conditional_utility'],
                                  no_drift_q10_pp=100*next(c['conditional_utility'] for c in risk['selections']['NO_DRIFT']['candidates'] if c['scale'] == scale),
                                  raw_delta_BA_pp=100*(raw['metrics']['ON']['balanced_accuracy']-offm['balanced_accuracy'])))
                g = on['guard_diagnostic']
                assert g['raw_new_loss_removed'] == raw['paired']['new']['harmed']
                guards.append(dict(job=job, task=task, candidate=name, final_stage=task == len(result['stages']), **g,
                                   old_raw_rescued=raw['paired']['old']['rescued'], old_raw_harmed=raw['paired']['old']['harmed'],
                                   old_guarded_rescued=on['paired']['old']['rescued'], old_guarded_harmed=on['paired']['old']['harmed'],
                                   raw_new_margin_negative=on['margins']['new']['raw_delta']['negative'],
                                   guarded_new_margin_negative=on['margins']['new']['applied_delta']['negative']))
                assert guards[-1]['guarded_new_margin_negative'] == 0
        summaries.append(dict(job=job, **counts, **reasons))
    assert len(cases) == 56 and len(classes) == 872 and len(guards) == 56
    assert sum(c['actual_sign'] == 'positive_BA' for c in cases) == 15
    assert sum(c['actual_sign'] == 'negative_BA' for c in cases) == 41
    out = ROUND / 'error_decomposition'
    out.mkdir(exist_ok=True)
    for filename, rows in [('CANDIDATE_REJECTIONS', cases), ('CLASS_CONTRIBUTIONS', classes), ('GUARD_TRADEOFFS', guards), ('OFF_CONFUSION_EDGES', edges)]:
        write_csv(out / (filename+'.csv'), rows)
    summary = dict(status='AGGREGATE_ANALYSIS_COMPLETE', scientific_source_commit=source['source_commit'],
                   source_report_commit='e52c57a424774d034ccf11a4f2c12f30fd26bef6',
                   GPU_processes=0, new_medical_image_reads=0, new_checkpoint_reads=0, optimizer_updates=0,
                   parameters_or_selections_changed=False, test_accessed=False, validation_retuning=False,
                   retrospective_val_description_not_selection=True, nonfirst_candidate_states=56,
                   first_stage_identity_states_excluded=16, missed_positive_BA_states=15, avoided_negative_BA_states=41,
                   positive_BA_with_negative_tail=sum(c['actual_sign']=='positive_BA' and c['delta_tail_pp'] < -1e-10 for c in cases),
                   utility_and_edge=sum(c['rejection']=='utility_and_edge' for c in cases),
                   utility_only=sum(c['rejection']=='utility_only' for c in cases), jobs=summaries,
                   class_rows=len(classes), guard_rows=len(guards),
                   rare_HK_labels_descriptive_only=[5,6,19,21],
                   repeated_stage_candidate_events_not_unique_images=True,
                   pipeline_seconds=time.monotonic()-begun, process_residence_seconds=None,
                   analysis_CPU_not_added_to_prior_experiment_ledger=True)
    (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
