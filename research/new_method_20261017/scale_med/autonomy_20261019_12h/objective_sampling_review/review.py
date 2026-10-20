"""Aggregate-only analytical review; no data/model/optimizer access or fitting."""
import json, math, time
from pathlib import Path

def calculate():
    counts = [307, 20, 170, 2058]
    n, batch = sum(counts), 8
    assert n == 2555
    full, tail = divmod(n, batch)
    # A uniformly selected size-b subset from this fixed population.
    cells = []
    for c, nc in enumerate(counts):
        p0 = math.prod((n - nc - j) / (n - j) for j in range(batch)) if n-nc >= batch else 0.
        cells.append(dict(class_index=c, rows=nc,
                          expected_rows_per_full_batch=batch*nc/n,
                          probability_no_class_in_full_batch=p0,
                          expected_full_batches_containing_class=full*(1-p0),
                          expected_rows_per_tail=tail*nc/n,
                          mean_image_objective_weight=1/(4*nc)))
    ess_upper = 16 / sum(1/nc for nc in counts)
    # Adam's bias-corrected first update, scalar gradients only. No training.
    eps = 1e-8
    grads = [9., -1.]
    expected_gradient = sum(grads)/2
    normalize = lambda g: g/(abs(g)+eps)
    average_normalized = sum(normalize(g) for g in grads)/2
    normalized_average = normalize(expected_gradient)
    # Uniform subset expectation for a toy fixed-parameter weighted objective.
    values, weights = [2., 7., 11., 5.], [.5, .5, .25, .75]
    import itertools
    direct = sum(w*v for w,v in zip(weights,values))/sum(weights)
    subsets = list(itertools.combinations(range(4),2))
    mean_estimator = sum((4/sum(weights))*sum(weights[i]*values[i] for i in ss)/2 for ss in subsets)/len(subsets)
    assert abs(direct-mean_estimator) < 1e-12
    assert abs(average_normalized-normalized_average) > .99
    return dict(scope='ANALYTICAL_REVIEW_NO_NEW_PERFORMANCE_EXPERIMENT',
                training_rows=n,batch_size=batch,full_batches_per_epoch=full,tail_rows=tail,
                optimizer_steps_per_epoch=full+1,optimizer_steps_ten_epochs=10*(full+1),
                class_cells=cells,weighted_image_ESS_upper_bound=ess_upper,
                ESS_is_not_patient_or_information_count=True,
                fixed_parameter_uniform_subset_identity_error=abs(direct-mean_estimator),
                adam_first_step_counterexample=dict(gradients=grads,expected_gradient=expected_gradient,
                    expected_normalized_gradient=average_normalized,
                    normalized_expected_gradient=normalized_average,epsilon=eps),
                no_claim_this_counterexample_caused_observed_risk=True,
                source_read=['initial_bias/train.py','initial_bias/PROTOCOL.md','information_attribution/core_attribution.py'],
                new_fit_units=0,new_image_forwards=0,new_test_scores=0,
                frozen_bias_stop_preserved=True,method_success=False)
if __name__ == '__main__':
    start=time.monotonic(); cpu=time.process_time(); result=calculate()
    result['review_cost']=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
                               scope='pure arithmetic only, excludes source reading/network/serialization')
    Path(__file__).with_name('aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
