import json
from pathlib import Path

METHODS = ['NATIVE_ROOT', 'DUPLICATE', 'FIXED_CENTER', 'GLOBAL_BOX', 'DETAIL_BOX']


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def score(rows, counts):
    den = [sum(r['class'] == c for r in rows) for c in range(23)]
    assert all(den)
    groups = {'macro': list(range(23)), 'head': [c for c,n in enumerate(counts) if n>=250],
              'tail': [c for c,n in enumerate(counts) if n<250], 'tiny': [c for c,n in enumerate(counts) if n<25],
              'non_tiny': [c for c,n in enumerate(counts) if n>=25]}
    results = []
    for method in METHODS:
        good = [sum(r['class']==c and r['results'][method]['prediction']==c for r in rows) for c in range(23)]
        result = {'name': method, 'correct': good, 'denominator': den, 'micro':sum(good)/sum(den)}
        result.update({g:sum(good[c]/den[c] for c in ids)/len(ids) for g,ids in groups.items()})
        result.update({k:sum(r['results'][method][k] for r in rows)
                       for k in ['input_tokens','generated_tokens','vision_encoder_rows','seconds']})
        results.append(result)
    return results


def summarize(rows, counts):
    results = score(rows, counts); parts = {s:score([r for r in rows if r['split']==s], counts) for s in ['A','B']}
    main=results[4]; gates={}
    for index,name in [(1,'DUPLICATE'),(0,'NATIVE_ROOT')]:
        ref=results[index]
        for key,threshold in [('macro',.01),('tail',.02),('head',-.01)]:
            gates[f'DETAIL_over_{name}_{key}_threshold'] = main[key]-ref[key]>=threshold
    for index,name in [(2,'FIXED_CENTER'),(3,'GLOBAL_BOX')]:
        for key in ['macro','tail']:gates[f'DETAIL_over_{name}_{key}_positive']=main[key]>results[index][key]
    for s in ['A','B']:gates[f'{s}_non_tiny_over_DUPLICATE_positive']=parts[s][4]['non_tiny']>parts[s][1]['non_tiny']
    gates['four_joint_visual_and_input_costs_equal']=len({(r['vision_encoder_rows'],r['input_tokens']) for r in results[1:]})==1
    assert len(gates)==13
    return {'scope':'TRAIN_only_joint_observation_diagnostic_not_RL_training','results':results,'partitions':parts,
            'gates':gates,'all_gates_pass':all(gates.values()),'components':175,'VAL_TEST_optimizer_updates':0,
            'classifier_TRAIN_in_sample':True,'R122_failed_gates_preserved':True}
