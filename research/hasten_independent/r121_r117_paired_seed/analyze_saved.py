"""Paired seed comparison and uniform class decomposition from saved counts only."""
import csv
import json
from pathlib import Path


def run(root):
    root=Path(root)
    load=lambda p:json.loads(p.read_text())
    previous=load(root.parent/'r117_tailness_query_allocation/FINAL_RESULTS.json')
    current=load(root/'FINAL_RESULTS.json')
    verification=load(root/'FINAL_VERIFICATION.json')
    assert verification['passed'] and not current['all_six_positive_directions_repeat']
    original={r['name']:r for r in previous['results']}
    new={r['name']:r for r in current['results']}
    paired=[(74005,previous['historical_SFT'],original['ALLOC_TARS_FULL_IS'],original['ALLOC_GRPO_FULL_IS'],original['ALLOC_CE_FULL_IS']),
            (74006,new['SEED_SFT'],new['SEED_TARS_FULL_IS'],new['SEED_GRPO_FULL_IS'],new['SEED_CE_FULL_IS'])]
    info=load(root/'TARS_WEIGHTS.json');metrics=['macro','head','tail','micro','tiny'];pairs={};classes=[]
    for seed,sft,main,grpo,ce in paired:
        assert all(x['denominator']==sft['denominator'] for x in [main,grpo,ce])
        pairs[str(seed)]={}
        for name,before in [('SFT',sft),('GRPO',grpo),('CE',ce)]:
            delta={k:100*(main[k]-before[k]) for k in metrics}
            pairs[str(seed)][name]=delta
            for c,(a,b,n) in enumerate(zip(main['correct'],before['correct'],main['denominator'])):
                classes.append({'seed':seed,'comparison':'main_vs_'+name,'class':c,'name':info['class_names'][str(c)],
                                'before_correct':b,'after_correct':a,'denominator':n,'recall_delta_pp':100*(a-b)/n,
                                'macro23_contribution_pp':100*(a-b)/n/23})
            assert abs(sum(r['macro23_contribution_pp'] for r in classes if r['seed']==seed and r['comparison']=='main_vs_'+name)-delta['macro'])<1e-10
    exposure={}
    for arm in ['SEED_TARS_FULL_IS','SEED_GRPO_FULL_IS']:
        rows=verification['learning'][arm]['classes'];counts=info['counts'];groups={}
        for name,ids in [('head',[c for c,n in enumerate(counts) if n>=250]),('tail',[c for c,n in enumerate(counts) if n<250]),('tiny',[c for c,n in enumerate(counts) if n<25])]:
            group={key:sum(rows[c][key] for c in ids) for key in ['queries','successful_responses','allwrong_groups','allright_groups','mixed_groups']}
            assert group['queries']==group['allwrong_groups']+group['allright_groups']+group['mixed_groups']
            group['mixed_fraction']=group['mixed_groups']/group['queries'];groups[name]=group
        exposure[arm]=groups
    result={'scope':'saved_only_two_training_seeds_same_development_VAL','new_model_queries_updates':0,
            'paired_deltas_pp':pairs,'newseed_seven_gates_passed':sum(current['gates'].values()),'newseed_seven_gates_total':7,
            'six_directional_components_repeated':sum(current['directional_positive_signal'].values()),
            'all_six_repeat':current['all_six_positive_directions_repeat'],
            'both_seeds_main_over_GRPO_macro_and_tail':all(pairs[str(seed)]['GRPO'][m]>0 for seed in [74005,74006] for m in ['macro','tail']),
            'both_seeds_main_over_SFT_and_CE_macro_and_tail':all(pairs[str(seed)][b][m]>0 for seed in [74005,74006] for b in ['SFT','CE'] for m in ['macro','tail']),
            'TRAIN_evolving_policy_group_composition':exposure,
            'limitations':['Two training seeds on reused developmentVAL; no formal significance, new patients or seed selection.',
                           'OldR117 per-class reward-group composition was not persisted; no invented seed-to-seed group comparison.',
                           'CE has no matched KL and fewer forwards; recipe comparison does not isolate RL-objective necessity.',
                           'Training response/group counts are repeated evolving-policy draws, not independent components or endpoint accuracy.']}
    (root/'PAIRED_SEED_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'PAIRED_CLASS_CONTRIBUTIONS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)
    return result


if __name__=='__main__':
    result=run(Path(__file__).resolve().parent)
    print(json.dumps(result,indent=2))
