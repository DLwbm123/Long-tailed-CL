"""Descriptive count decomposition; no model queries, refitting or gate changes."""
import csv
import json
from pathlib import Path


def run(root):
    root=Path(root)
    info=json.loads((root/'r114_tars_rl/TARS_WEIGHTS.json').read_text())
    counts=info['counts'];tiny=[c for c,n in enumerate(counts) if n<25]
    head=[c for c,n in enumerate(counts) if n>=250];non_tiny=[c for c in range(23) if c not in tiny]
    assert tiny==[5,6,17,19,21] and len(head)==12
    sources={}
    for folder in ['r114_tars_rl','r115_head_protected_tars','r116_reference_kl_tars',
                   'r117_tailness_query_allocation','r118_reward_support_bootstrap','r119_visual_evidence_tars']:
        assert json.loads((root/folder/'FINAL_VERIFICATION.json').read_text())['passed']
        data=json.loads((root/folder/'FINAL_RESULTS.json').read_text())
        sources[folder.split('_')[0]]={r['name']:r for r in data['results']}
    original=sources['r114']['SFT'];all_rows=[]
    def macro(row,ids):
        return sum(row['correct'][c]/row['denominator'][c] for c in ids)/len(ids)
    def comparison(name,a,b):
        assert a['denominator']==b['denominator'] and sum(a['denominator'])==1698
        for row in [a,b]:
            assert all(isinstance(x,int) and 0<=x<=n for x,n in zip(row['correct'],row['denominator']))
            assert abs(macro(row,list(range(23)))-row['macro'])<1e-12
        contributions=[]
        for c in range(23):
            delta=100*(a['correct'][c]-b['correct'][c])/a['denominator'][c]
            value={'comparison':name,'class':c,'name':info['class_names'][str(c)],'TRAIN_count':counts[c],
                   'before_correct':b['correct'][c],'after_correct':a['correct'][c],'VAL_denominator':a['denominator'][c],
                   'class_recall_delta_pp':delta,'all23_macro_contribution_pp':delta/23,
                   'head12_mean_contribution_pp':delta/12 if c in head else 0.,'tiny_from_TRAIN':c in tiny}
            all_rows.append(value);contributions.append(value)
        full=100*(a['macro']-b['macro'])
        assert abs(sum(x['all23_macro_contribution_pp'] for x in contributions)-full)<1e-10
        return {'macro_delta_pp':full,'head_delta_pp':100*(a['head']-b['head']),
                'tail_delta_pp':100*(a['tail']-b['tail']),
                'all_five_tiny_removed_18class_macro_delta_pp':100*(macro(a,non_tiny)-macro(b,non_tiny)),
                'tiny5_signed_contribution_to_original23_macro_pp':sum(x['all23_macro_contribution_pp'] for x in contributions if x['tiny_from_TRAIN']),
                'class_contributions':contributions}
    comparisons={}
    for exp,arms in sources.items():
        for name,row in arms.items():comparisons[exp+'/'+name+'_vs_originalSFT']=comparison(exp+'/'+name+'_vs_originalSFT',row,original)
    support=sources['r118']['SUPPORT_SFT'];main=sources['r118']['SUPPORT_TARS_FULL_IS'];ce=sources['r118']['SUPPORT_CE_FULL_IS']
    for name,row in sources['r118'].items():
        if name!='SUPPORT_SFT':comparisons['r118/'+name+'_vs_support']=comparison('r118/'+name+'_vs_support',row,support)
    local=comparison('r118/TARS_vs_CE',main,ce);comparisons['r118/TARS_vs_CE']=local
    effect=comparisons['r118/SUPPORT_TARS_FULL_IS_vs_support'];one=effect['class_contributions'][17]['all23_macro_contribution_pp']
    assert main['correct'][17]-support['correct'][17]==1 and main['denominator'][17]==5
    m117=sources['r117']['ALLOC_TARS_FULL_IS'];effect117=comparisons['r117/ALLOC_TARS_FULL_IS_vs_originalSFT']
    selected=sum(x['head12_mean_contribution_pp'] for x in effect117['class_contributions'] if x['class'] in [10,20])
    m119=sources['r119']['VISUAL_TARS_FULL_IS'];v119=sources['r119']['VISUAL_SFT'];c119=sources['r119']['VISUAL_CE_FULL_IS']
    local119=comparison('r119/TARS_vs_visualSFT',m119,v119);comparisons['r119/TARS_vs_visualSFT']=local119
    ce119=comparison('r119/CE_vs_visualSFT',c119,v119);comparisons['r119/CE_vs_visualSFT']=ce119
    metrics=['macro','head','tail']
    interaction={k:100*((m119[k]-v119[k])-(m117[k]-original[k])) for k in metrics}
    ce_interaction={k:100*((c119[k]-v119[k])-(sources['r117']['ALLOC_CE_FULL_IS'][k]-original[k])) for k in metrics}
    result={'scope':'saved_only_descriptive','new_model_queries_updates':0,'tiny_set_fixed_by_TRAIN_counts':tiny,
            'all_completed_endpoints_included':sum(len(a) for a in sources.values()),
            'R118':{'full_macro_increment_pp':effect['macro_delta_pp'],'UC01_one_row_contribution_pp':one,
                    'UC01_share_of_net_increment':one/effect['macro_delta_pp'],
                    'other22_signed_contribution_to_original23_macro_pp':effect['macro_delta_pp']-one,
                    'non_tiny18_macro_over_support_pp':effect['all_five_tiny_removed_18class_macro_delta_pp'],
                    'non_tiny18_macro_over_CE_pp':local['all_five_tiny_removed_18class_macro_delta_pp'],
                    'support_share_of_net_head_loss':(support['head']-original['head'])/(main['head']-original['head'])},
            'R117':{'UC2_and_normalZ_signed_head_contribution_pp':selected,
                    'UC2_and_normalZ_share_of_net_head_loss':selected/effect117['head_delta_pp'],
                    'oesophagitis_pair_net_correct_gain':sum(m117['correct'][c]-original['correct'][c] for c in [11,12]),
                    'UC2_UC3_pair_net_correct_gain':sum(m117['correct'][c]-original['correct'][c] for c in [20,22])},
            'resolution_interaction_R119_vs_R117':{'RL_increment_difference_pp':interaction,
                'CE_increment_difference_pp':ce_interaction,'RL_minus_CE_advantage_change_pp':{k:interaction[k]-ce_interaction[k] for k in metrics},
                'interpretation':'Descriptive fixed-endpoint differences; recovery from a weaker448start is not proof that448preferentiallyhelpsRL.'},
            'comparisons':comparisons,
            'limitations':['Original frozen metrics and gates remain unchanged. Sensitivity is not a replacement primary metric.',
                           'Posthoc head localization is not an authorization for targeted class-specific protection tuning.',
                           'Signed shares divide by net changes and can reflect offsetting effects; they are not causal fractions.',
                           'Same repeatedly used development VAL, one training seed; not independent patient replication.']}
    out=root/'tars_rl_saved_analysis';out.mkdir(exist_ok=True)
    (out/'CONTRIBUTION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'CLASS_CONTRIBUTIONS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    return result


if __name__=='__main__':
    result=run(Path(__file__).resolve().parent.parent)
    print(json.dumps({k:v for k,v in result.items() if k in ['scope','new_model_queries_updates','all_completed_endpoints_included','R118','R117','resolution_interaction_R119_vs_R117']},indent=2))
