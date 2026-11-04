"""TRAIN-only stable-ID binding; never read images or VAL/TEST."""
import csv,json,io,random,importlib.util
from pathlib import Path
from common import write

def run(root,legacy_path,binding_path):
    root=Path(root);legacy=json.loads(Path(legacy_path).read_text());source=json.loads(Path(binding_path).read_text())
    rows=list(csv.DictReader((Path(legacy['manifest'])/'train.csv').open()))
    previous=list(csv.DictReader(io.StringIO(source['train_manifest'])))
    assert len(rows)==6821 and all(v['split']=='train' for v in rows)
    stable={v['sample_id']:i for i,v in enumerate(rows)};assert len(stable)==6821
    assert {v['sample_id']:(v['original_label'],v['identity_component']) for v in rows}=={v['sample_id']:(v['original_label'],v['identity_component']) for v in previous}
    spec=importlib.util.spec_from_file_location('taskbuilder',root/'build_task_settings.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    counts=[sum(int(v['original_label'])==label for v in rows) for label in legacy['order']];assert counts==list(module.COUNTS)
    roles={k:[stable[previous[i]['sample_id']] for i in source['roles'][old]] for k,old in [('S','S'),('B','W'),('F','F')]}
    ss=[set(v) for v in roles.values()];assert len(set.union(*ss))==6821 and all(not a&b for i,a in enumerate(ss) for b in ss[i+1:])
    comps=[{rows[i]['identity_component'] for i in v} for v in roles.values()];assert all(not a&b for i,a in enumerate(comps) for b in comps[i+1:])
    write(root/'SOURCE_BINDING.private.json',{'train_manifest':(Path(legacy['manifest'])/'train.csv').read_text(),'roles':roles,'stable_sample_ids':{k:[rows[i]['sample_id'] for i in v] for k,v in roles.items()},'no_legacy_teacher_or_feature_import':True})
    streams=[module.make_spec('HK_Ordered_NoLargeBase_5Tasks',module.ordered_tasks(),None,legacy['order'])]
    streams += [module.make_spec(f'HK_StratifiedShuffled_NoLargeBase_order{s}',module.stratified_tasks(s),s,legacy['order']) for s in [0,1,2]]
    mappings=[];tasks=[]
    for stream in streams:
        flat=stream['legacy_order_new_arrival'];assert len(set(flat))==23 and sorted(flat)==list(range(23))
        for task in stream['tasks']:
            for old,new in zip(task['legacy_arrival_indices'],task['new_classifier_indices']):
                label=legacy['order'][old];rc={k:sum(int(rows[i]['original_label'])==label for i in ids) for k,ids in roles.items()}
                assert all(v>0 for v in rc.values()),('EMPTY_ROLE',stream['name'],old)
                mappings.append([stream['name'],old,label,legacy['names'][old],new,task['task_number_1based'],counts[old],counts[old]<250,old in [1,2,12],rc['S'],rc['B'],rc['F']])
            tasks.append([stream['name'],task['task_number_1based'],len(task['legacy_arrival_indices']),task['train_total'],task['within_task_imbalance_ratio_max_over_min'],task['cumulative_seen_classes']])
        stream['history_access']='all arrived TRAIN teacher; fixed arrived S/B module; not bounded memory'
        stream['memory_budget_note']='Not a finite-memory experiment; null is unspecified/unbounded history, not zero.'
    assert [t['train_total'] for t in streams[0]['tasks']]==[3331,2400,892,165,33]
    assert [t['train_total'] for t in streams[1]['tasks']]==[1563,1468,1505,1223,1062]
    write(root/'hk_task_settings.json',{'streams':streams,'binding_evidence':{'passed':True,'TRAIN':6821,'stable_ids_unique':True,'all_roles_nonempty':True,'component_disjoint':True,'VAL_TEST_read':False,'no_hashing':True},'training_streams':[streams[0]['name'],streams[1]['name']],'order1_order2_sealed_not_trained':True})
    for name,header,values in [('CLASS_MAPPING.csv',['stream','legacy_index','original_label','name_from_legacy_config','classifier_index','task','TRAIN_count','tail','tiny','S_count','B_count','F_count'],mappings),('TASK_COUNTS.csv',['stream','task','new_classes','TRAIN_rows','task_IR','seen_classes'],tasks)]:
        with (root/name).open('w') as f:w=csv.writer(f);w.writerow(header);w.writerows(values)
    write(root/'BINDING_VERIFICATION.json',{'passed':True,'streams_defined':4,'streams_to_train':2,'TRAIN_rows':6821,'class_count':23,'per_task_classes':[5,5,5,4,4],'stable_id_roles_bound':True,'component_disjoint':True,'tail_original_count_lt250':True,'tiny_original_labels':[legacy['order'][k] for k in [1,2,12]],'VAL_TEST_reads':0})
    return streams[:2]
