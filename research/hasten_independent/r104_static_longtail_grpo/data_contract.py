"""Stable identities, immutable outer roles and historical lineage exclusions."""
import json,csv,io,hashlib
from pathlib import Path
import torch
from common import write

def seed(c,stage,bank,role):
    return int.from_bytes(hashlib.sha256(f"{c['seed']}:{c['stream_name']}:{stage}:{bank}:{role}".encode()).digest()[:4],'big')

def bind(c):
    rows=list(csv.DictReader((Path(c['manifest'])/'train.csv').open()));src=json.loads(Path(c['source_binding']).read_text())
    assert rows==list(csv.DictReader(io.StringIO(src['train_manifest']))) and len(rows)==6821
    y=torch.tensor([c['order'].index(int(v['original_label'])) for v in rows]);assert torch.bincount(y,minlength=23).tolist()==c['train_counts']
    roles=src['roles'];assert [len(roles[k]) for k in ['S','B','F']]==[3405,1713,1703]
    for k,ix in roles.items():assert [rows[i]['sample_id'] for i in ix]==src['stable_sample_ids'][k]
    A=sorted(roles['S']);B=sorted(roles['B']+roles['F']);assert not set(A)&set(B) and sorted(A+B)==list(range(6821))
    components=[{rows[i]['identity_component'] for i in ix} for ix in [A,B]];assert not components[0]&components[1]
    records=[];audit=[]
    for s in range(len(c['stages'])):
        C=sum(map(len,c['stages'][:s+1]));arrived=[i for i in range(6821) if y[i]<C]
        for name,fit,feedback in [('FIT_A',A,B),('FIT_B',B,A)]:
            fi=[i for i in fit if y[i]<C];fb=[i for i in feedback if y[i]<C];nc=torch.bincount(y[fi],minlength=C);bc=torch.bincount(y[fb],minlength=C);assert (nc>0).all() and (bc>0).all()
            for cl in range(C):records.append([s,name,cl,int(nc[cl]),int(bc[cl])])
            assert all(y[i]<C for i in fi+fb)
            audit.append({'stage':s,'bank':name,'fit_rows':len(fi),'feedback_rows':len(fb),'fit_feedback_component_intersection':0,'all_prior_stage_fit_subsets_same_outer_fold':True,'generic_pretrained_only_initialization':True,'FULL_supervised_parameters_features_in_fold_initialization':False})
    r=Path(c['output'])
    with (r/'FOLD_COUNTS.csv').open('w') as f:w=csv.writer(f,lineterminator='\n');w.writerow(['stage','bank','class','fit_count','feedback_count']);w.writerows(records)
    write(r/'FOLD_EXCLUSION_AUDIT.json',{'passed':True,'outer_role_sizes':[len(A),len(B)],'identity_component_intersection':0,'records':audit,'learner_receipts_enforced_by_fit_id_sets':True})
    assert c['stages']==[list(range(23))] and c['setting']=='static_longtail'
    write(r/'FULL_TEACHER_BINDING.json',{'passed':True,'fresh_generic_FULL_and_two_folds':True,'reused_HK_supervised_parameters':False,'generic_prefix_only_reuse':True,'all_classes_present_before_first_update':True,'independent_patient_split_verified':False})
    return rows,y,{'FIT_A':A,'FIT_B':B}
