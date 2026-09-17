"""HK1-M1 CPU identity audit and fixed train/val protocol. No model imports."""
import csv
import hashlib
import io
import json
import os
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from audit_hyperkvasir_hk1_inputs import legacy_membership

ORDERS = {
    1993: [0,4,12,8,2,11,17,3,21,16,22,13,18,14,9,19,15,6,5,20,10,7,1],
    1994: [22,16,12,13,21,4,19,17,6,11,3,15,0,9,5,7,18,1,10,14,20,8,2],
    1995: [9,13,16,14,15,5,12,1,8,22,19,17,18,6,20,0,3,11,2,21,4,10,7],
}
SEEN = [13,15,17,19,21,23]
PERM = [0,16,1,2,8,3,4,11,12,5,6,7,13,9,14,15,17,18,19,20,21,22,10]
COUNTS = [41,53,646,1148,1009,1002,989,403,260,6,9,131,1028,999,391,764,35,201,11,443,28,133,932]
OFFICIAL_SHA = '0230b465bdf73c23e949c74e1d4d68c4905a758195ea6b69b10f32c71a3be426'
OFFICIAL_BLOB = '0d8abe6781a82c4452416390d93a176f875a6f23'
MAPPING_BLOB = 'ce84182ae5e68c54dfb5e6ee9afa99c32f2187c5'


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def json_hash(x): return sha_bytes(json.dumps(x, sort_keys=True, separators=(',',':')).encode())
def blob(b): return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def read_csv(p, delimiter=','):
    with Path(p).open() as f: return list(csv.DictReader(f, delimiter=delimiter))
def write(p, x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.part');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def csvwrite(p, rows):
    assert rows
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def clean(rows):
    groups=defaultdict(list)
    for row in rows: groups[row['identity_component']].append(row)
    kept=[];excluded=[];conflict_groups=0;cross_reserved_groups=0
    for component, members in sorted(groups.items()):
        members=sorted(members,key=lambda r:r['sample_id'])
        if len({r['original_label'] for r in members})>1:
            conflict_groups+=1
            excluded.extend(dict(r,exclusion_reason='EXACT_CONTENT_LABEL_CONFLICT') for r in members)
            continue
        has_reserved=any(r['outer_membership']=='reserved' for r in members)
        if has_reserved and any(r['outer_membership']=='outer_train' for r in members):cross_reserved_groups+=1
        eligible=[]
        for r in members:
            if has_reserved and r['outer_membership']=='outer_train':
                excluded.append(dict(r,exclusion_reason='RESERVED_PRIORITY'))
            else:eligible.append(r)
        kept.append(eligible[0])
        excluded.extend(dict(r,exclusion_reason='WITHIN_OUTER_SPLIT_EXACT_DUPLICATE') for r in eligible[1:])
    return sorted(kept,key=lambda r:r['sample_id']),excluded,dict(conflict_groups=conflict_groups,cross_reserved_groups=cross_reserved_groups)


def selfcheck():
    inverse=np.argsort(PERM);assert np.array_equal(np.array(PERM)[inverse],np.arange(23))
    scores=np.eye(23);scores[0]=0;scores[0,3]=2
    assert np.mean(scores.argmax(1)==np.arange(23))==np.mean(scores[:,inverse].argmax(1)==np.array(PERM))
    for order in ORDERS.values():
        assert sorted(order)==list(range(23))
        head={c:i for i,c in enumerate(order)};counts={c:2*c+3 for c in range(23)};lt=[counts[c] for c in order]
        for c in range(23):assert order[head[c]]==c and lt[head[c]]==counts[c]
        assert [len(order[:n]) for n in SEEN]==SEEN
    rs=[dict(sample_id=n,original_label=c,outer_membership=s,identity_component=g)
        for n,c,s,g in [('a',0,'outer_train','x'),('b',1,'reserved','x'),('c',2,'outer_train','y'),
                         ('d',2,'reserved','y'),('e',3,'outer_train','z'),('f',3,'outer_train','z')]]
    kept,excluded,stats=clean(rs)
    assert [r['sample_id'] for r in kept]==['d','e']
    assert Counter(r['exclusion_reason'] for r in excluded)=={'EXACT_CONTENT_LABEL_CONFLICT':2,'RESERVED_PRIORITY':1,'WITHIN_OUTER_SPLIT_EXACT_DUPLICATE':1}
    assert stats=={'conflict_groups':1,'cross_reserved_groups':1}
    return dict(status='PASS',bijection_inverse=True,synthetic_relabel_metrics_invariant=True,
                unchanged_numeric_orders=True,target_count_roundtrip=True,stage_prefixes=True,
                conflict_quarantine=True,reserved_priority=True,stable_duplicate_representative=True)


def main():
    started=time.monotonic();cfg=json.loads(Path(os.environ['P19_CONFIG']).read_text())
    root=Path(cfg['output_root']);pub=root/'public';private=root/'private'
    pub.mkdir(parents=True,exist_ok=False);private.mkdir()
    access=dict(new_formal_S0_runs=0,new_formal_neural_training_epochs=0,formal_training_optimizer_steps=0,
        engineering_optimizer_steps=0,encoder_forward_calls=0,new_test_predictions=0,new_test_feature_reads=0,
        new_test_model_forwards=0,historical_prediction_payload_reads=0,reserved_identity_audit_file_reads=0,
        reserved_identity_audit_decodes=0,outer_train_identity_audit_file_reads=0,outer_train_identity_audit_decodes=0,
        monitoring_tasks_created=0,further_experiments_started=False)
    try:
        tests=selfcheck();mapping_raw=Path(cfg['reference_mapping_csv']).read_bytes();assert blob(mapping_raw)==MAPPING_BLOB
        oldmap=read_csv(cfg['reference_mapping_csv']);assert [int(r['official_id']) for r in oldmap]==PERM
        assert [int(r['n_official_images']) for r in oldmap]==COUNTS
        namespace=[dict(legacy_id=int(r['legacy_id']),legacy_class_name=r['legacy_class_name'],
            official_name_sorted_id=int(r['official_id']),official_class_name=r['official_class_name'],
            whitelist_count=int(r['n_official_images'])) for r in oldmap]
        canonical={r['official_name_sorted_id']:r['official_class_name'] for r in namespace}
        assert [canonical[c] for c in range(23)]==sorted(canonical.values())
        assert sum(r['legacy_class_name']!=r['official_class_name'] for r in namespace)==8
        assert sum(r['legacy_id']!=r['official_name_sorted_id'] for r in namespace)==20
        csvwrite(pub/'CLASS_NAMESPACE_HK1_M1.csv',namespace)
        raw=Path(cfg['official_csv']).read_bytes();assert blob(raw)==OFFICIAL_BLOB and sha_bytes(raw)==OFFICIAL_SHA
        official=read_csv(cfg['official_csv'],';');official_by_name={Path(r['file-name']).name:r for r in official}
        assert len(official)==len(official_by_name)==10662
        historical=read_csv(cfg['legacy_alignment']);seed1=[r for r in historical if r['seed']=='1']
        assert len(seed1)==len({r['basename'] for r in seed1})==10662
        image_root=Path(cfg['images']);rows=[];membership=[];observed_counts=Counter()
        for old in sorted(seed1,key=lambda r:r['basename']):
            name=old['basename'];official_row=official_by_name[name];lid=int(old['class_id']);entry=namespace[lid]
            path=Path(old['path']);relative=path.relative_to(image_root)
            assert path.is_file() and path.parent.name==old['class_name']==entry['legacy_class_name']
            assert official_row['class-name']==entry['official_class_name']
            outer={'train':'outer_train','test':'reserved'}[old['membership']]
            row=dict(sample_id=name,relative_path=str(relative),legacy_id=lid,legacy_class_name=entry['legacy_class_name'],
                     official_name_sorted_id=PERM[lid],official_class_name=entry['official_class_name'],
                     original_label=PERM[lid],outer_membership=outer,official_fold=int(official_row['split-index']))
            rows.append(row);membership.append([name,outer]);observed_counts[lid]+=1
        assert [observed_counts[c] for c in range(23)]==COUNTS
        assert Counter(r['outer_membership'] for r in rows)=={'outer_train':8519,'reserved':2143}
        for lid in range(23):
            selected=[r for r in rows if r['legacy_id']==lid]
            check=legacy_membership([r['sample_id'] for r in selected],lid,1)
            assert all(check[r['sample_id']]==('train' if r['outer_membership']=='outer_train' else 'test') for r in selected)
        membership_digest=json_hash(membership)
        write(pub/'OUTER_MEMBERSHIP_INVARIANCE_HK1_M1.json',dict(status='PASS',before_cleanup=True,outer_train=8519,reserved=2143,
            union=10662,intersection=0,sample_membership_differences=0,membership_sha256_before=membership_digest,
            membership_sha256_after=membership_digest,hash_format='SHA256 canonical JSON sorted [basename,outer_membership] pairs',
            reconstructed=False,legacy_rule_checked_with_legacy_id=True,new_ID_used_to_resplit=False))
        write(pub/'CLASS_MAPPING_LOCK_HK1_M1.json',dict(status='MAPPING_AMENDMENT_APPLIED',protocol_revision='HK1-M1',
            label_namespace='official_name_sorted',integer_ID_source='project encoding from official names, not an upstream CSV integer ID',
            official_csv_sha256=OFFICIAL_SHA,official_csv_git_blob=OFFICIAL_BLOB,reference_alias_git_blob=MAPPING_BLOB,
            reference_alias_sha256=sha_bytes(mapping_raw),namespace_sha256=sha_bytes((pub/'CLASS_NAMESPACE_HK1_M1.csv').read_bytes()),
            legacy_to_canonical=PERM,canonical_to_legacy=np.argsort(PERM).tolist(),source_documents=cfg['source_documents']))
        tasks={str(seed):[dict(stage=t,new_classes=[dict(original_label=c,official_class_name=canonical[c],legacy_id=PERM.index(c),head_id=order.index(c))
                    for c in order[(0 if t==0 else SEEN[t-1]):n]],seen_original_labels=order[:n]) for t,n in enumerate(SEEN)] for seed,order in ORDERS.items()}
        write(pub/'TASK_SEMANTICS_HK1_M1.json',dict(orders=ORDERS,order_namespace='official_name_sorted',order_remapped_again=False,stages=tasks))
        # The auxiliary "Video file" column is image-unique, not a verified exam/patient grouping key.
        auxiliary=read_csv(image_root/'image-labels.csv')
        assert len(auxiliary)==10662 and {r['Video file'] for r in auxiliary}=={Path(r['sample_id']).stem for r in rows}
        seen_byte_to_pixel={}
        for i,r in enumerate(rows):
            prefix='reserved' if r['outer_membership']=='reserved' else 'outer_train'
            b=(image_root/r['relative_path']).read_bytes();access[prefix+'_identity_audit_file_reads']+=1
            r['file_hash']=sha_bytes(b);r['file_bytes']=len(b)
            with Image.open(io.BytesIO(b)) as im:
                rgb=im.convert('RGB');rgb.load();access[prefix+'_identity_audit_decodes']+=1
                r['width'],r['height']=rgb.size
                r['pixel_hash']=sha_bytes(b'HK1_RGB\0'+struct.pack('<II',*rgb.size)+rgb.tobytes())
            previous=seen_byte_to_pixel.setdefault(r['file_hash'],r['pixel_hash']);assert previous==r['pixel_hash']
            r['identity_component']=r['pixel_hash'];r['verified_group']=r['identity_component']
            if (i+1)%1000==0:
                write(pub/'ACCESS_AUDIT.json',access);print('IDENTITY_ROWS',i+1,flush=True)
        csvwrite(private/'all_identity_audit.csv',rows)
        kept,excluded,clean_stats=clean(rows)
        if excluded:csvwrite(private/'exclusions.csv',excluded)
        outer=[r for r in kept if r['outer_membership']=='outer_train'];reserved=[r for r in kept if r['outer_membership']=='reserved']
        reserved_groups={r['verified_group'] for r in reserved};assert not reserved_groups & {r['verified_group'] for r in outer}
        candidates=[];candidate_rows=[];selected=None
        for fold in range(5):
            vg={r['verified_group'] for r in outer if r['official_fold']==fold}
            val=[r for r in outer if r['verified_group'] in vg];fit=[r for r in outer if r['verified_group'] not in vg]
            ok=True
            for c in range(23):
                fr=[r for r in fit if r['original_label']==c];vr=[r for r in val if r['original_label']==c]
                valid=len(fr)>=2 and len(vr)>=1 and len({r['verified_group'] for r in fr})>=2 and len({r['verified_group'] for r in vr})>=1
                ok &= valid
                candidate_rows.append(dict(fold=fold,original_label=c,official_class_name=canonical[c],n_fit=len(fr),n_val=len(vr),
                    n_fit_groups=len({r['verified_group'] for r in fr}),n_val_groups=len({r['verified_group'] for r in vr}),coverage_pass=valid))
            candidates.append(dict(fold=fold,all_23_classes_pass=ok,fit=len(fit),val=len(val),grouping_adjustments=sum(r['official_fold']!=fold for r in val)))
            if ok and selected is None:selected=(fold,fit,val)
        csvwrite(pub/'FOLD_COVERAGE.csv',candidate_rows)
        write(pub/'DATA_AUDIT_HK1.json',dict(status='DATA_PASS' if selected else 'BLOCKED_HK1_CLASS_COVERAGE',protocol_revision='HK1-M1',
            whitelist_images=10662,classes=23,identity_components=len({r['identity_component'] for r in rows}),
            duplicate_extra_images=10662-len({r['identity_component'] for r in rows}),**clean_stats,
            exclusions_by_reason=dict(Counter(r['exclusion_reason'] for r in excluded)),excluded_images=len(excluded),
            cleaned_outer_train=len(outer),cleaned_reserved=len(reserved),candidate_folds=candidates,
            selected_official_fold=selected[0] if selected else None,patient_disjoint='UNKNOWN',exam_disjoint='UNKNOWN',
            known_identity_disjoint=True,known_group_source='exact file and decoded RGB identity; no verified wider patient/exam keys',
            auxiliary_video_file_field='10662 unique image identifiers, not inferred patient/video IDs',
            identity_audit_seconds=time.monotonic()-started,metadata_and_mapping_selfcheck=tests))
        if selected is None:
            print('BLOCKED_HK1_CLASS_COVERAGE',flush=True);return
        fold,fit,val=selected;counts={c:sum(r['original_label']==c for r in fit) for c in range(23)}
        ranked=sorted(counts,key=lambda c:(counts[c],c));groups=dict(tail_rank8=ranked[:8],middle_rank7=ranked[8:15],head_rank8=ranked[15:])
        class_rows=[]
        for c in range(23):
            vr=[r for r in val if r['original_label']==c]
            class_rows.append(dict(original_label=c,official_class_name=canonical[c],legacy_id=PERM.index(c),n_train=counts[c],n_val=len(vr),
                n_train_components=counts[c],n_val_components=len({r['identity_component'] for r in vr}),frequency_group=next(k for k,v in groups.items() if c in v),
                coarse_validation_resolution=len(vr)<=2))
        csvwrite(pub/'CLASS_COUNTS_AND_GROUPS.csv',class_rows)
        manifest_hashes={}
        for split,rs in [('train',fit),('val',val),('reserved',reserved)]:
            rows_split=[dict(r,split=split) for r in rs];csvwrite(private/(split+'.csv'),rows_split)
            manifest_hashes[split]=sha_bytes((private/(split+'.csv')).read_bytes())
        exposure=[]
        for seed in (1,2,3):
            old={r['basename']:r['membership'] for r in historical if int(r['seed'])==seed}
            for name,rs in [('train',fit),('val',val),('reserved',reserved)]:
                exposure.append(dict(historical_seed=seed,historical_fold=1,new_split=name,**dict(Counter('historical_'+old[r['sample_id']] for r in rs))))
        targeted=[]
        for split,path in cfg['historical_targeted_manifests'].items():
            names={r['basename'] for r in read_csv(path)}
            targeted.append(dict(historical_role=split,n_unique=len(names),fit_overlap=sum(r['sample_id'] in names for r in fit),val_overlap=sum(r['sample_id'] in names for r in val)))
        write(pub/'EXPOSURE_LEDGER_HK1.json',dict(status='IDENTITY_ONLY',outer_protocol_overlaps=exposure,targeted_legacy_val_helper_overlaps=targeted,
            historical_performance_payload_read=False,unknown_additional_historical_exposure_possible=True,independent_confirmation=False))
        for seed,order in ORDERS.items():
            lt=[counts[c] for c in order];assert max(lt)<12726
            for r in fit:assert lt[order.index(r['original_label'])]==counts[r['original_label']]
        protocol=dict(status='DATA_LOCKED_ENGINEERING_PENDING',protocol_revision='HK1-M1',protocol_id='hyperkvasir23_legacy_s1f1_official5val_hk1_m1',
            label_namespace='official_name_sorted',order_namespace='official_name_sorted',legacy_outer_membership_preserved=True,
            class_orders=ORDERS,task_sizes=[13,2,2,2,2,2],seen_classes=SEEN,selected_official_fold=fold,
            n_train=len(fit),n_val=len(val),n_reserved=len(reserved),fit_counts=counts,frequency_groups=groups,
            actual_fit_imbalance_ratio=max(counts.values())/min(counts.values()),manifest_sha256=manifest_hashes,
            outer_membership_sha256=membership_digest,order_sha256=json_hash(ORDERS),source_documents=cfg['source_documents'],
            methods=['A-CB','S-J-CB','S-M-CB'],main_comparison='S-J-CB minus A-CB',ridge_lambda=.001,ridge_bias=False,
            ridge_objective='mean class mean squared error',ridge_statistics_dtype='float64',neural_dtype='float32',batch_size=48,
            planned_S0_epochs_per_run=10,incremental_neural_updates=0,planned_val_metric_rows=54,planned_val_per_class_rows=972,
            bootstrap_resamples=2000,bootstrap_seed=44001,bootstrap_mode='class-stratified exact-identity-component',parents_resampled=False,
            prospective_input_staging_bytes=sum(r['file_bytes'] for r in fit+val),formal_training_released=False,new_test_model_forwards=0)
        write(pub/'PROTOCOL_HK1.json',protocol)
        write(pub/'MAPPING_AUDIT_HK1_M1.json',dict(tests,sample_identity_bijection=True,legacy_outer_membership_preserved=True,
            approved_alias_count=8,numeric_ID_differences=20,actual_fit_count_target_alignment=True,full_neural_engineering='PENDING'))
        # Input transfer manifest has only train/val. The model runtime never receives reserved.csv.
        csvwrite(private/'input_staging_manifest.csv',[dict(sample_id=r['sample_id'],relative_path=r['relative_path'],file_hash=r['file_hash'],file_bytes=r['file_bytes']) for r in sorted(fit+val,key=lambda r:r['sample_id'])])
        (private/'input_files.txt').write_text(''.join(r['relative_path']+'\n' for r in sorted(fit+val,key=lambda r:r['sample_id'])))
        write(pub/'DATA_CODE_LOCK.json',dict(source_reference='b4a46c022b098e496e2292488a16767eaf28211b',source_sha256=sha_bytes(Path(__file__).read_bytes())))
        print('DATA_PASS',len(fit),len(val),'fold',fold,'IR',protocol['actual_fit_imbalance_ratio'],flush=True)
    finally:
        access['identity_audit_wall_seconds']=time.monotonic()-started
        write(pub/'ACCESS_AUDIT.json',access)


if __name__=='__main__':main()
