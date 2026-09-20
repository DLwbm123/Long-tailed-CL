#!/usr/bin/env python3
"""The approved V2 association-graph cleaning rule; no training or test prediction."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def clean(rows, metadata):
    rows = [dict(r, original_label=int(r['original_label'])) for r in rows]
    assert len({r['sample_id'] for r in rows}) == len(rows)
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b: parent[max(a,b)] = min(a,b)
    first = {}
    invalid = {'','na','n/a','none','nan','null','unknown'}
    for i,r in enumerate(rows):
        lesion = metadata.get(r['sample_id'],{}).get('lesion_id','').strip()
        r['lesion_id'] = '' if lesion.lower() in invalid else 'isic_metadata_v1:'+lesion
        keys=[('content',r['file_hash'])]
        if r['lesion_id']:keys.append(('lesion',r['lesion_id']))
        for key in keys:
            if key in first:union(i,first[key])
            else:first[key]=i
    components=defaultdict(list)
    for i,r in enumerate(rows):components[find(i)].append(r)
    priority={'train':0,'val':1,'test':2}
    for group in components.values():
        cid=hashlib.sha256('\n'.join(sorted(r['sample_id'] for r in group)).encode()).hexdigest()
        highest=max((r['split'] for r in group),key=priority.get)
        conflict=len({r['original_label'] for r in group})>1
        for r in group:
            r['original_split']=r['split'];r['identity_component']=cid;r['component_size']=len(group)
            r['component_priority_split']=highest
            if conflict:r['disposition']='quarantine_label_conflict'
            elif not r['lesion_id']:r['disposition']='quarantine_unknown_lesion'
            elif r['split']!=highest:r['disposition']='excluded_lower_split_priority'
            else:r['disposition']='retained'
    hashes={}
    for r in sorted(rows,key=lambda r:r['sample_id']):
        if r['disposition']!='retained':continue
        if r['file_hash'] in hashes:r['disposition']='excluded_duplicate_content'
        else:hashes[r['file_hash']]=r['sample_id']
    return rows


def summarize(rows):
    kept=[r for r in rows if r['disposition']=='retained']
    labels=list(range(8));by_class=[];failures=[];low=[]
    for label in labels:
        row={'original_label':label,'original_images':sum(r['original_label']==label for r in rows)}
        for split in ('train','val','test'):
            selected=[r for r in kept if r['original_label']==label and r['split']==split]
            row[f'n_{split}_images']=len(selected)
            row[f'n_{split}_lesions']=len({r['lesion_id'] for r in selected})
            n=len({r['identity_component'] for r in selected})
            row[f'n_{split}_components']=n
            if n<(2 if split=='train' else 1):failures.append({'label':label,'split':split,'components':n})
            if split=='test' and n<20:low.append(label)
        row['retained_images']=sum(row[f'n_{s}_images'] for s in ('train','val','test'))
        row['excluded_images']=row['original_images']-row['retained_images']
        row['retention_fraction']=row['retained_images']/row['original_images']
        row['n_identity_components']=len({r['identity_component'] for r in kept if r['original_label']==label})
        by_class.append(row)
    intersections={}
    for field in ('sample_id','file_hash','lesion_id','identity_component'):
        for a,b in [('train','val'),('train','test'),('val','test')]:
            overlap={r[field] for r in kept if r['split']==a}&{r[field] for r in kept if r['split']==b}
            intersections[f'{field}:{a}:{b}']=len(overlap)
    assert not any(intersections.values()),intersections
    assert all(r['split']==r['original_split'] and r['lesion_id'] for r in kept)
    assert len({r['file_hash'] for r in kept})==len(kept)
    for field in ['lesion_id','file_hash','identity_component']:
        grouped=defaultdict(set)
        for r in kept:grouped[r[field]].add(r['original_label'])
        assert all(len(v)==1 for v in grouped.values())
    rank=sorted(by_class,key=lambda r:(r['n_train_images'],r['original_label']))
    def ratio(field):
        values=[r[field] for r in by_class]
        return max(values)/min(values) if min(values)>0 else None
    return {'protocol_id':'isic19_fopro_s1_documented_lesion_disjoint_v2',
            'status':'BLOCKED_V2_CLASS_SUPPORT' if failures else 'V2_DATA_SUPPORT_PASS',
            'class_support_failures':failures,'LOW_TEST_SUPPORT':low,'class_counts':by_class,
            'original_count':len(rows),'retained_count':len(kept),
            'retained_split_counts':dict(Counter(r['split'] for r in kept)),
            'disposition_counts':dict(Counter(r['disposition'] for r in rows)),
            'pairwise_intersections':intersections,'train_image_imbalance_ratio':ratio('n_train_images'),
            'train_lesion_imbalance_ratio':ratio('n_train_lesions'),
            'frequency_groups':{'tail_rank2':[r['original_label'] for r in rank[:2]],'mid_rank4':[r['original_label'] for r in rank[2:6]],'head_rank2':[r['original_label'] for r in rank[6:]]},
            'lesion_coverage':1.0,'patient_isolation':'UNKNOWN','development_exposure':'UNKNOWN',
            'semantic_map':'SEMANTIC_MAP_UNVERIFIED','model_test_predictions':0,
            'selection_bias':'Exclusion of unknown lesion IDs restricts coverage; not patient-disjoint or external validation.'}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(args):
    audit=args.v1_audit
    reference=json.loads((audit/'data/dataset_audit.json').read_text())
    with (audit/'data/image_manifest.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    indexed={(r['split'],r['sample_id'],int(r['original_label'])) for r in rows}
    actual=set()
    for split in ['train','val','test']:
        p=args.split_root/f'{split}_skin1_0.01.csv'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==reference['split_sha256'][split]
        with p.open(newline='') as f:
            actual.update((split,r['image'],int(r['finding'])) for r in csv.DictReader(f))
    assert indexed==actual and len(rows)==len(actual)
    # V1 decoded and hashed every image. Reuse that record when size is nonzero
    # and the file predates the completed audit; only changed candidates are re-read.
    audit_start=(audit/'data/dataset_audit.json').stat().st_mtime-reference['elapsed_seconds']
    reused=0;changed=0
    for r in rows:
        p=args.image_root/r['relative_path'];st=p.stat();assert st.st_size>0
        if st.st_mtime>audit_start:
            data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==r['file_hash'],str(p)
            from PIL import Image
            with Image.open(p) as im:im.load()
            changed+=1
        else:reused+=1
    metadata=json.loads((audit/'matched_group_metadata_private.json').read_text())
    m=json.loads((audit/'group_metadata_audit.json').read_text())
    assert hashlib.sha256(Path(m['source_path']).read_bytes()).hexdigest()==m['source_sha256']
    cleaned=clean(rows,metadata);result=summarize(cleaned)
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'all_dispositions_private.csv',cleaned)
    result['manifest_sha256']={}
    for split in ['train','val','test']:
        selected=sorted((r for r in cleaned if r['split']==split and r['disposition']=='retained'),key=lambda r:r['sample_id'])
        if selected:
            write_csv(args.output/f'{split}.csv',selected)
            result['manifest_sha256'][split]=hashlib.sha256((args.output/f'{split}.csv').read_bytes()).hexdigest()
    write_csv(args.output/'class_counts.csv',result['class_counts'])
    result['input_verification']={'reused_V1_image_hashes_and_decode':reused,'changed_candidates_rehashed':changed,'basis':'same CSV identity, nonzero file size, no newer mtime than V1 audit; trusted filesystem, not a fresh bytewise recheck','metadata_sha256':m['source_sha256'],'source_manifest_sha256':hashlib.sha256((audit/'data/image_manifest.csv').read_bytes()).hexdigest()}
    (args.output/'V2_DATASET_SUMMARY.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1-audit',type=Path,required=True);p.add_argument('--split-root',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
