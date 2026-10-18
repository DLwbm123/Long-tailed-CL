"""Metadata-only preparation for the existing test/reserved benchmark."""
import csv,json
from pathlib import Path
from collections import defaultdict
from run import write_json


def prepare(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.private.json',cfg)
    assert cfg['local_lambda']==.5 and cfg['primary']=='consensus'
    source=json.loads(Path(cfg['transfer_config']).read_text());audit={}
    for case in source['datasets']:
        n=case['name'];order=case['order'][:case['old']+case['new']];spec=cfg['datasets'][n]
        with Path(spec['manifest']).open() as f:raw=list(csv.DictReader(f))
        chosen=[r for r in raw if int(r['original_label']) in order]
        rows=[dict(relative_path=r['relative_path'],identity_component=r['identity_component'],label=order.index(int(r['original_label']))) for r in chosen]
        assert len(rows)==cfg['expected_test_images'][n]
        assert len({r['relative_path'] for r in rows})==len(rows)
        assert all(r['identity_component'] and not Path(r['relative_path']).is_absolute() and '..' not in Path(r['relative_path']).parts for r in rows)
        labels=defaultdict(set)
        for r in rows:labels[r['identity_component']].add(r['label'])
        assert all(len(s)==1 for s in labels.values())
        overlaps={}
        for role in ['S_old','S_new','Qfit','Qgate','E']:
            prior=json.loads((Path(cfg['prior_rows'])/(n+'.'+role+'.rows.private.json')).read_text())
            overlaps[role]={k:len({r[k] for r in rows}&{r[k] for r in prior}) for k in ['relative_path','identity_component']}
            assert not any(overlaps[role].values())
        # Existing manifest content identifiers only; no file rehashing.
        content_overlap={}
        for part in ['train','val']:
            with (Path(spec['support_manifest'])/(part+'.csv')).open() as f:known=list(csv.DictReader(f))
            content_overlap[part]=len({r['file_hash'] for r in chosen if r.get('file_hash')}&{r['file_hash'] for r in known if r.get('file_hash')})
            assert content_overlap[part]==0
        missing=sum(not (Path(spec['images'])/r['relative_path']).is_file() for r in rows)
        assert missing==0,'Missing test images'
        counts=[dict(class_index=c,original_label=lab,images=sum(r['label']==c for r in rows),known_groups=len({r['identity_component'] for r in rows if r['label']==c})) for c,lab in enumerate(order)]
        assert [r['known_groups'] for r in counts]==cfg['expected_test_groups'][n]
        write_json(out/(n+'.test.rows.private.json'),rows)
        audit[n]=dict(counts=counts,overlap=overlaps,existing_content_id_overlap=content_overlap,missing_images=missing,patient_isolation='UNKNOWN',scope='first transition only',test_source=spec['public_split_name'])
    write_json(out/'DATA_AUDIT.json',audit)
    write_json(out/'status.json',dict(state='PREPARED',models_run=False))
    return audit
