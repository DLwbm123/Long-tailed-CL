"""Prepare official RFMiD train/validation only; do not infer a benchmark mapping."""
import csv
import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import time
import zipfile

DISEASES=('DR','ARMD','MH','DN','MYA','BRVO','TSLN','ERM','LS','MS','CSR','ODC','CRVO',
          'TV','AH','ODP','ODE','ST','AION','PT','RT','RS','CRS','EDN','RPEC','MHL','RP',
          'CWS','CB','ODPM','PRH','MNF','HR','CRAO','TD','CME','PTCR','CF','VH','MCA',
          'VS','BRAO','PLQ','HPED','CL')


def read_labels(text,expected):
    reader=csv.DictReader(io.StringIO(text));assert reader.fieldnames==['ID','Disease_Risk',*DISEASES]
    rows=list(reader);assert len(rows)==expected
    ids=set()
    for row in rows:
        name=row['ID'];assert name.isdecimal() and int(name)>0 and name==str(int(name)) and name not in ids
        assert all(row[k] in ('0','1') for k in ('Disease_Risk',*DISEASES))
        ids.add(name)
    return rows


def prepare(config):
    begun=time.monotonic();root=Path(config['root']);assert root.is_dir()
    mount=subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip()
    assert mount.startswith('nfs') and shutil.disk_usage(root).free>20*1024**3
    probe=root/'.probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
    data=root/'data';data.mkdir(exist_ok=False);(data/'images').mkdir();(data/'labels').mkdir()
    audit=dict(status='DATA_READY_LABEL_MAPPING_PENDING',test_accessed=False,raw_disease_labels=list(DISEASES),
               patient_identifiers_available=False,patient_isolation_verified=False,
               image_content_duplicates_checked=False,split_ids_global=False,splits={})
    for split,archive,prefix,folder,labelname,n in (
        ('train','Training_Set.zip','Training_Set','Training','RFMiD_Training_Labels.csv',1920),
        ('val','Evaluation_Set.zip','Evaluation_Set','Validation','RFMiD_Validation_Labels.csv',640)):
        with zipfile.ZipFile(root/'incoming'/archive) as z:
            entries=[v for v in z.infolist() if not v.is_dir() and not Path(v.filename).name.startswith('._') and '__MACOSX/' not in v.filename]
            names=[v.filename for v in entries];assert len(names)==len(set(names))==n+1
            labelpath=prefix+'/'+labelname;text=z.read(labelpath).decode('utf-8-sig');rows=read_labels(text,n)
            required={prefix+'/'+folder+'/'+row['ID']+'.png' for row in rows}|{labelpath}
            assert set(names)==required
            (data/'labels'/(split+'.csv')).write_text(text);dest=data/'images'/split;dest.mkdir()
            for entry in entries:
                path=Path(entry.filename)
                assert not path.is_absolute() and '..' not in path.parts and not stat.S_ISLNK(entry.external_attr>>16)
                if entry.filename==labelpath:continue
                with z.open(entry) as source,(dest/path.name).open('xb') as target:shutil.copyfileobj(source,target)
            from PIL import Image
            first=rows[0]['ID']+'.png'
            with Image.open(dest/first) as image:
                image.load();size=image.size;mode=image.mode;assert min(size)>0
            counts={k:sum(int(row[k]) for row in rows) for k in DISEASES}
            risk_disagrees=sum(int(row['Disease_Risk'])!=int(any(row[k]=='1' for k in DISEASES)) for row in rows)
            audit['splits'][split]=dict(images=n,labels_n=n,label_ids_unique=True,image_label_coverage_exact=True,
                binary_labels=True,disease_positive_counts=counts,disease_negative_counts={k:n-v for k,v in counts.items()},
                disease_risk_positive=sum(int(row['Disease_Risk']) for row in rows),
                normal_positive=sum(row['Disease_Risk']=='0' for row in rows),risk_or_disease_disagreement_n=risk_disagrees,
                multi_disease_images_n=sum(sum(int(row[k]) for k in DISEASES)>1 for row in rows),
                one_image_fully_decoded=True,decoded_size=list(size),decoded_mode=mode,archive_bytes=(root/'incoming'/archive).stat().st_size)
    audit['cpu_extraction_seconds']=time.monotonic()-begun
    (data/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (data/'READY.json').write_text(json.dumps(dict(status=audit['status'],test_accessed=False,training_ready=False))+'\n')
    return audit


if __name__=='__main__':print(json.dumps(prepare(json.load(__import__('sys').stdin))))
