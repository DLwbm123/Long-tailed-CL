"""Losslessly preserve this experiment's closed qualification fixtures under NFS accounting."""
import gzip,io,json,os,resource,shutil,time
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());whole=Path(cfg['whole_budget_root']);active=Path(cfg['output']);start=time.time()
status=json.loads((active/'status.private.json').read_text());assert status['phase']=='science_train_encodes' and status['cost']['formal_solves']==0
assert json.loads((active/'QUALIFICATION.json').read_text())['state']=='PASS'
# Exact current-experiment fixtures only. The formal verifier consumes receipts, not these filenames.
files=[whole/'run'/f'qualification_{a}.private.pt' for a in cfg['arms']]+[active/f'qualification_{a}.private.pt' for a in cfg['arms']]+[active/'qualification_moments.private.pt']
receipt=[]
for source in files:
    assert source.is_file() and not source.is_symlink()
    target=source.with_suffix(source.suffix+'.gz');temporary=Path(str(target)+'.tmp');assert not target.exists() and not temporary.exists()
    before=source.stat()
    with source.open('rb') as inp,gzip.open(temporary,'wb',compresslevel=6) as out:shutil.copyfileobj(inp,out,1024*1024)
    with gzip.open(temporary,'rb') as inp:restored=inp.read() # Complete decompression validates gzip CRC.
    assert len(restored)==before.st_size
    loaded=torch.load(io.BytesIO(restored),map_location='cpu',weights_only=False)
    if 'moments' in source.name:
        assert len(loaded)==10
        assert all(r['linear']['gram'].shape==(1537,1537) and r['kernel']['gram'].shape==(2049,2049) and r['online_independent']['state']=='PASS' for r in loaded)
    else:assert loaded['head'].shape[1]==10 and loaded['ref']['anchors'].shape==(2048,1536)
    del loaded,restored
    temporary.replace(target);after=target.stat()
    item=dict(original=str(source.relative_to(whole)),archive=str(target.relative_to(whole)),original_bytes=before.st_size,original_accounted_bytes=max(before.st_size,before.st_blocks*512),archive_bytes=after.st_size,archive_accounted_bytes=max(after.st_size,after.st_blocks*512),full_CRC_decompression=True,torch_container_readable=True,source_consumer_finished=True)
    receipt.append(item)
    # Only remove the redundant raw representation after complete archive recovery succeeds.
    source.unlink()
    (whole/'repair2/QUALIFICATION_ARCHIVE.private.json').write_text(json.dumps(dict(state='IN_PROGRESS',files=receipt),indent=2)+'\n')
result=dict(state='PASS',files=receipt,elapsed_seconds=time.time()-start,RSS_process_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_encodes=0,new_model_forwards=0,new_optimization=0,all_qualification_evidence_retained_losslessly=True,scope='Only seven completed qualification tensor files in this experiment. No scientific output, old experiments or cost receipts deleted.',old_total_accounted_bytes=sum(x['original_accounted_bytes'] for x in receipt),new_total_accounted_bytes=sum(x['archive_accounted_bytes'] for x in receipt))
(whole/'repair2/QUALIFICATION_ARCHIVE.private.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
