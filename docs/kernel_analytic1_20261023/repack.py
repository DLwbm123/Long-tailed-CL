"""Use a larger lossless dictionary for the six repeated-reference qualification fixtures."""
import gzip,json,lzma,os,resource,shutil,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['whole_budget_root']);manifest=root/'repair2/QUALIFICATION_ARCHIVE.private.json';old=json.loads(manifest.read_text());assert old['state']=='PASS';start=time.time();items=[]
for item in old['files']:
    item=dict(item);source=root/item['archive']
    if 'moments' in source.name:items.append(item);continue
    assert source.name.startswith('qualification_') and source.suffix=='.gz' and source.is_file()
    target=source.with_suffix('.xz');temp=Path(str(target)+'.tmp');assert not target.exists() and not temp.exists()
    with gzip.open(source,'rb') as inp,lzma.open(temp,'wb',preset=6) as out:shutil.copyfileobj(inp,out,1024*1024)
    count=0
    with lzma.open(temp,'rb') as inp:
        while True:
            block=inp.read(1024*1024)
            if not block:break
            count+=len(block)
    assert count==item['original_bytes'];temp.replace(target);st=target.stat();source.unlink()
    item.update(archive=str(target.relative_to(root)),archive_bytes=st.st_size,archive_accounted_bytes=max(st.st_size,st.st_blocks*512),codec='xz',full_CRC_decompression=True)
    items.append(item)
old.update(files=items,reference_repack_seconds=time.time()-start,reference_repack_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,reference_repack_new_encodes=0,reference_repack_new_model_forwards=0,reference_repack_new_optimization=0,new_total_accounted_bytes=sum(x['archive_accounted_bytes'] for x in items))
manifest.write_text(json.dumps(old,indent=2)+'\n');print(json.dumps(old))
