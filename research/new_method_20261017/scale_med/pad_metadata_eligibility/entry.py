import os,json,csv,io,time,resource,urllib.request,random,traceback
from pathlib import Path
from collections import defaultdict,Counter
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
def write(n,v):(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 req=urllib.request.Request('https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded',headers={'User-Agent':'Mozilla/5.0'})
 with urllib.request.urlopen(req,timeout=30) as r:
  data=r.read(2*2**20+1);status=r.status
 write('DOWNLOAD.json',dict(HTTP=status,bytes=len(data),expected_bytes=316209));assert status==200 and len(data)==316209
 (out/'metadata.private.csv').write_bytes(data);rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))));assert len(rows)==2298
 keys=['patient_id','lesion_id','img_id','diagnostic'];assert all(k in rows[0] for k in keys);assert all(all(r[k] for k in keys) for r in rows);assert len({r['img_id'] for r in rows})==len(rows)
 labs=['ACK','BCC','MEL','NEV','SCC','SEK'];assert sorted({r['diagnostic'] for r in rows})==labs
 pts=sorted({r['patient_id'] for r in rows});les=defaultdict(list)
 for r in rows:les[r['lesion_id']].append(r)
 assert len(pts)==1373 and len(les)==1641
 assert all(len({r['patient_id'] for r in rs})==1 and len({r['diagnostic'] for r in rs})==1 for rs in les.values())
 random.Random(74380).shuffle(pts);fold={p:i%2 for i,p in enumerate(pts)}
 write('SPLIT.private.json',dict(seed=74380,patient_fold=fold,rows=[{k:r[k] for k in keys}|dict(fold=fold[r['patient_id']]) for r in rows]))
 cells=[]
 for k in [0,1]:
  for c in labs:
   rr=[r for r in rows if fold[r['patient_id']]==k and r['diagnostic']==c];ll=Counter(r['lesion_id'] for r in rr);sizes=Counter(ll.values());eligible=sum(n for m,n in sizes.items() if m>=2 and n>=2)
   cells.append(dict(fold=k,diagnosis=c,images=len(rr),patients=len({r['patient_id'] for r in rr}),lesions=len(ll),multi_image_lesions=sum(m>1 for m in ll.values()),same_size_repartition_eligible_lesions=eligible,size_histogram={str(m):n for m,n in sorted(sizes.items())},partition_evaluable=eligible>=2))
 fit_ok=all(c['patients']>=8 for c in cells);all_partition=all(c['partition_evaluable'] for c in cells)
 write('aggregate.json',dict(decision='PATIENT_SPLIT_FEASIBLE' if fit_ok else 'INSUFFICIENT_PATIENT_SUPPORT_NO_RESPLIT',rows=len(rows),patients=len(pts),lesions=len(les),classes=labs,per_fold_class=cells,patient_overlap=0,lesion_overlap=0,metadata_integrity_pass=True,patient_split_fit_eligible=fit_ok,all_class_partition_evaluable=all_partition,MEL_partition_both=all(c['partition_evaluable'] for c in cells if c['diagnosis']=='MEL'),external_validation_complete=False,method_success=False,actual_fits=0,actual_solves=0,query_scored=False,images_downloaded=0,source_commit=cfg['source_commit'],license='CC BY4.0',version=1,DOI='10.17632/zr7vgbcyr2.1'))
start=time.time();code=1
try:main();code=0
except BaseException as e:
 write('FAILURE.json',dict(type=type(e).__name__,no_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=False));write('WORKER_EXIT.json',dict(exit_code=code))
raise SystemExit(code)
