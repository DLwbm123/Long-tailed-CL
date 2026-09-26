"""Private streaming relay endpoint; base directory comes from a local config."""
import hashlib,io,json,os,sys,tarfile,time
from pathlib import Path
PUBLIC=('FINAL_REPORT_ZH.md','FINAL_REPORT.json','METHOD_MATRIX.csv','stage_metrics.csv','class_metrics.csv','summary_by_parent_and_projection.csv','lambda_cv_scores.csv','lambda_selection.json','bootstrap_intervals.csv','error_decomposition.csv','candidate_coverage.csv','projection_robustness.csv','paired_differences.csv','forgetting.csv','SOLVER_PARITY.json','ENGINEERING_REPORT.json','PROTOCOL_LOCK.json','SOURCE_LOCK.json','RANDOM_MAP_LOCK.json','ACCESS_LEDGER.json','RESOURCE_REPORT.json','BACKUP_REPORT.json','completion_receipts.json','failure_receipts.json','NEXT_DECISION.json','PROCESS_RESIDENCE.json','RESOURCE_ADMISSION.json')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def safe(root,name):
    p=root/name
    if Path(name).is_absolute() or '..' in Path(name).parts or not p.resolve().is_relative_to(root.resolve()):raise ValueError('UNSAFE_MEMBER')
    return p

def main():
    os.umask(0o077);cfg=json.loads(Path('/tmp/n63io.json').read_text());out=Path(cfg['root']);op=sys.argv[1]
    if op=='status':
        req=json.loads((out/'BACKUP_REQUEST.json').read_text()) if (out/'BACKUP_REQUEST.json').exists() else None
        print(json.dumps({'request':req,'acked':bool(req and (out/'backup_acks'/f"{req['token']}.json").exists()),'ready':(out/'DELIVERY_READY.json').exists(),'controller_failed':False}));return
    if op=='ack':
        r=json.load(sys.stdin);p=out/'backup_acks'/f"{r['token']}.json";p.parent.mkdir(exist_ok=True);tmp=p.with_suffix('.part');tmp.write_text(json.dumps(r));tmp.replace(p);return
    if op=='archive':
        req=json.loads((out/'BACKUP_REQUEST.json').read_text());files=[]
        for n in req['paths']:
            p=safe(out,n)
            files.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file() and not x.is_symlink()])
        req['files']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files}
        with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as a:
            data=json.dumps(req).encode();m=tarfile.TarInfo('_TRANSFER.json');m.size=len(data);a.addfile(m,io.BytesIO(data))
            for p in files:a.add(p,arcname=str(p.relative_to(out)),recursive=False)
        return
    if op=='receive':
        out.mkdir(parents=True,exist_ok=True);seen=set();req=None;total=0
        with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as a:
            for m in a:
                if not m.isfile():raise ValueError('ONLY_REGULAR_FILES')
                if m.name=='_TRANSFER.json':req=json.load(a.extractfile(m));continue
                if req is None or m.name not in req['files'] or m.name in seen:raise ValueError('TRANSFER_MANIFEST')
                p=safe(out,m.name);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.part');h=hashlib.sha256();n=0
                with tmp.open('wb') as f:
                    src=a.extractfile(m)
                    for b in iter(lambda:src.read(8*1024*1024),b''):f.write(b);h.update(b);n+=len(b)
                expected=req['files'][m.name]
                if n!=expected['bytes'] or h.hexdigest()!=expected['sha256']:raise ValueError('DESTINATION_DIGEST')
                tmp.replace(p);seen.add(m.name);total+=n
        if seen!=set(req['files']):raise ValueError('INCOMPLETE_TRANSFER')
        receipt={'token':req['token'],'status':'VERIFIED','files':len(seen),'bytes':total,'time':time.time(),'verification':'SHA256 of written stream, atomic destination files'}
        (out/'backup_acks').mkdir(exist_ok=True);(out/'backup_acks'/f"{req['token']}.json").write_text(json.dumps(receipt))
        for name in seen:
            p=Path(name)
            if p.parts[0]=='states' and p.suffix=='.npz' and p.stem.startswith('task_'):
                t=int(p.stem.split('_')[1]);old=safe(out,str(p.with_name(f'task_{t-2:02d}.npz')))
                if t>2 and old.exists():old.unlink()
        print(json.dumps(receipt));return
    if op=='public':
        with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as a:
            for name in PUBLIC:
                p=out/name
                if p.exists():a.add(p,arcname=name,recursive=False)
        return
    raise ValueError('UNKNOWN_OPERATION')
if __name__=='__main__':main()
