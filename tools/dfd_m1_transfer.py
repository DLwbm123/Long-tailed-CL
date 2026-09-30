"""Bounded private transfer channel; stream bytes without local large-file staging."""
import hashlib,io,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    cfg=json.loads(Path(os.environ['N76_IO_CONFIG']).read_text());role=os.environ['N76_IO_ROLE'];root=Path(cfg['root'])
    if role=='state':
        q=root/'private/TRANSFER_ACK.json';print(q.read_text() if q.exists() else '{}');return
    if role=='request':
        p=root/'private/TRANSFER_REQUEST.json';print(p.read_text() if p.exists() else '{}');return
    if role=='ack':
        v=json.load(sys.stdin);p=root/'private/TRANSFER_ACK.json';tmp=p.with_suffix('.part');tmp.write_text(json.dumps(v));tmp.replace(p);return
    if role in ('send','public'):
        req=json.loads((root/'private/TRANSFER_REQUEST.json').read_text())
        with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
            if role=='send':
                b=json.dumps(req).encode();info=tarfile.TarInfo('TRANSFER_MANIFEST.json');info.size=len(b);t.addfile(info,io.BytesIO(b))
            for e in req['items']:
                if role=='send' or e['file'].startswith('public/'):t.add(root/e['file'],arcname=e['file'],recursive=False)
        return
    if role=='receive':
        with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as t:
            members=iter(t);first=next(members);assert first.name=='TRANSFER_MANIFEST.json';req=json.load(t.extractfile(first));expected={e['file']:e for e in req['items']};seen=set()
            for item in members:
                assert item.isfile() and item.name in expected and item.name not in seen
                p=root/item.name;assert p.resolve().is_relative_to(root.resolve());p.parent.mkdir(parents=True,exist_ok=True)
                tmp=p.with_name(p.name+'.part')
                with tmp.open('wb') as f:shutil.copyfileobj(t.extractfile(item),f,1024*1024)
                assert sha(tmp)==expected[item.name]['sha256'];tmp.replace(p)
                assert sha(p)==expected[item.name]['sha256'];seen.add(item.name)
            assert seen==set(expected)
        print(json.dumps(dict(id=req['id'],status='PASS',label=req['label'],items=req['items'],independent_SHA_reread=True,
            backup_total_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),unix=time.time())));return
    assert role=='relay'
    root.mkdir(parents=True,exist_ok=True);seen=None;deadline=time.monotonic()+cfg.get('deadline_seconds',13*3600)
    def cmd(host,op):return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'env N76_IO_CONFIG=/tmp/n76io.json N76_IO_ROLE='+op+' python3 /tmp/n76io.py']
    prior=json.loads(subprocess.check_output(cmd('my-gpu','state'),timeout=45))
    if prior.get('status')=='PASS':seen=prior['id']
    while time.monotonic()<deadline:
        try:
            req=json.loads(subprocess.check_output(cmd('my-gpu','request'),timeout=45))
            if req and req['id']!=seen:
                sender=subprocess.Popen(cmd('my-gpu','send'),stdout=subprocess.PIPE)
                receiver=subprocess.run(cmd('jiangsuiyang','receive'),stdin=sender.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=900)
                sender.stdout.close();code=sender.wait(timeout=30)
                assert code==receiver.returncode==0,receiver.stderr.decode()
                ack=json.loads(receiver.stdout);assert ack['items']==req['items'] and ack['id']==req['id']
                subprocess.run(cmd('my-gpu','ack'),input=json.dumps(ack).encode(),check=True,timeout=45)
                # Public text only is mirrored locally for later release. No scientific assets.
                public=[x['file'] for x in req['items'] if x['file'].startswith('public/')]
                if public:
                    stream=subprocess.Popen(cmd('my-gpu','public'),stdout=subprocess.PIPE)
                    with tarfile.open(fileobj=stream.stdout,mode='r|') as tar:
                        for m in tar:
                            assert m.isfile() and m.name in public
                            (root/Path(m.name).name).write_bytes(tar.extractfile(m).read())
                    assert stream.wait()==0
                seen=req['id'];print('TRANSFER',req['label'],len(req['items']),flush=True)
                if req['label']=='completion_receipt':return
            time.sleep(3)
        except (subprocess.SubprocessError,ConnectionError) as error:
            print(type(error).__name__,str(error),flush=True);time.sleep(10)
    raise TimeoutError('transfer channel deadline exceeded')
if __name__=='__main__':main()
