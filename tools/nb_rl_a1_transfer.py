"""Immutable asynchronous backup queue; neutral entry points take private config."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    cfg=json.loads(Path(os.environ['N78_IO_CONFIG']).read_text());role=os.environ['N78_IO_ROLE'];root=Path(cfg['root'])
    requests=root/'private/transfer_requests';acks=root/'private/transfer_acks'
    if role=='request':
        pending=[p for p in sorted(requests.glob('*.json')) if not (acks/p.name).exists()]
        print(pending[0].read_text() if pending else '{}');return
    if role=='ack':
        value=json.load(sys.stdin);acks.mkdir(exist_ok=True);p=acks/(value['id']+'.json')
        tmp=p.with_suffix('.part');tmp.write_text(json.dumps(value));tmp.replace(p);return
    if role=='send':
        key=os.environ['N78_IO_ID'];assert key.isdigit()
        req=json.loads((requests/(key+'.json')).read_text())
        with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
            b=json.dumps(req).encode();info=tarfile.TarInfo('MANIFEST.json');info.size=len(b);tar.addfile(info,io.BytesIO(b))
            for item in req['items']:tar.add(root/item['file'],arcname=item['file'],recursive=False)
        return
    if role=='receive':
        with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
            members=iter(tar);first=next(members);assert first.name=='MANIFEST.json'
            req=json.load(tar.extractfile(first));expected={x['file']:x for x in req['items']};seen=set()
            for m in members:
                assert m.isfile() and m.name in expected and m.name not in seen
                p=root/m.name;assert p.resolve().is_relative_to(root.resolve())
                p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.part')
                with tmp.open('wb') as f:shutil.copyfileobj(tar.extractfile(m),f,1024*1024)
                assert digest(tmp)==expected[m.name]['sha256'];tmp.replace(p)
                assert digest(p)==expected[m.name]['sha256'];seen.add(m.name)
            assert seen==set(expected)
        ack=dict(id=req['id'],label=req['label'],status='PASS',items=req['items'],independent_SHA_reread=True,unix=time.time())
        p=root/'private/transfer_acks';p.mkdir(parents=True,exist_ok=True);(p/(req['id']+'.json')).write_text(json.dumps(ack))
        print(json.dumps(ack));return
    assert role=='relay'
    def command(host,op,key=None):
        env='env N78_IO_CONFIG='+cfg.get('remote_config','/tmp/p78io.json')+' N78_IO_ROLE='+op
        if key is not None:
            assert key.isdigit();env+=' N78_IO_ID='+key
        return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,env+' python3 '+cfg.get('remote_entry','/tmp/p78io.py')]
    failures=0
    while time.time()<cfg['wall_T0_unix']+57600:
        try:
            req=json.loads(subprocess.check_output(command('my-gpu','request'),timeout=45))
            if not req:time.sleep(10);continue
            remaining=cfg['wall_T0_unix']+57600-time.time();assert remaining>0
            sender=subprocess.Popen(command('my-gpu','send',req['id']),stdout=subprocess.PIPE)
            try:
                received=subprocess.run(command('jiangsuiyang','receive'),stdin=sender.stdout,capture_output=True,timeout=min(900,remaining))
                sender.stdout.close();status=sender.wait(timeout=30)
            finally:
                if sender.poll() is None:sender.terminate();sender.wait(timeout=10)
            assert status==received.returncode==0,received.stderr.decode()
            ack=json.loads(received.stdout);assert ack['id']==req['id'] and ack['items']==req['items']
            subprocess.run(command('my-gpu','ack'),input=json.dumps(ack).encode(),check=True,timeout=45)
            print('TRANSFER',req['id'],req['label'],len(req['items']),flush=True);failures=0
            if req['label']=='completion_receipt':
                if cfg.get('on_complete'):
                    try:
                        finished=subprocess.run(cfg['on_complete'],timeout=300)
                    except subprocess.TimeoutExpired as error:
                        raise RuntimeError('Completion publication timed out; no automatic rerun') from error
                    if finished.returncode:raise SystemExit(finished.returncode)
                return
        except (subprocess.SubprocessError,ConnectionError,OSError) as error:
            failures+=1;print(type(error).__name__,str(error),flush=True)
            if failures>=3:raise
            time.sleep(10)
    raise TimeoutError('16h transfer deadline')


if __name__=='__main__':main()
