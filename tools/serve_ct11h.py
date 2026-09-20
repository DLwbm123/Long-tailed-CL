"""GET-only bounded transfer for twelve fixed read-only archived states."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import zstandard

def main():
    root=Path('/tmp/p30archive');assert root.stat().st_dev!=Path('/').stat().st_dev
    fixed=json.loads((root/'CHECKPOINTS.json').read_text());allowed={(e['origin'],e['name']):e['sha256'] for e in fixed};assert len(allowed)==12
    ssh=['ssh','-S','/tmp/q25-transfer-r1','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-p','30128','root@hb01-ssh.gpuhome.cc']
    def send(v):subprocess.run(ssh+['cat > /tmp/p30root/output/private/ACK.json.part && mv /tmp/p30root/output/private/ACK.json.part /tmp/p30root/output/private/ACK.json'],input=json.dumps(v).encode(),check=True,timeout=30)
    done=set()
    while True:
        q=subprocess.run(ssh+['cat /tmp/p30root/output/private/REQUEST.json'],capture_output=True,timeout=30)
        if q.returncode==0:
            e=json.loads(q.stdout);key=e['id']
            if key not in done:
                began=time.monotonic()
                try:
                    assert e['op']=='GET' and allowed[(e['origin'],e['name'])]==e['sha256']
                    p=Path('/tmp/'+e['origin']+'archive')/e['name'];assert p.resolve().stat().st_dev!=Path('/').stat().st_dev
                    b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==e['sha256']
                    subprocess.run(ssh+['/tmp/n154env/bin/python /tmp/p30io.py receive'],input=zstandard.ZstdCompressor(level=3,threads=4).compress(b),check=True,timeout=600)
                    send(dict(id=key,status='PASS',sha256=e['sha256'],bytes=len(b)));del b;done.add(key)
                    print('GET',e['origin'],e['name'],time.monotonic()-began,flush=True)
                except BaseException as err:send(dict(id=key,status='BLOCKED',reason=str(err)));raise
        if subprocess.run(ssh+['test -f /tmp/p30root/output/private/STOP_TRANSFER.json'],capture_output=True,timeout=30).returncode==0:
            (root/'STOP.json').write_text(json.dumps(dict(status='STOP',requests=len(done))));return
        time.sleep(2)
if __name__=='__main__':main()
