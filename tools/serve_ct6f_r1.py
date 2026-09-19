"""Bounded existing-channel parent transfer and new-checkpoint archival, CPU only."""
import hashlib,io,json,os,shutil,subprocess,time
import zstandard
from pathlib import Path
import torch
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def tensor_digest(items):
    h=hashlib.sha256()
    for k,v in sorted(items):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def write_json(path,x):
    path=Path(path);part=Path(str(path)+'.part');part.write_text(json.dumps(x,indent=2));part.replace(path)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(os.environ['P25_ARCHIVE']);old=Path('/tmp/p22archive');root.mkdir(parents=True,exist_ok=True)
    assert os.stat(root).st_dev!=os.stat('/').st_dev
    p=root/'.probe';p.write_text('ready');assert p.read_text()=='ready';p.unlink()
    ssh=['ssh','-S','/tmp/q25-transfer-r1','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-p',os.environ.get('P25_SSH_PORT','30128'),'root@hb01-ssh.gpuhome.cc']
    def send(x):subprocess.run(ssh+['cat > /tmp/p25root/output/private/ACK.json.part && mv /tmp/p25root/output/private/ACK.json.part /tmp/p25root/output/private/ACK.json'],input=json.dumps(x).encode(),check=True,timeout=30)
    oldnames={f'{n}_{s}_F_t{t:02d}.pt' for n in ('HK','ISIC') for s in (1993,1994,1995) for t in (3,)}
    newnames={f'{n}_{s}_F_t{t:02d}.pt' for n in ('HK','ISIC') for s in (1993,1994,1995) for t in (range(4,12) if n=='HK' else (4,))}
    done=set()
    while True:
        q=subprocess.run(ssh+['cat /tmp/p25root/output/private/REQUEST.json'],capture_output=True,timeout=30)
        if q.returncode==0:
            req=json.loads(q.stdout);key=req['id'];name=req['name'];op=req['op']
            if key not in done:
                began=time.monotonic()
                try:
                    assert name in oldnames|newnames
                    if op=='GET':
                        p=(old if name in oldnames else root)/name
                        assert sha(p)==req['sha256']
                        subprocess.run(ssh+['/tmp/n154env/bin/python /tmp/p25io.py receive'],input=zstandard.ZstdCompressor(level=3,threads=4).compress(p.read_bytes()),check=True,timeout=600)
                        ack=dict(id=key,status='PASS',sha256=req['sha256'],bytes=p.stat().st_size)
                    else:
                        assert op=='PUT' and name in newnames and not (root/name).exists()
                        p=root/(name+'.part')
                        assert sum(f.stat().st_size for f in root.iterdir() if f.is_file())+req['bytes']<10*1024**3
                        data=subprocess.check_output(ssh+['/tmp/n154env/bin/python /tmp/p25io.py send'],timeout=600);
                        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst,1024**2)
                        del data
                        assert p.stat().st_size==req['bytes'] and sha(p)==req['sha256']
                        v=torch.load(p,map_location='cpu',weights_only=False)
                        assert v['network_sha256']==req['network_sha256'] and tensor_digest(v['delta'].items())==req['delta_sha256']
                        assert sorted(v['delta'])==v['nonshared_keys']
                        assert all(torch.isfinite(t).all() for t in v['delta'].values())
                        p.replace(root/name);del v
                        ack=dict(id=key,status='PASS',sha256=req['sha256'],bytes=req['bytes'],readable=True,
                                 delta_sha256=req['delta_sha256'],archive_bytes=sum(f.stat().st_size for f in root.iterdir() if f.is_file()))
                        write_json(root/(name+'.json'),ack)
                    ack['transfer_seconds']=time.monotonic()-began;send(ack);done.add(key);print(op,name,ack['transfer_seconds'],flush=True)
                except BaseException as e:send(dict(id=key,status='BLOCKED',reason=str(e)));raise
        stop=subprocess.run(ssh+['test -f /tmp/p25root/output/private/STOP_TRANSFER.json'],capture_output=True,timeout=30)
        if stop.returncode==0:write_json(root/'STOP.json',dict(status='STOP',requests=len(done)));return
        time.sleep(2)
if __name__=='__main__':main()
