"""Bounded CT1 archive worker on the existing storage server; no GPU use."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(os.environ['P20_ARCHIVE']);root.mkdir(parents=True,exist_ok=True)
    assert os.stat(root).st_dev!=os.stat('/').st_dev,'BLOCKED_ARCHIVE_MOUNT'
    probe=root/'.probe';probe.write_bytes(b'q20');assert probe.read_bytes()==b'q20';probe.unlink()
    ssh=['ssh','-S','/tmp/q20-pull2','-o','BatchMode=yes','-p','30154','root@hb01-ssh.gpuhome.cc']
    def fetch(relative):
        p=subprocess.run(ssh+['cat /tmp/p20root/output/private/'+relative],capture_output=True,timeout=30)
        return json.loads(p.stdout) if p.returncode==0 else None
    def send(relative,obj):
        raw=json.dumps(obj).encode()
        subprocess.run(ssh+['cat > /tmp/p20root/output/private/'+relative],input=raw,check=True,timeout=30)
    seen=set();deadline=time.monotonic()+14*3600
    while time.monotonic()<deadline:
        request=fetch('ARCHIVE_REQUEST.json')
        if request and request['name'] not in seen:
            name=request['name'];assert Path(name).name==name and name.endswith('.pt')
            try:
                current=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
                assert current+request['bytes']<=10*1024**3,'BLOCKED_ARCHIVE_BUDGET'
                target=root/name;part=root/(name+'.part')
                with part.open('wb') as f:
                    subprocess.run(ssh+['cat /tmp/p20root/output/private/'+name],stdout=f,check=True,timeout=150)
                assert part.stat().st_size==request['bytes'] and sha(part)==request['sha256'],'BLOCKED_TRANSFER_SHA'
                payload=torch.load(part,map_location='cpu',weights_only=False)
                assert payload['network_sha256']==request['network_sha256']
                assert sorted(payload['delta'])==payload['nonshared_keys']
                h=hashlib.sha256()
                for k,v in sorted(payload['delta'].items()):
                    assert torch.isfinite(v).all();h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
                assert h.hexdigest()==request['nonshared_tensor_sha256']
                part.replace(target)
                ack=dict(status='PASS',name=name,sha256=request['sha256'],archive=str(target),
                         nonshared_tensor_sha256=h.hexdigest(),readable_tensor_keys=len(payload['delta']),
                         strict_source_restore=request['strict_source_restore'],archive_bytes=current+request['bytes'])
                (root/(name+'.json')).write_text(json.dumps(ack,indent=2))
                send('archive_ack/'+name+'.json',ack);seen.add(name);del payload
                print('ARCHIVED',name,flush=True)
            except BaseException as e:
                send('ARCHIVE_FAILURE.json',dict(status='BLOCKED',reason=str(e),name=name));raise
        stop=fetch('ARCHIVE_STOP.json')
        if stop:
            (root/'STOP.json').write_text(json.dumps(dict(stop,archived_tasks=sum(n.startswith(('HK_','ISIC_')) for n in seen),
                                                         engineering_archive_probes=sum(n.startswith('engineering') for n in seen))))
            return
        time.sleep(3)
    raise TimeoutError('BOUNDED_ARCHIVE_WORKER_EXPIRED')


if __name__=='__main__':main()
