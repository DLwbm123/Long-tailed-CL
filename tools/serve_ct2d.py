"""Read-only CT1 archive audit and bounded one-at-a-time CT2-D transfer."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import torch

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def group(k):
    if 'prompt_key' in k:return 'few_key' if '.pool_few.' in k else 'main_key'
    if '.pool_few.' in k:return 'few_adapter'
    if '.pool.' in k:return 'main_adapter'
    if '.assigner.' in k:return 'assigner'
    if '.head_few.' in k:return 'few_head'
    if '.head.' in k:return 'main_head'
    raise AssertionError(k)

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    root=Path(os.environ['P21_ARCHIVE']);archive=Path('/tmp/p20archive');started=time.monotonic()
    ssh=['ssh','-S','/tmp/q20-pull2','-o','BatchMode=yes','-p','30154','root@hb01-ssh.gpuhome.cc']
    def send(name,x):subprocess.run(ssh+['cat > /tmp/p21root/'+name],input=json.dumps(x).encode(),check=True,timeout=30)
    rows=[];prev={};init={};shared=set();code=None;names=None
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            for stream in ('U','R'):
                last=None
                for task in range(1,12 if name=='HK' else 5):
                    path=archive/f'{name}_{seed}_{stream}_t{task:02d}.pt'
                    p=torch.load(path,map_location='cpu',weights_only=False)
                    ack=json.loads(path.with_suffix('.pt.json').read_text());assert sha(path)==ack['sha256']
                    assert (p['dataset'],p['seed'],p['stream'],p['task'],p['epoch'])==(name,seed,stream,task-1,10)
                    assert sorted(p['delta'])==p['nonshared_keys']
                    if names is None:names=p['nonshared_keys'];code=p['code_sha256']
                    assert p['nonshared_keys']==names and p['code_sha256']==code
                    shared.add(p['shared_sha256']);changed={};hashes={}
                    for g in sorted({group(k) for k in names}):
                        keys=[k for k in names if group(k)==g];h=hashlib.sha256()
                        for k in keys:
                            v=p['delta'][k];assert torch.isfinite(v).all();h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
                        hashes[g]=h.hexdigest()
                        changed[g]=None if last is None else sum(not torch.equal(p['delta'][k],last[k]) for k in keys)
                    if task==1:
                        if stream=='U':init[name,seed]=p['network_sha256']
                        else:assert init[name,seed]==p['network_sha256']
                    rows.append(dict(dataset=name,seed=seed,stream=stream,task=task,file=path.name,sha256=ack['sha256'],
                          network_sha256=p['network_sha256'],shared_sha256=p['shared_sha256'],steps=p['steps'],
                          known=p['known'],seen=p['seen'],nonshared_group_sha256=hashes,changed_tensors_by_group=changed))
                    last=p['delta'];del p
                del last
    assert len(rows)==90 and len(shared)==1
    audit=dict(status='PASS',states=rows,nonshared_names=names,shared_sha256=list(shared),
               original_code_sha256=code,paired_task1=True,audit_kind='Serialized state and source SHA audit; not 90 encoder reinstantiations',
               cpu_seconds=time.monotonic()-started)
    (root/'ARCHIVE_SOURCE_AUDIT.json').write_text(json.dumps(audit,indent=2));send('output/public/ARCHIVE_SOURCE_AUDIT.json',audit)
    done=set();deadline=time.monotonic()+4*3600
    while time.monotonic()<deadline:
        q=subprocess.run(ssh+['cat /tmp/p21root/output/private/PARENT_REQUEST.json'],capture_output=True,timeout=30)
        if q.returncode==0:
            req=json.loads(q.stdout);key=req['request_id'];name=req['name']
            if key not in done:
                assert name in {r['file'] for r in rows};p=archive/name;digest=sha(p)
                assert digest==req['sha256']
                with p.open('rb') as f:subprocess.run(ssh+['cat > /tmp/p21root/output/private/parent.pt.part'],stdin=f,check=True,timeout=150)
                send('output/private/PARENT_ACK.json',dict(request_id=key,name=name,sha256=digest,bytes=p.stat().st_size,source_SHA_verified=True));done.add(key)
        stop=subprocess.run(ssh+['test -f /tmp/p21root/output/private/STOP_TRANSFER.json'],timeout=30)
        if stop.returncode==0:break
        time.sleep(2)
    (root/'TRANSFER_STOP.json').write_text(json.dumps(dict(requests=len(done),cpu_seconds=time.monotonic()-started)))

if __name__=='__main__':main()
