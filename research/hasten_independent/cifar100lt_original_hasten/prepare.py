"""Local proxy-only downloads, bounded transfer, then detached remote training."""
import os,sys,json,time,signal,subprocess,traceback,tarfile
from pathlib import Path
r=Path('/tmp/q1100');assets=Path('/tmp/q1100-assets');clock=json.loads((r/'FIRST_ACTUAL.private.json').read_text());deadline=clock['deadline_unix']
def write(name,x):
    p=r/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def run(args,**kw):
    return subprocess.run(args,check=True,timeout=max(1,min(600,deadline-time.time()-120)),**kw)
def size():
    return sum(p.stat().st_size for root in [r,assets] for p in root.rglob('*') if p.is_file() and not p.is_symlink())
env=os.environ.copy();env.update(HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',https_proxy='http://127.0.0.1:7897',http_proxy='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
try:
    write('PREP_STATUS.json',dict(time=time.time(),phase='DOWNLOAD',optimizer_steps=0,deadline_unix=deadline))
    sources=[('cifar-100-python.tar.gz','https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'),('vit_b_16-laion400m_e32-55e67d44.pt','https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt')]
    for name,url in sources:
        marker=assets/(name+'.complete')
        if marker.exists():continue
        attempts=0
        while True:
            assert time.time()<deadline-180 and size()<3_400_000_000
            attempts+=1
            config='\n'.join(['fail','location','continue-at = "-"','retry = 2','connect-timeout = 30','max-time = 300','max-filesize = 800000000','url = '+json.dumps(url),'output = '+json.dumps(str(assets/name))])
            p=subprocess.run(['curl','--config','-'],input=config,text=True,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            write('PREP_STATUS.json',dict(time=time.time(),phase='DOWNLOAD',asset=name,attempt=attempts,curl_exit=p.returncode,bytes=(assets/name).stat().st_size,optimizer_steps=0))
            if p.returncode==0:marker.write_text(str(time.time()));break
            if p.returncode==33:raise RuntimeError('Server does not support partial download; retained partial asset, no silent restart')
            time.sleep(10)
    with tarfile.open(assets/'cifar-100-python.tar.gz') as t:assert 'cifar-100-python/train' in t.getnames()
    write('PREP_STATUS.json',dict(time=time.time(),phase='TRANSFER',optimizer_steps=0))
    run(['ssh','my-gpu','mkdir -p /tmp/q1101/assets'])
    target='my-gpu:/tmp/q1101/'
    for name in ['official','worker.py','control.py','RUN_CONFIG.json','SPLIT_SPEC.json','FIRST_ACTUAL.private.json','PROTOCOL.md','UPSTREAM.json','TREE_PROVENANCE.json']:
        run(['scp','-qr',str(r/name),target])
    for name,_ in sources:run(['scp','-q',str(assets/name),target+'assets/'])
    script="""
from pathlib import Path
import os,json,time,tarfile,subprocess
a=Path('/tmp/q1101');assert a.exists(),'Provision the neutral run alias before transfer';r=a.resolve()
assert a.resolve()==r.resolve()
assert time.time()<json.loads((r/'FIRST_ACTUAL.private.json').read_text())['deadline_unix']-300
free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=14336,('GPU1_FREE_MEMORY_PRELAUNCH',free)
dest=r/'assets/data/cifar100';dest.mkdir(parents=True,exist_ok=True)
with tarfile.open(r/'assets/cifar-100-python.tar.gz') as t:
 for m in t.getmembers():
  assert not m.name.startswith('/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk()
 t.extractall(dest)
for name,expected in [('train',155249918),('test',31049707),('meta',1473)]:
 p=dest/'cifar-100-python'/name
 assert p.exists() and p.stat().st_size==expected,(name,p.stat().st_size)
log=(r/'controller.private.log').open('a')
p=subprocess.Popen(['/tmp/q107v/bin/python','/tmp/q1101/control.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r)
print(json.dumps(dict(controller_pid=p.pid,phase='REMOTE_CONTROLLER_LAUNCHED',time=time.time())))
"""
    result=run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=script,text=True,capture_output=True)
    write('PREP_COMPLETE.json',dict(time=time.time(),remote=json.loads(result.stdout),assets_bytes=size()))
except BaseException as error:
    write('PREP_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),stdout=getattr(error,'stdout',None),stderr=getattr(error,'stderr',None)));raise
