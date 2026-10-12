"""Publish-first, one-shot neutral evaluation-only deployment."""
import ast
import base64
import json
import os
from pathlib import Path
import subprocess


def launch(round_dir):
    r=Path(round_dir);runtime=r/'local_runtime'
    if (runtime/'LAUNCH.private.json').exists():raise ValueError('Already launched')
    c=json.loads((runtime/'SOURCE_CONFIG.private.json').read_text());sha=c['source_commit']
    assert json.loads((r/'CPU_CHECK.json').read_text())['status']=='PASS'
    env=os.environ.copy()
    for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):env[k]='http://127.0.0.1:7897'
    for k in ('NO_PROXY','no_proxy'):env[k]='localhost,127.0.0.1'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/feat/prototype-analytic'],env=env,text=True).split()[0]
    assert sha==remote==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    url=f'https://raw.githubusercontent.com/DLwbm123/Long-tailed-CL/{sha}/{r}/PROTOCOL_ZH.md'
    x=subprocess.run(['curl','--proxy',env['HTTPS_PROXY'],'--fail','--max-time','45','--silent','--show-error',
        '--output',str(runtime/'PROTOCOL_HTTP.private.md'),'--write-out','%{http_code}',url],env=env,text=True,capture_output=True)
    if x.returncode or x.stdout!='200':raise RuntimeError(x.stderr or 'Protocol unavailable')
    archive=subprocess.check_output(['git','archive',sha,'tools','third_party/air_apart_pinned',str(r)])
    wrappers={entry:'import os,sys,runpy\np=os.environ["Q128_SOURCE"]+"/tools"\nsys.path.insert(0,p)\nrunpy.run_path(p+"/'+program+'",run_name="__main__")\n'
        for entry,program in [('q157c.py','run_apart_batch_diagnostic.py'),('q157w.py','apart_batch_diagnostic.py')]}
    code='import base64,io,json,os,shutil,subprocess,tarfile,time\nfrom pathlib import Path\nc='+repr(c)+'\nw='+repr(wrappers)+'\na=base64.b64decode('+repr(base64.b64encode(archive).decode())+')\n'+'''
root=Path(c['root']);assert root.exists() and not (root/'LAUNCH.private.json').exists() and not (root/'CONFIG.private.json').exists()
assert not Path('/tmp/q157c.py').exists() and not Path('/tmp/q157w.py').exists()
assert subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip().startswith('nfs') and shutil.disk_usage(root).free>2*1024**3
probe=root/('.probe_'+str(time.time_ns()));probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
parent=Path(c['parent_root']);p=json.loads((parent/'PROGRAM_STATE.json').read_text());assert p['status']=='COMPLETE' and not p['active']
d=json.loads((parent/'DELIVERY.private.json').read_text());assert d['commit']==c['parent_delivery_commit'] and d['nas_saved']
display=subprocess.check_output(['ps','-eo','args'],text=True);assert '/tmp/q155w.py' not in display
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in gpu};assert all(free[g]>=32768 for g in (0,1))
source=root/'source';source.mkdir()
with tarfile.open(fileobj=io.BytesIO(a)) as tf:
 for m in tf.getmembers():
  q=Path(m.name);assert not q.is_absolute() and '..' not in q.parts and not m.issym() and not m.islnk()
  target=source/q
  if m.isdir():target.mkdir(parents=True,exist_ok=True)
  elif m.isfile():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(m).read())
for folder in ('logs','tmp'):(root/folder).mkdir(exist_ok=True)
for name,value in w.items():Path('/tmp/'+name).write_text(value)
c['launched_at']=time.time();c['deadline']=c['launched_at']+86400
for j in c['jobs']:j['diagnostic_lock']=str(source/c['protocol_relative'])
(root/'CONFIG.private.json').write_text(json.dumps(c,indent=2)+chr(10))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',Q128_SOURCE=str(source),Q128_CONFIG=str(root/'CONFIG.private.json'),TMPDIR=str(root/'tmp'))
with (root/'coordinator.private.log').open('x') as log:
 p=subprocess.Popen(['/tmp/q107v/bin/python','-u','/tmp/q157c.py'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
receipt=dict(pid=p.pid,source_commit=c['source_commit'],launched_at=c['launched_at'],deadline=c['deadline'],gpu_free_mib={g:free[g] for g in (0,1)},validation_only=True,optimizer_updates=0,anonymous_protocol_http_status=200)
(root/'LAUNCH.private.json').write_text(json.dumps(receipt,indent=2)+chr(10));print(json.dumps(receipt))
'''
    ast.parse(code)
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    (runtime/'LAUNCH_ATTEMPT.private.json').write_text(json.dumps(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),indent=2)+'\n')
    if x.returncode:raise RuntimeError(x.stderr)
    receipt=json.loads(x.stdout);(runtime/'LAUNCH.private.json').write_text(json.dumps(receipt,indent=2)+'\n')
    h=json.loads((runtime/'HANDOFF.private.json').read_text());h.update(current_status='RUNNING_DIAGNOSTIC_NOT_RESULTS',source_commit=sha,
        GPU_launched=True,coordinator_pid=receipt['pid'],launched_at=receipt['launched_at'],deadline=receipt['deadline'])
    (runtime/'HANDOFF.private.json').write_text(json.dumps(h,indent=2)+'\n')
    code='import json\nfrom pathlib import Path\nh='+repr(h)+'\nPath(h["current_root"],"HANDOFF.private.json").write_text(json.dumps(h,indent=2)+chr(10))\n'
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    return receipt
