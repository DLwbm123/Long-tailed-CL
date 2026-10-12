"""One frozen launch, proxied publication verification and neutral GPU01 workers."""
import ast
import base64
import json
import os
from pathlib import Path
import subprocess
import sys


def launch(round_dir):
    r=Path(round_dir);runtime=r/'local_runtime'
    if (runtime/'LAUNCH.private.json').exists():raise ValueError('Already launched; never repeat')
    config=json.loads((runtime/'SOURCE_CONFIG.private.json').read_text());sha=config['source_commit']
    if json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text())['status']!='PASS':raise ValueError('CPU production check missing')
    env=os.environ.copy()
    for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):env[k]='http://127.0.0.1:7897'
    for k in ('NO_PROXY','no_proxy'):env[k]='localhost,127.0.0.1'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/feat/prototype-analytic'],env=env,text=True).split()[0]
    if sha!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() or sha!=remote:
        raise ValueError('Exact frozen source must be published before launch')
    url=f'https://raw.githubusercontent.com/DLwbm123/Long-tailed-CL/{sha}/{r}/PROTOCOL_ZH.md'
    x=subprocess.run(['curl','--proxy',env['HTTPS_PROXY'],'--fail','--max-time','45','--silent','--show-error','--output',str(runtime/'PROTOCOL_HTTP.private.md'),'--write-out','%{http_code}',url],env=env,text=True,capture_output=True)
    if x.returncode or x.stdout!='200':raise RuntimeError('Proxied exact protocol inaccessible: '+x.stderr)
    archive=subprocess.check_output(['git','archive',sha,'tools','third_party/air_apart_pinned',str(r)])
    wrappers={entry:'import os,sys,runpy\np=os.environ["Q128_SOURCE"]+"/tools"\nsys.path.insert(0,p)\nrunpy.run_path(p+"/'+program+'",run_name="__main__")\n'
              for entry,program in [('q155c.py','run_external_medical_cycle.py'),('q155w.py','run_external_medical.py')]}
    script='import base64,io,json,os,shutil,subprocess,tarfile,time\nfrom pathlib import Path\nc='+repr(config)+'\nwrappers='+repr(wrappers)+'\na=base64.b64decode('+repr(base64.b64encode(archive).decode())+')\n'+'''
root=Path(c['root']);assert root.exists() and not (root/'LAUNCH.private.json').exists() and not (root/'CONFIG.private.json').exists()
assert not Path('/tmp/q155c.py').exists() and not Path('/tmp/q155w.py').exists()
mount=subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip();assert mount.startswith('nfs')
free=shutil.disk_usage(root).free;assert free>40*1024**3
probe=root/('.probe_'+str(time.time_ns()));probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
for name,local in c['protected_delivery_receipts'].items():
 d=json.loads((root.parent/name/'DELIVERY.private.json').read_text());p=json.loads((root.parent/name/'PROGRAM_STATE.json').read_text())
 assert all(d[k]==local[k] for k in ('status','commit','nas_saved','anonymous_http_status')) and d['status']=='DELIVERED' and d['nas_saved'] and d['anonymous_http_status']==200
 assert p['status']=='COMPLETE' and not p['active']
display=subprocess.check_output(['ps','-eo','args'],text=True);assert not any('/tmp/'+x in display for x in ('q151w.py','q153w.py'))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();available={int(x.split(',')[0]):int(x.split(',')[1]) for x in gpu};assert all(available[g]>=32768 for g in (0,1))
source=root/'source';source.mkdir()
with tarfile.open(fileobj=io.BytesIO(a)) as tf:
 for m in tf.getmembers():
  p=Path(m.name);assert not p.is_absolute() and '..' not in p.parts and not m.issym() and not m.islnk()
  target=source/p
  if m.isdir():target.mkdir(parents=True,exist_ok=True)
  elif m.isfile():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(m).read())
for folder in ('logs','tmp'):(root/folder).mkdir(exist_ok=True)
for entry,content in wrappers.items():Path('/tmp/'+entry).write_text(content)
c['launched_at']=time.time();c['deadline']=c['launched_at']+86400
for j in c['jobs']+c['preflights']:j['original_deadline']=c['deadline']
(root/'CONFIG.private.json').write_text(json.dumps(c,indent=2)+chr(10))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',Q128_SOURCE=str(source),Q128_CONFIG=str(root/'CONFIG.private.json'),TMPDIR=str(root/'tmp'))
with (root/'coordinator.private.log').open('x') as log:
 p=subprocess.Popen(['/tmp/q107v/bin/python','-u','/tmp/q155c.py'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
receipt=dict(pid=p.pid,launched_at=c['launched_at'],deadline=c['deadline'],source_commit=c['source_commit'],gpu_free_mib={g:available[g] for g in (0,1)},mount=mount,nas_free_bytes=free,real_preflights_dispatched_not_passed=True,anonymous_protocol_http_status=200)
(root/'LAUNCH.private.json').write_text(json.dumps(receipt,indent=2)+chr(10));print(json.dumps(receipt))
'''
    ast.parse(script)
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=script,text=True,capture_output=True)
    (runtime/'LAUNCH_ATTEMPT.private.json').write_text(json.dumps(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),indent=2)+'\n')
    if x.returncode:raise RuntimeError(x.stderr)
    receipt=json.loads(x.stdout);(runtime/'LAUNCH.private.json').write_text(json.dumps(receipt,indent=2)+'\n')
    h=json.loads((runtime/'HANDOFF.private.json').read_text());h.update(current_status='RUNNING_NOT_RESULTS',source_commit=sha,
        GPU_launched=True,coordinator_pid=receipt['pid'],launched_at=receipt['launched_at'],deadline=receipt['deadline'],real_GPU_preflights_passed=0,
        current_phase='PREFLIGHT',authorized_gpus=[0,1],collector_ready=True,delivery_helper_ready=True)
    (runtime/'HANDOFF.private.json').write_text(json.dumps(h,indent=2)+'\n')
    code='import json\nfrom pathlib import Path\nh='+repr(h)+'\nPath(h["current_root"],"HANDOFF.private.json").write_text(json.dumps(h,indent=2)+chr(10))\n'
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    return receipt


if __name__=='__main__':print(json.dumps(launch(json.load(sys.stdin)['round_dir']),indent=2))
