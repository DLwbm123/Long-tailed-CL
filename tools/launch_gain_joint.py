"""Publish-first, single deployment of fixed three-stage statistical mechanism evaluations."""
import base64
import json
import os
from pathlib import Path
import subprocess


def launch(round_dir):
    r=Path(round_dir);rt=r/'local_runtime'
    if (rt/'LAUNCH.private.json').exists():raise ValueError('Already launched')
    c=json.loads((rt/'SOURCE_CONFIG.private.json').read_text());sha=c['source_commit']
    checked=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text())
    if checked['status']!='PASS' or not checked.get('full_high_dimensional_cpu_PASS'):raise ValueError('Complete high-dimensional CPU PASS required')
    env=os.environ.copy()
    for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):env[k]='http://127.0.0.1:7897'
    for k in ('NO_PROXY','no_proxy'):env[k]='localhost,127.0.0.1'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/feat/prototype-analytic'],env=env,text=True).split()[0]
    if sha!=remote or sha!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip():raise ValueError('Public source freeze differs')
    url=f'https://raw.githubusercontent.com/DLwbm123/Long-tailed-CL/{sha}/{r}/PROTOCOL_ZH.md'
    x=subprocess.run(['curl','--proxy',env['HTTPS_PROXY'],'--fail','--max-time','45','--silent','--show-error','--output',str(rt/'PROTOCOL_HTTP.private.md'),'--write-out','%{http_code}',url],env=env,text=True,capture_output=True)
    if x.returncode or x.stdout!='200':raise RuntimeError('Exact protocol unavailable via proxy')
    archive=subprocess.check_output(['git','archive',sha,'tools','third_party/APART',str(r)])
    wrappers={path:'import os,sys,runpy\np=os.environ["Q128_SOURCE"]+"/tools"\nsys.path.insert(0,p)\nrunpy.run_path(p+"/'+program+'",run_name="__main__")\n'
        for path,program in [('/tmp/q185c.py','run_gain_joint.py'),('/tmp/q185w.py','gain_joint_worker.py')]}
    payload=json.dumps(dict(config=c,archive=base64.b64encode(archive).decode(),wrappers=wrappers))
    code='import base64,io,json,os,shutil,subprocess,tarfile,time\nfrom pathlib import Path\np=json.loads('+repr(payload)+')\n'+'''
c=p['config'];root=Path(c['root']);assert root.is_dir() and not (root/'LAUNCH.private.json').exists() and not (root/'CONFIG.private.json').exists()
assert all(not Path(q).exists() for q in p['wrappers'])
mount=subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip();assert mount.startswith('nfs')
free=shutil.disk_usage(root).free;assert free>2*1024**3
probe=root/'.launch_probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
for j in c['jobs']:
 parent=Path(j['parent_root']);receipt=json.loads((parent/'DELIVERY.private.json').read_text());assert receipt['nas_saved']
gpu={int(r.split(',')[0]):int(r.split(',')[1]) for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
assert all(gpu[g]>=32768 for g in (0,1))
source=root/'source';source.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['archive']))) as tf:
 for m in tf.getmembers():
  rel=Path(m.name);assert not rel.is_absolute() and '..' not in rel.parts and not m.issym() and not m.islnk()
  target=source/rel
  if m.isdir():target.mkdir(parents=True,exist_ok=True)
  elif m.isfile():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(m).read())
for path,body in p['wrappers'].items():Path(path).write_text(body)
for name in ('logs','tmp'):(root/name).mkdir(exist_ok=True)
c['launched_at']=time.time();c['deadline']=c['launched_at']+14400
c['protocol']=str(source/c['protocol_relative'])
(root/'CONFIG.private.json').write_text(json.dumps(c,indent=2)+chr(10))
(root/'PROTOCOL_FREEZE.json').write_text(json.dumps(dict(source_commit=c['source_commit'],anonymous_protocol_HTTP=200,optimizer_updates=0,test_accessed=False))+chr(10))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',Q128_SOURCE=str(source),Q128_CONFIG=str(root/'CONFIG.private.json'),TMPDIR=str(root/'tmp'))
with (root/'coordinator.private.log').open('x') as log:
 proc=subprocess.Popen(['/tmp/q107v/bin/python','-u','/tmp/q185c.py'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
receipt=dict(pid=proc.pid,source_commit=c['source_commit'],launched_at=c['launched_at'],deadline=c['deadline'],mount=mount,nas_free_bytes=free,GPU_free_mib={g:gpu[g] for g in (0,1)},protocol_anonymous_HTTP=200,optimizer_updates=0)
(root/'LAUNCH.private.json').write_text(json.dumps(receipt,indent=2)+chr(10));print(json.dumps(receipt))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    attempt=len(list(rt.glob('LAUNCH_attempt*.private.json')))+1
    (rt/f'LAUNCH_attempt{attempt}.private.json').write_text(json.dumps(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),indent=2)+'\n')
    if x.returncode:raise RuntimeError(x.stderr)
    receipt=json.loads(x.stdout);(rt/'LAUNCH.private.json').write_text(json.dumps(receipt,indent=2)+'\n')
    h=dict(current_round=r.name,current_root=c['root'],source_commit=sha,
        current_status='RUNNING_PREPARATION_NOT_RESULTS',coordinator_pid=receipt['pid'],launched_at=receipt['launched_at'],
        deadline=receipt['deadline'],GPU_launched=False,authorized_gpus=[0,1],optimizer_updates=0,monitor_created=False)
    (rt/'HANDOFF.private.json').write_text(json.dumps(h,indent=2)+'\n');return receipt


if __name__=='__main__':print(json.dumps(launch(json.load(__import__('sys').stdin)['round_dir']),indent=2))
