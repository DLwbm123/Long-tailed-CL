"""One frozen source deployment; no retries, no historical controls launched."""
import base64
import json
import os
from pathlib import Path
import re
import subprocess


def launch(round_dir):
    r=Path(round_dir);rt=r/'local_runtime'
    if (rt/'LAUNCH.private.json').exists():raise ValueError('Already launched')
    c=json.loads((rt/'SOURCE_CONFIG.private.json').read_text());sha=c['source_commit']
    if json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text())['status']!='PASS' or json.loads((r/'CONTROL_MATCH.json').read_text())['status']!='PASS':
        raise ValueError('CPU/control matching required')
    env=os.environ.copy()
    for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):env[k]='http://127.0.0.1:7897'
    for k in ('NO_PROXY','no_proxy'):env[k]='localhost,127.0.0.1'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/feat/prototype-analytic'],env=env,text=True).split()[0]
    if remote!=sha:raise ValueError('Exact public freeze required before launch')
    url=f'https://raw.githubusercontent.com/DLwbm123/Long-tailed-CL/{sha}/{r}/PROTOCOL_ZH.md'
    x=subprocess.run(['curl','--proxy',env['HTTPS_PROXY'],'--fail','--silent','--show-error','--max-time','45','--output',str(rt/'PROTOCOL_HTTP.private.md'),'--write-out','%{http_code}',url],env=env,text=True,capture_output=True)
    if x.returncode or x.stdout!='200':raise RuntimeError('Exact anonymous protocol not available through proxy')
    archive=subprocess.check_output(['git','archive',sha,'tools','third_party/APART','docs/dynamic_anchor_design_20261014',str(r)])
    coordinator=c.get('coordinator_entry','/tmp/q185c.py');worker=c['worker_entry']
    if not re.fullmatch(r'/tmp/q[0-9]+c\.py',coordinator) or not re.fullmatch(r'/tmp/q[0-9]+w\.py',worker):
        raise ValueError('Neutral coordinator and worker entries required')
    wrappers={path:'import os,sys,runpy\np=os.environ["Q128_SOURCE"]+"/tools"\nsys.path.insert(0,p)\nrunpy.run_path(p+"/'+program+'",run_name="__main__")\n'
        for path,program in [(coordinator,'run_dynamic_anchor_cycle.py'),(worker,'run_dynamic_anchor_worker.py')]}
    c['coordinator_entry']=coordinator
    payload=json.dumps(dict(config=c,archive=base64.b64encode(archive).decode(),wrappers=wrappers))
    code='import base64,io,json,os,shutil,subprocess,tarfile,time\nfrom pathlib import Path\np=json.loads('+repr(payload)+')\n'+'''
c=p['config'];root=Path(c['root']);assert root.is_dir() and not (root/'CONFIG.private.json').exists() and not (root/'LAUNCH.private.json').exists()
for path in p['wrappers']:assert not Path(path).exists()
mount=subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip();assert mount.startswith('nfs')
free=shutil.disk_usage(root).free;assert free>40*1024**3
probe=root/'.launch_probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
for name,commit in [('module_additions1_20261009','ecd783ffd7b32eea4e2e2a1311b9f033231f1677'),('module_local2_20261010','4b6e0c87887b101ae2970738919760f00e19726e')]:
 delivery=json.loads((root.parent/name/'DELIVERY.private.json').read_text());assert delivery['commit']==commit and delivery['status']=='DELIVERED' and delivery['nas_saved']
available={int(r.split(',')[0]):int(r.split(',')[1]) for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
source=root/'source';source.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['archive']))) as tf:
 for m in tf.getmembers():
  rel=Path(m.name);assert not rel.is_absolute() and '..' not in rel.parts and not m.issym() and not m.islnk()
  f=source/rel
  if m.isdir():f.mkdir(parents=True,exist_ok=True)
  elif m.isfile():f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(tf.extractfile(m).read())
for path,body in p['wrappers'].items():Path(path).write_text(body)
for name in ('logs','tmp'): (root/name).mkdir(exist_ok=True)
launched=time.time();c.update(launched_at=launched,deadline=launched+86400)
for j in c['jobs']+c['preflights']:j.update(source_commit=c['source_commit'],original_deadline=c['deadline'])
(root/'CONFIG.private.json').write_text(json.dumps(c,indent=2)+chr(10))
(root/'PROTOCOL_FREEZE.json').write_text(json.dumps(dict(source_commit=c['source_commit'],protocol_anonymous_HTTP=200,GPU_scope=[0,1],new_trajectories=8,new_updates=2576))+chr(10))
env=os.environ.copy();env.update(Q128_SOURCE=str(source),Q128_CONFIG=str(root/'CONFIG.private.json'),CUDA_VISIBLE_DEVICES='',TMPDIR=str(root/'tmp'))
with (root/'coordinator.private.log').open('x') as log:
 proc=subprocess.Popen(['/tmp/q107v/bin/python','-u',c['coordinator_entry']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,source_commit=c['source_commit'],launched_at=launched,deadline=c['deadline'],mount=mount,nas_free_bytes=free,GPU_free_mib={g:available[g] for g in (0,1)},protocol_anonymous_HTTP=200,new_trajectories=8,new_updates=2576)
(root/'LAUNCH.private.json').write_text(json.dumps(receipt,indent=2)+chr(10));print(json.dumps(receipt))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    (rt/('LAUNCH_attempt'+str(len(list(rt.glob('LAUNCH_attempt*.private.json')))+1)+'.private.json')).write_text(json.dumps(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),indent=2)+'\n')
    if x.returncode:raise RuntimeError(x.stderr)
    receipt=json.loads(x.stdout);(rt/'LAUNCH.private.json').write_text(json.dumps(receipt,indent=2)+'\n')
    h=dict(current_round=r.name,current_root=c['root'],source_commit=sha,current_status='RUNNING_NOT_RESULTS',current_phase='PREFLIGHT',
        coordinator_pid=receipt['pid'],launched_at=receipt['launched_at'],deadline=receipt['deadline'],GPU_launched=True,real_GPU_preflights_passed=0,
        new_trajectories=8,new_updates=2576,authorized_gpus=[0,1],collector_ready=True,delivery_helper_ready=True,monitor_created=False,monitor_id="lca",monitor_interval_minutes=30)
    (rt/'HANDOFF.private.json').write_text(json.dumps(h,indent=2)+'\n');return receipt


if __name__=='__main__':print(json.dumps(launch(json.load(__import__('sys').stdin)['round_dir']),indent=2))
