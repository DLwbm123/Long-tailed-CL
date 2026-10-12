"""After public push, verify exact report and save an independent anonymous NAS delivery."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def deliver(round_dir):
    r=Path(round_dir);runtime=r/'local_runtime'
    if (runtime/'DELIVERY.private.json').exists():raise ValueError('Already delivered')
    h=json.loads((runtime/'HANDOFF.private.json').read_text());a=json.loads((r/'results/AUDIT.json').read_text())
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();env=os.environ.copy()
    for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):env[k]='http://127.0.0.1:7897'
    for k in ('NO_PROXY','no_proxy'):env[k]='localhost,127.0.0.1'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/feat/prototype-analytic'],env=env,text=True).split()[0]
    if remote!=sha:raise ValueError('Push and exact remote verification required')
    url=f'https://raw.githubusercontent.com/DLwbm123/Long-tailed-CL/{sha}/{r}/results/REPORT_ZH.md'
    x=subprocess.run(['curl','--proxy',env['HTTPS_PROXY'],'--fail','--max-time','45','--silent','--show-error','--output',str(runtime/'REPORT_HTTP.private.md'),'--write-out','%{http_code}',url],env=env,text=True,capture_output=True)
    if x.returncode or x.stdout!='200':raise RuntimeError(x.stderr or 'Anonymous report unavailable')
    receipt=dict(status='DELIVERED',commit=sha,branch='feat/prototype-analytic',anonymous_http_status=200,nas_saved=True,audit=a,delivered_at=time.time())
    h.update(current_status='COMPLETE_DELIVERED' if a['status']=='PASS' else 'INCOMPLETE_DELIVERED',delivery_commit=sha,costs_final=True,current_active={})
    archive=subprocess.check_output(['git','archive',sha,'tools','third_party/air_apart_pinned',str(r)])
    code='import base64,io,json,shutil,subprocess,tarfile,time\nfrom pathlib import Path\nh='+repr(h)+'\nr='+repr(receipt)+'\na=base64.b64decode('+repr(base64.b64encode(archive).decode())+')\n'+'''
root=Path(h['current_root']);p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
assert not (root/'DELIVERY.private.json').exists()
assert subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip().startswith('nfs') and shutil.disk_usage(root).free>1024**3
probe=root/('.probe_'+str(time.time_ns()));probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
dest=root/'public_delivery';dest.mkdir()
with tarfile.open(fileobj=io.BytesIO(a)) as tf:
 for m in tf.getmembers():
  q=Path(m.name);assert not q.is_absolute() and '..' not in q.parts and not m.issym() and not m.islnk()
  t=dest/q
  if m.isdir():t.mkdir(parents=True,exist_ok=True)
  elif m.isfile():t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(tf.extractfile(m).read())
(root/'DELIVERY.private.json').write_text(json.dumps(r,indent=2)+chr(10));(root/'HANDOFF.private.json').write_text(json.dumps(h,indent=2)+chr(10));print(json.dumps(dict(commit=r['commit'],nas_saved=True)))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    (runtime/'DELIVERY.private.json').write_text(json.dumps(receipt,indent=2)+'\n');(runtime/'HANDOFF.private.json').write_text(json.dumps(h,indent=2)+'\n')
    return receipt


if __name__=='__main__':print(json.dumps(deliver(json.load(sys.stdin)['round_dir']),indent=2))
