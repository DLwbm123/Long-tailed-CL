"""Launch exactly one explicitly authorized certificate-based continuation."""
import json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path(os.environ['RUN_CONFIG']).parent;out=r/'run'
assert json.loads((out/'status.private.json').read_text())['state']=='FAILED'
assert json.loads((out/'status.private.json').read_text())['costs']['attempts']==2
assert not (r/'cert_guardian.private.json').exists()
for name in ['launch.private.json','guardian.private.json','resume_launch.private.json','resume_guardian.private.json']:
 item=json.loads((r/name).read_text());pid=item.get('pid',item.get('guardian_pid'));assert not pathlib.Path(f'/proc/{pid}').exists()
a=out/'failure_attempt2';a.mkdir(exist_ok=False)
for name in ['status.private.json','TRAINING_AUDIT.json','costs.json']:
 p=out/name
 if p.exists():shutil.copyfile(p,a/name)
shutil.copyfile(r/'resume_terminal.private.json',a/'terminal.private.json')
baseline=json.loads((r/'zero_analysis_costs.json').read_text())['cumulative_measured_seconds']+json.loads((r/'certificate_check.private.json').read_text())['wall_seconds']
os.environ['CUMULATIVE_BASE_WALL']=str(baseline)
with (r/'cert.guard.private.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).with_name('guard.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
receipt=dict(guardian_pid=p.pid,started=time.time(),prior_and_repair_wall_seconds=baseline,remaining_wall_seconds=10800-baseline,source_commit=os.environ['SOURCE_COMMIT'])
(r/'cert_guardian.private.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
