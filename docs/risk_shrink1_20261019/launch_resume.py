import json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path(os.environ['RUN_CONFIG']).parent;out=r/'run'
assert json.loads((out/'status.private.json').read_text())['state']=='FAILED'
assert not pathlib.Path('/proc/871584').exists() and not pathlib.Path('/proc/871581').exists()
assert not (r/'resume_guardian.private.json').exists()
a=out/'failure_attempt1';a.mkdir(exist_ok=True)
for name in ['status.private.json','TRAINING_AUDIT.json','costs.json']:
 p=out/name
 if p.exists():shutil.copyfile(p,a/name)
shutil.copyfile(r/'terminal.private.json',a/'terminal.private.json')
baseline=json.loads((r/'terminal.private.json').read_text())['elapsed_seconds']+json.loads((out/'FAILURE_VERIFICATION.json').read_text())['verification_wall_seconds']+json.loads((r/'repair_setup.private.json').read_text())['wall_seconds']+0.5003781318664551
os.environ['CUMULATIVE_BASE_WALL']=str(baseline)
with (r/'resume.guard.private.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).with_name('guard.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
receipt=dict(guardian_pid=p.pid,started=time.time(),prior_and_repair_wall_seconds=baseline,remaining_wall_seconds=10800-baseline,source_commit=os.environ['SOURCE_COMMIT'])
(r/'resume_guardian.private.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
