"""Archive first failure and launch one explicitly authorized continuation."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']).parent;out=Path(cfg['output'])
assert json.loads((out/'status.private.json').read_text())['state']=='FAILED';assert not (r/'resume_guardian.private.json').exists()
for name in ['guardian.private.json','launch.private.json']:
 i=json.loads((r/name).read_text());p=Path(f"/proc/{i['pid']}/stat");assert not p.exists() or p.read_text().split()[21]!=i['start_ticks']
archive=out/'failure_attempt1';archive.mkdir(exist_ok=False)
for name in ['config.private.json','status.private.json','failure.private.json','optimization_0.json','last_candidate_0_plain_correction.private.npy','FAILURE_VERIFICATION.json']:
 shutil.copyfile(out/name,archive/name)
shutil.copyfile(r/'terminal.private.json',archive/'terminal.private.json')
with (r/'resume.guard.private.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('guard.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 receipt=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=time.time(),source_commit=os.environ['SOURCE_COMMIT'],attempts=2);(r/'resume_guardian.private.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
