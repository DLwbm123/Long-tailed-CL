"""Neutral detached admission guardian. Never restarts an existing run."""
import json,os,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent
assert cfg['engineering_check']=='PASS' and not (root/'guardian.private.json').exists()
assert (Path(cfg['output']).exists() and Path(cfg['resume_checkpoint']).exists()) if cfg.get('resume_checkpoint') else not Path(cfg['output']).exists()
assert time.time()<cfg['hard_deadline_epoch']-60
with (root/'guard.private.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('guard.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL);d=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=time.time(),source_commit=os.environ['SOURCE_COMMIT'],original_start_epoch=cfg['original_start_epoch'],hard_deadline_epoch=cfg['hard_deadline_epoch']);(root/'guardian.private.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
