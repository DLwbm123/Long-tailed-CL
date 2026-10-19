"""Launch one background guardian after frozen preparation, without retry."""
import json,os,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']).parent
assert not (r/'guardian.private.json').exists() and not Path(cfg['output']).exists()
assert cfg['engineering_check']=='PASS' and cfg['new_encoding_cap']==0
with (r/'guard.private.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('guard.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 item=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=time.time(),source_commit=os.environ['SOURCE_COMMIT']);(r/'guardian.private.json').write_text(json.dumps(item,indent=2));print(json.dumps(item))
