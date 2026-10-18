"""One bounded run, with no automatic extension or performance-based retry."""
import os,json,subprocess,time
from pathlib import Path
root=Path(json.loads(Path(os.environ['RUN_CONFIG']).read_text())['output']);start=time.time();code=1
with (root.parent/'worker.log').open('wb') as log:
    for phase in ['prepare','fit','evaluate']:
        p=subprocess.Popen(['/tmp/q570/python','/tmp/q570/entry.py'],env=dict(os.environ,PHASE=phase),stdout=log,stderr=subprocess.STDOUT)
        (root/'worker.private.json').write_text(json.dumps(dict(pid=p.pid,phase=phase)))
        try:code=p.wait(timeout=max(1,3600-(time.time()-start)))
        except subprocess.TimeoutExpired:p.terminate();p.wait();code=124
        if code:break
(root/'EXIT.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.time()-start)))
raise SystemExit(code)
