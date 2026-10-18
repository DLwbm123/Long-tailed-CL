"""Run learner, privileged auditor and sealed test in separate interpreters."""
import json,os,subprocess,time
from pathlib import Path
root=Path(json.loads(Path(os.environ['RUN_CONFIG']).read_text())['output'])
start=time.time();code=1
with (root.parent/'worker.log').open('wb') as log:
    for phase in ['fit','audit','test']:
        env=dict(os.environ,PHASE=phase)
        p=subprocess.Popen(['/tmp/q560/python','/tmp/q560/entry.py'],env=env,stdout=log,stderr=subprocess.STDOUT)
        (root/'worker.private.json').write_text(json.dumps(dict(pid=p.pid,phase=phase)))
        code=p.wait()
        if code:break
(root/'EXIT.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.time()-start)))
raise SystemExit(code)
