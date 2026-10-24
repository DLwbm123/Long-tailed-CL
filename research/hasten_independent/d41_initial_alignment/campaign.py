import os,json,subprocess,time,signal
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
for ds in ['NIH','ISIC','HK']:
    if (out/ds/'RESULTS.json').exists():
        assert json.loads((out/ds/'RESULTS.json').read_text())['passed'];continue
    spec=out/ds/'config.private.json';env=os.environ.copy();env['RUN_CONFIG']=str(spec)
    d=out/ds;runtime=d/'runtime';runtime.mkdir(exist_ok=True)
    key=json.loads(spec.read_text()).get('runtime_dataset',json.loads(spec.read_text())['dataset'])
    (runtime/'knowledge').mkdir(exist_ok=True);(runtime/'knowledge/virtual_templates.json').write_text(json.dumps({key:[json.loads(spec.read_text())['parent_template']]}))
    (out/'STATUS.json').write_text(json.dumps(dict(phase='INITIAL_ALIGNMENT_PROBE',dataset=ds,time=time.time())))
    with (d/'worker.private.log').open('a') as log:
        p=subprocess.Popen([cfg['python'],'/tmp/q1064/worker.py'],env=env,cwd=runtime,stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:
            try:rss=int(next(t for t in Path(f'/proc/{p.pid}/status').read_text().splitlines() if t.startswith('VmRSS:')).split()[1])*1024
            except (FileNotFoundError,StopIteration):rss=0
            if rss>8*2**30:os.killpg(os.getpgrp(),signal.SIGTERM)
            time.sleep(1)
        assert p.returncode==0,(ds,p.returncode)
result={ds:json.loads((out/ds/'RESULTS.json').read_text()) for ds in ['NIH','ISIC','HK']}
assert all(v['passed'] for v in result.values())
(out/'RESULTS.json').write_text(json.dumps(result,indent=2));(out/'STATUS.json').write_text(json.dumps(dict(phase='COMPLETE',time=time.time())))
