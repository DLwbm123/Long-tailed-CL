"""Sequential fixed watchdogs; unused solver time is never reallocated."""
import json,os,subprocess,time
from pathlib import Path
from report import generate
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);mode=os.environ['MODE'];start=time.time()
def write(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def stop(p):
    p.terminate()
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
if mode=='prepare':
    with (root.parent/'prepare.log').open('wb') as f:
        p=subprocess.Popen(['/tmp/q580/python','/tmp/q580/prepare.py'],stdout=f,stderr=subprocess.STDOUT)
        write(root/'worker.private.json',dict(pid=p.pid,phase='prepare'))
        try:code=p.wait(timeout=900)
        except subprocess.TimeoutExpired:
            stop(p);code=124;write(root/'PREPARE_EXIT.json',dict(exit_code=code,wall_seconds=time.time()-start,state='UNRESOLVED_PREPARATION_TIME_LIMIT'))
    if code:raise SystemExit(code)
else:
    assert mode=='solve' and (root/'DISPATCH_SEAL.json').exists()
    known=json.loads((root/'KNOWN_WITNESSES.json').read_text()) if (root/'KNOWN_WITNESSES.json').exists() else {}
    for space in ['G','C','U']:
        assert not (root/(space+'.result.json')).exists();tick=time.time();peak=0;reason=None
        if space in known:
            write(root/(space+'.result.json'),dict(known[space],watchdog_wall_seconds=0,watchdog_peak_RSS_bytes=0,worker_exit_code=None));continue
        with (root/(space+'.worker.private.log')).open('wb') as f:
            p=subprocess.Popen(['/tmp/q580/python','/tmp/q580/solve.py'],env=dict(os.environ,SPACE=space),stdout=f,stderr=subprocess.STDOUT)
            write(root/'worker.private.json',dict(pid=p.pid,phase='solve',space=space));write(root/'status.json',dict(state='RUNNING',phase='solve',space=space))
            while p.poll() is None:
                try:
                    lines=Path('/proc',str(p.pid),'status').read_text().splitlines();rss=int(next(s for s in lines if s.startswith('VmRSS:')).split()[1])*1024;peak=max(peak,rss)
                except (FileNotFoundError,StopIteration):pass
                if peak>16*2**30:reason='UNRESOLVED_MEMORY_LIMIT'
                if time.time()-tick>=300:reason='UNRESOLVED_TIME_LIMIT'
                if reason:stop(p);break
                try:p.wait(timeout=.5)
                except subprocess.TimeoutExpired:pass
        receipt=root/(space+'.result.json')
        if reason or not receipt.exists():write(receipt,dict(space=space,classification=reason or 'UNRESOLVED_ENGINEERING_FAILURE',full_family_infeasible_proven=False))
        r=json.loads(receipt.read_text());r.update(watchdog_wall_seconds=time.time()-tick,watchdog_peak_RSS_bytes=peak,worker_exit_code=p.returncode);write(receipt,r)
    write(root/'EXIT.json',dict(exit_code=0,wall_seconds=time.time()-start));generate(root,root.parent/'public_results')
    write(root/'status.json',dict(state='COMPLETE',automatic_extension=False,results={a:json.loads((root/(a+'.result.json')).read_text())['classification'] for a in ['G','C','U']}))
