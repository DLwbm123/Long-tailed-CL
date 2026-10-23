"""Deadline fallback uses only local proc and /tmp; no NFS dependency."""
import os,json,time,signal
with open('/tmp/q1044/watch_config.json') as f:cfg=json.load(f)
receipt='/tmp/q1044/deadline_watch.json'
def alive(pid,tick):
    try:
        with open('/proc/%s/stat'%pid) as f:fields=f.read().rsplit(')',1)[1].split()
        return fields[19]==tick and fields[0]!='Z'
    except FileNotFoundError:return False
def record(phase,**kw):
    with open(receipt+'.tmp','w') as f:json.dump(dict(phase=phase,time=time.time(),deadline_unix=cfg['deadline_unix'],watch_pid=os.getpid(),**kw),f)
    os.replace(receipt+'.tmp',receipt)
def live():return [int(pid) for pid,tick in cfg['targets'].items() if alive(pid,tick)]
def terminate(sig):
    current=live()
    if any(os.getpgid(pid)==cfg['group'] for pid in current):
        try:os.killpg(cfg['group'],sig)
        except ProcessLookupError:pass
    if 1373182 in current:
        try:os.kill(1373182,sig)
        except ProcessLookupError:pass
record('ARMED',targets=cfg['targets'])
while time.time()<cfg['deadline_unix']-30:
    if not live():record('TARGETS_EXITED');raise SystemExit
    time.sleep(1)
terminate(signal.SIGTERM);record('TERM_SENT')
time.sleep(5)
terminate(signal.SIGKILL);record('KILL_SENT',pending_pids=live())
