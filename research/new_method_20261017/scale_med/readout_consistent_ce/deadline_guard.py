"""Reserve ten seconds for shutdown within the original three-hour envelope."""
import json,os,resource,signal,time
from pathlib import Path
r=Path(os.environ['ROUND_ROOT']);clock=json.loads((r/'ROUND_CLOCK.private.json').read_text());pid=json.loads((r/'DRIVER_LAUNCH.private.json').read_text())['driver_pid'];start=time.time();forced=False;retired=False
while not (r/'DRIVER_EXIT.json').exists():
 try:
  command=Path('/proc')/str(pid)/'cmdline'
  if not command.exists():retired=True;break
  assert b'/tmp/q746/driver.py' in command.read_bytes() and os.getpgid(pid)==pid,'driver identity/group mismatch'
 except ProcessLookupError:retired=True;break
 if time.time()>=clock['deadline_unix']-10:
  forced=True;os.killpg(pid,signal.SIGTERM);time.sleep(5)
  try:os.killpg(pid,signal.SIGKILL)
  except ProcessLookupError:pass
  (r/'HARD_DEADLINE_FAILURE.json').write_text(json.dumps(dict(reason='SHARED_THREE_HOUR_ENVELOPE',partial_outputs_preserved=True)))
  break
 time.sleep(min(30,max(.01,clock['deadline_unix']-10-time.time())))
u=resource.getrusage(resource.RUSAGE_SELF)
(r/'HARD_DEADLINE_EXIT.json').write_text(json.dumps(dict(forced=forced,driver_retired_without_final_receipt=retired,CPU_seconds=u.ru_utime+u.ru_stime,overlapping_wall_seconds=time.time()-start,do_not_add_overlapping_wall_to_driver=True,deadline_unix=clock['deadline_unix'],completed_unix=time.time()),indent=2))
