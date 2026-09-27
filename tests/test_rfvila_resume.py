"""Resume accepts only named legacy locks and explicit time override."""
import tempfile,time
from pathlib import Path
from tools.run_rfvila import Budget,fingerprints,check_lock
from tools.run_nb2_vlm_r1 import _write,_sha

def main():
 with tempfile.TemporaryDirectory() as td:
  r=Path(td);_write(r/'CLOCK_LOCK.json',{'t0_epoch':time.time()-100000,'t0_monotonic':time.monotonic()-100000})
  b=Budget(r)
  try:b.check('fit');raise AssertionError('expired budget accepted')
  except TimeoutError:pass
  for name in ['SOURCE_LOCK.json','PROTOCOL_LOCK.json','RANDOM_MAP_LOCK.json']:_write(r/name,{})
  p=r/'stages/ISIC/1995/task_02';p.mkdir(parents=True);(p/'W.npz').write_bytes(b'fixed')
  _write(p/'STATE_LOCK.json',{'locks':fingerprints(r),'W_sha256':_sha(p/'W.npz')})
  clock=(r/'CLOCK_LOCK.json').read_bytes();old_lock=(p/'STATE_LOCK.json').read_bytes()
  _write(r/'RESUME_AMENDMENT.json',{'ignore_time_budget':True,'legacy_stages':{str(p.relative_to(r)):_sha(p/'STATE_LOCK.json')}})
  _write(r/'SOURCE_LOCK_RESUME.json',{'new_source':True})
  assert Budget(r).elapsed()>100000;Budget(r).check('fit');check_lock(r,p)
  assert clock==(r/'CLOCK_LOCK.json').read_bytes() and old_lock==(p/'STATE_LOCK.json').read_bytes()
  _write(p/'STATE_LOCK.json',{'locks':{'source':'changed'},'W_sha256':_sha(p/'W.npz')})
  try:check_lock(r,p);raise AssertionError('changed state accepted')
  except ValueError:pass
 print('RESUME_OVERRIDE_AND_IMMUTABLE_STATE_TEST_PASS')
if __name__=='__main__':main()
