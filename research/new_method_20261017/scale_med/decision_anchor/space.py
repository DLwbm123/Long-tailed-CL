from pathlib import Path
import stat,os,time
SOFT=17000000000;HARD=20000000000;OUTSIDE=1001040384
ROUND_CAP=2**30;ROUND_OUTSIDE=16*2**20
def used(root):
 total=0
 for p in Path(root).rglob('*'):
  try:s=p.lstat()
  except FileNotFoundError:continue
  if stat.S_ISREG(s.st_mode):total+=max(s.st_size,s.st_blocks*512)
  elif stat.S_ISLNK(s.st_mode):total+=s.st_size
 return total
def check_space(root,reserve=0):
 n=used(root);assert n+OUTSIDE<SOFT and n+OUTSIDE+reserve<HARD,'WINDOW_SPACE_GUARD'
 rr=os.environ.get('ROUND_ROOT')
 if rr:assert used(rr)+ROUND_OUTSIDE+reserve<ROUND_CAP,'ROUND_SPACE_GUARD'
 deadline=float(os.environ.get('ROUND_DEADLINE','inf'));assert time.time()<deadline,'ROUND_WALL_ENVELOPE'
 return n
