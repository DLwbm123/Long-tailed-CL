"""Whole-window limits include outside cache and conservative local/Git reserve."""
from pathlib import Path
import stat
SOFT=17000000000;HARD=20000000000;OUTSIDE=1000000000+1040384
def used(root):
 total=0
 for p in Path(root).rglob('*'):
  try:s=p.lstat()
  except FileNotFoundError:continue
  if stat.S_ISREG(s.st_mode):total+=max(s.st_size,s.st_blocks*512)
  elif stat.S_ISLNK(s.st_mode):total+=s.st_size
 return total
def allowed(remote,reserve=0):return remote+OUTSIDE<SOFT and remote+OUTSIDE+reserve<HARD
def check_space(root,reserve=0):
 n=used(root)
 if not allowed(n,reserve):raise RuntimeError('WINDOW_SPACE_GUARD')
 return n
def selfcheck():
 assert allowed(0) and not allowed(SOFT-OUTSIDE) and not allowed(0,HARD-OUTSIDE)
