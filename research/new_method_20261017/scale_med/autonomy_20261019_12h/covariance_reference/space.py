"""All new runtime writes are confined to this window; reserve 1GB externally."""
from pathlib import Path
import stat
SOFT=17000000000;HARD=19000000000

def used(root):
 total=0
 for p in Path(root).rglob('*'):
  try:s=p.lstat()
  except FileNotFoundError:continue
  if stat.S_ISREG(s.st_mode):total+=max(s.st_size,s.st_blocks*512)
  elif stat.S_ISLNK(s.st_mode):total+=s.st_size
 return total

def check_space(root,reserve=0):
 n=used(root)
 if n>=SOFT or n+reserve>HARD:raise RuntimeError('WINDOW_SPACE_GUARD')
 return n
