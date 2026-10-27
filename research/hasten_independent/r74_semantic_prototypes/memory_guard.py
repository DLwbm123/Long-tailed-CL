"""Conservative aggregate RSS of only this experiment's process group."""
import os
from pathlib import Path

def group_rss(pgid,proc='/proc',extra_pid=None):
    total=0;included=set()
    for p in Path(proc).iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=pgid and int(p.name)!=extra_pid:continue
            rss=int(next(x for x in (p/'status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
        except (OSError,StopIteration,ValueError):continue
        total+=rss;included.add(int(p.name))
    return total,sorted(included)
