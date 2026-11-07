"""Bounded append-only engineering/science accounting."""
import json
import os
import time
from pathlib import Path


class BudgetLedger:
    def __init__(self,path,deadline,max_bytes=20_000_000_000):
        self.path=Path(path);self.deadline=deadline;self.max_bytes=max_bytes
        assert max_bytes<50_000_000_000
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def append(self,event,**values):
        assert time.time()<self.deadline, 'ORIGINAL_DEADLINE_EXCEEDED'
        allocated=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in self.path.parent.rglob('*') if p.is_file() and not p.is_symlink())
        assert allocated+512*2**20<self.max_bytes, 'STORAGE_EARLY_STOP'
        row=dict(event=event,wall_epoch=time.time(),CPU_seconds=time.process_time(),allocated_bytes=allocated,**values)
        with self.path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')


def require_admission(config):
    assert config.get('science') is False, 'SCIENCE_REQUIRES_SEPARATE_FROZEN_PROTOCOL_AND_NATIVE_ADMISSION'
    assert config.get('max_wall_seconds',0)<=86400
    assert config.get('max_updates',0)<=5000, 'ENGINEERING_UPDATE_CAP'

