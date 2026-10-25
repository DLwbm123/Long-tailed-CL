"""Receive private stage metadata over existing SSH, never credentials."""
import json,sys
from pathlib import Path
from common import write
cfg=json.loads(Path('/tmp/q1094/config.json').read_text());job=json.load(sys.stdin);assert isinstance(job['id'],int) and 0<=job['id']<100;r=Path(cfg['output'])/'jobs'/str(job['id']);r.mkdir(parents=True,exist_ok=True);(r/'runtime').mkdir(exist_ok=True);c=dict(cfg);c.update(output=str(r),job=job,entry='/tmp/q1094/producer.py',runtime=str(r/'runtime'));write(r/'config.private.json',c)
