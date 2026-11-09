import re
import json
from pathlib import Path

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)

def normalize(name):
    return re.sub(r'[\s_\-]+', '', name.lower().strip())

def advantages(rewards):
    return (rewards-rewards.mean(-1,keepdim=True))/(rewards.std(-1,unbiased=False,keepdim=True)+1e-6)
