"""Only this run's acknowledged, regenerable staging files; preserve all receipts."""
import json,sys
from pathlib import Path
from common import write
cfg=json.loads(Path('/tmp/q1102/config.json').read_text());j=json.load(sys.stdin);assert isinstance(j['id'],int);root=Path(cfg['output']);r=root/'jobs'/str(j['id']);assert (r/'TRANSFER_ACK.json').exists();meta=json.loads((r/'export/META.json').read_text());removed=0;files=[]
for c in meta['chunks']:
 for name in [c['file'],c['base']]:
  p=r/'export'/name;assert p.parent==r/'export' and not p.is_symlink()
  if p.exists():removed+=p.stat().st_size;files.append(name);p.unlink()
p=r/'reference.private.pt'
if p.exists():removed+=p.stat().st_size;p.unlink()
write(r/'STAGING_CLEANUP.json',dict(removed_regenerable_bytes=removed,files=files,all_scientific_receipts_retained=True))
