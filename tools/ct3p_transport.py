"""Neutral stream endpoint: lossless compression, original checkpoint bytes preserved."""
import json,shutil,sys
import zstandard
from pathlib import Path
root=Path('/tmp/p22root/output/private')
if sys.argv[1]=='send':
    q=json.loads((root/'REQUEST.json').read_text());name=q['name']
    assert q['op']=='PUT' and Path(name).name==name and name.endswith('.pt')
    with (root/name).open('rb') as src,zstandard.ZstdCompressor(level=3,threads=4).stream_writer(sys.stdout.buffer,closefd=False) as dst:shutil.copyfileobj(src,dst,1024**2)
elif sys.argv[1]=='receive':
    with zstandard.ZstdDecompressor().stream_reader(sys.stdin.buffer,closefd=False) as src,(root/'parent.part').open('wb') as dst:shutil.copyfileobj(src,dst,1024**2)
else:raise ValueError('unknown stream direction')
