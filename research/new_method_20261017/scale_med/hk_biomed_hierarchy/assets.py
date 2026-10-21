"""Pinned single-copy public checkpoint download within the experiment guard."""
import json,os,time
from pathlib import Path
import requests
MODEL='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
REV='9f341de24bfb00180f1b847274256e9b65a3a32e'
TEXT='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
TREV='d673b8835373c6fa116d6d8006b33d48734e305d'
def prepare(root):
 root.mkdir(exist_ok=True);(root/'text').mkdir(exist_ok=True)
 proxy=os.environ['HTTPS_PROXY'];assert proxy.startswith('http://127.0.0.1:')
 session=requests.Session();session.trust_env=False;session.proxies={'http':proxy,'https':proxy}
 files=[(MODEL,REV,n,root/n) for n in ['open_clip_config.json','open_clip_pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']]+[(TEXT,TREV,'config.json',root/'text/config.json')]
 receipts=[]
 for repo,revision,name,path in files:
  # Retry transport errors only; no alternate model, revision or scoring.
  for attempt in range(3):
   try:
    with session.get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',stream=True,timeout=(20,120)) as r:
     r.raise_for_status();limit=783705670 if name.endswith('.bin') else 2000000;n=0
     with path.open('wb') as f:
      for chunk in r.iter_content(1024*1024):
       n+=len(chunk);assert n<=limit;f.write(chunk)
     if name.endswith('.bin'):assert n==783705670
     elif r.headers.get('Content-Length'):assert n==int(r.headers['Content-Length'])
    break
   except requests.RequestException:
    if attempt==2:raise
    time.sleep(2)
  receipts.append(dict(file=name,bytes=path.stat().st_size,revision=revision))
 (root.parent/'ASSET_RECEIPT.json').write_text(json.dumps(receipts,indent=2))
 return root,REV
