import json
from pathlib import Path
import torch

def initial(cfg):
 obj=torch.load(Path(cfg['reference'])/'inputs/0.private.pt',map_location='cpu',weights_only=False)
 tr=~obj['valid'];ids=torch.where(tr)[0];assert len(ids)==2555
 return dict(y=obj['y'][tr],g=obj['g'][tr].double(),groups=[obj['groups'][i] for i in ids],row_ids=ids)

def pixels(cfg):
 # Shared chunks can contain excluded rows; select original-training rows before any forward.
 plan=torch.load(Path(cfg['root'])/'prep/PLAN.private.pt',weights_only=False)
 keep=set(plan['row_ids'].tolist());parts=[];total=0
 idx=json.loads((Path(cfg['matched'])/'pixels_0/INDEX.json').read_text());assert idx['rows']==3236
 for cell in idx['chunks']:
  x=torch.load(Path(cfg['matched'])/'pixels_0'/cell['file'],weights_only=False)
  mask=torch.tensor([cell['first']+i in keep for i in range(len(x))]);parts.append(x[mask]);total+=len(x)
 assert total==3236
 out=torch.cat(parts);assert len(out)==2555;return out
