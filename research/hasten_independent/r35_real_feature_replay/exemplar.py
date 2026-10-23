"""Small deterministic TRAIN feature memory, selected only at class arrival."""
from pathlib import Path
import torch
BANK={}
def clear_exemplars():BANK.clear()
def bank_sizes():return {str(c):len(v) for c,v in sorted(BANK.items())}
@torch.no_grad()
def register_exemplars(cache,all_labels,out):
 labels=all_labels[cache['ids']];x=cache['x'];assert x.shape==(len(labels),512)
 assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6)
 for c in sorted(set(labels.tolist())):
  assert c not in BANK,'FUTURE_OR_DUPLICATE_EXEMPLAR_CLASS'
  rows=x[labels==c];k=min(8,len(rows));indices=torch.arange(k)*(len(rows)-1)//max(k-1,1)
  BANK[c]=rows[indices].detach().clone().cpu()
 assert list(sorted(BANK))==list(range(max(BANK)+1))
 # One current memory snapshot, private features; reconstruction on resume uses arrived caches.
 tmp=Path(out)/'exemplars.private.tmp';torch.save(BANK,tmp);tmp.replace(Path(out)/'exemplars.private.pt')
@torch.no_grad()
def pick(label,counter,device,dtype):
 x=BANK[label];assert 1<=len(x)<=8
 return x[counter%len(x):counter%len(x)+1].to(device=device,dtype=dtype)
