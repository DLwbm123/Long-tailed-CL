import torch,json
def write(p,a):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(a,indent=2,allow_nan=False));t.replace(p)
def schedule(y,groups):
 gen=torch.Generator().manual_seed(74250);members={c:{g:torch.tensor([i for i,v in enumerate(groups) if v==g and int(y[i])==c]) for g in sorted({groups[i] for i in torch.where(y==c)[0].tolist()})} for c in range(4)};orders=[];assert all(len(v)>=8 for v in members.values())
 for step in range(800):
  ids=[]
  for c in range(4):
   keys=list(members[c]);chosen=torch.randperm(len(keys),generator=gen)[:8]
   for j in chosen:
    val=members[c][keys[int(j)]];ids.append(int(val[torch.randint(len(val),(1,),generator=gen)]))
  assert len({groups[i] for i in ids})==32;orders.append(torch.tensor(ids))
 return torch.stack(orders)
