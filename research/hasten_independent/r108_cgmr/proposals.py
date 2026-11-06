"""Reward-independent bounded local/bundle repair; unique full memories are actions."""
import torch

def key(bank):return bank.pid*8+bank.view

def validate(pool,base,memory,anchors,max_removed):
 memory.assert_subset(pool);kb=key(base);km=key(memory);assert len(memory)==len(base) and int((memory.view==0).sum())==int((base.view==0).sum());assert torch.isin(anchors,km).all();assert int((~torch.isin(kb,km)).sum())<=max_removed
 assert set(memory.y.tolist())==set(base.y.tolist());return True

@torch.no_grad()
def generate(pool,base,cover,generator,counter=None):
 kp=key(pool);parent_ids=pool.parents.tolist();parent_cap=dict(zip(parent_ids,torch.bincount(pool.parent_index[pool.view>0],minlength=len(pool.parents)).tolist()));base_parent_set=set(base.parents.tolist());kb=key(base);base_ix=torch.searchsorted(kp,kb);anchors=key(pool)[cover];limit=min(24,len(base)//10);memories=[base];edits=[dict(type='NOOP',operations=0,removed_tokens=0,replaced_parents=0)];seen={tuple(kb.tolist())};aud=[]
 if limit==0 or len(pool)==len(base):return memories,edits,[dict(type=t,target_operations=n,attempts=0,present=False) for t,n in [('LOCAL',1),('LOCAL',1),('LOCAL',4),('LOCAL',4),('BUNDLE',1),('BUNDLE',1),('BUNDLE',4),('BUNDLE',4)]]
 for typ,nop in [('LOCAL',1),('LOCAL',1),('LOCAL',4),('LOCAL',4),('BUNDLE',1),('BUNDLE',1),('BUNDLE',4),('BUNDLE',4)]:
  chosen=None;tries=0
  for attempt in range(64):
   tries+=1;sel=base_ix.clone();removed=set();added=set();removed_parent=set();added_parent=set();removed_count=0;ok=True
   for op in range(nop):
    if typ=='LOCAL':
     rem=sel[(pool.view[sel]>0)&torch.isin(kp[sel],kb)]
     if removed:rem=rem[~torch.isin(kp[rem],torch.tensor(sorted(removed),device=kp.device))]
     add=torch.where((pool.view>0)&~torch.isin(torch.arange(len(pool),device=kp.device),sel)&torch.isin(pool.pid,pool.pid[sel[pool.view[sel]==0]])&~torch.isin(kp,kb))[0]
     if added:add=add[~torch.isin(kp[add],torch.tensor(sorted(added),device=kp.device))]
     if not len(rem) or not len(add) or removed_count+1>limit:ok=False;break
     a=rem[int(torch.randint(len(rem),(1,),device=kp.device,generator=generator))];b=add[int(torch.randint(len(add),(1,),device=kp.device,generator=generator))];sel=torch.cat([sel[sel!=a],b.reshape(1)]);removed.add(int(kp[a]));added.add(int(kp[b]));removed_count+=1
    else:
     parents=pool.pid[sel[pool.view[sel]==0]];rempar=[int(i) for i in parents.tolist() if i not in removed_parent and i not in added_parent and not bool(torch.isin(torch.tensor([i*8],device=kp.device),anchors).any())]
     addpar=[int(i) for i in parent_ids if i not in set(parents.tolist()) and i not in removed_parent and i not in added_parent and i not in base_parent_set]
     if not rempar or not addpar:ok=False;break
     a=rempar[int(torch.randint(len(rempar),(1,),device=kp.device,generator=generator))];old=sel[pool.pid[sel]==a];k=len(old)
     possible=[i for i in addpar if parent_cap[i]>=k-1]
     if not possible or removed_count+k>limit:ok=False;break
     b=possible[int(torch.randint(len(possible),(1,),device=kp.device,generator=generator))];glob=pool.gidx[pool.pid[pool.gidx]==b];local=torch.where((pool.pid==b)&(pool.view>0))[0];local=local[torch.randperm(len(local),device=kp.device,generator=generator)[:k-1]];new=torch.cat([glob,local]);sel=torch.cat([sel[pool.pid[sel]!=a],new]);removed_parent.add(a);added_parent.add(b);removed_count+=k
   if ok:
    candidate=pool.take(sel);validate(pool,base,candidate,anchors,limit);signature=tuple(key(candidate).tolist())
    if signature not in seen:chosen=candidate;seen.add(signature);break
  if chosen is not None:memories.append(chosen);edits.append(dict(type=typ,operations=nop,removed_tokens=int((~torch.isin(kb,key(chosen))).sum()),replaced_parents=len(removed_parent)))
  aud.append(dict(type=typ,target_operations=nop,attempts=tries,present=chosen is not None))
 if counter is not None:counter['proposal_calls']=counter.get('proposal_calls',0)+1;counter['proposal_candidates']=counter.get('proposal_candidates',0)+len(memories);counter['proposal_attempts']=counter.get('proposal_attempts',0)+sum(x['attempts'] for x in aud)
 return memories,edits,aud
