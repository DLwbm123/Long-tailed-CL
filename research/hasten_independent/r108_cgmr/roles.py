"""Immutable component roles, assigned on first legitimate class arrival only."""
import random,math
import torch
from memory import Bank
class Roles:
 def __init__(self):self.role={};self.arrived=set()
 def arrive(self,ids,rows):
  groups={}
  for i in ids:groups.setdefault(int(rows[i]['original_label']),{}).setdefault(rows[i]['identity_component'],[]).append(i)
  for label,components in sorted(groups.items()):
   if label in self.arrived:continue
   keys=sorted(components);r=random.Random(947108+label);r.shuffle(keys);q=min(4,max(1,len(keys)//4)) if len(keys)>=3 else 0
   for role in ['FIT','AUDIT']:
    needed=max(0,q-sum(self.role.get(k)==role for k in keys));free=[k for k in keys if k not in self.role]
    # Never consume the last unassigned support component to fill a query quota.
    for k in free[:min(needed,max(0,len(free)-int(not any(self.role.get(x)=='S' for x in keys))))]:self.role[k]=role
   for k in keys:self.role.setdefault(k,'S')
   self.arrived.add(label)
 def split(self,pool,rows):
  groups={'S':[],'FIT':[],'AUDIT':[]}
  for pid in pool.parents.tolist():groups[self.role[rows[pid]['identity_component']]].append(pid)
  return {k:pool.parents_only(torch.tensor(v,device=pool.z.device,dtype=torch.long)) for k,v in groups.items()}

def relabel(bank,mapping):
 m=torch.tensor(mapping,device=bank.z.device);return Bank(bank.z,bank.pid,m[bank.y],bank.view)
