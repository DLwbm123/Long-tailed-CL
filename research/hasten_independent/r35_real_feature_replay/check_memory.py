"""Membership, rare-class coverage, cycle, memory and RNG checks."""
import tempfile
from pathlib import Path
import torch
from exemplar import register_exemplars,pick,clear_exemplars,bank_sizes,BANK
for device in ['cpu','cuda']:
 clear_exemplars();x=torch.nn.functional.normalize(torch.arange(1,21*512+1,dtype=torch.float32).reshape(21,512),dim=1)
 y=torch.tensor([0]*20+[1]);cache=dict(ids=list(range(21)),x=x)
 with tempfile.TemporaryDirectory() as d:
  register_exemplars(cache,y,Path(d));assert bank_sizes()=={'0':8,'1':1}
  indices=(torch.arange(8)*19//7).tolist();assert torch.equal(BANK[0],x[indices]);assert torch.equal(BANK[1],x[20:])
  assert max(BANK)==1
  cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
  for j in range(19):
   assert torch.equal(pick(0,j,device,torch.float32).cpu(),x[indices[j%8]:indices[j%8]+1])
   assert torch.equal(pick(1,j,device,torch.float32).cpu(),x[20:])
  assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
  assert sum(v.numel()*v.element_size() for v in BANK.values())==9*512*4
clear_exemplars();print('REAL_MEMORY_MEMBERSHIP_CYCLE_RARE_RNG_PASS')
