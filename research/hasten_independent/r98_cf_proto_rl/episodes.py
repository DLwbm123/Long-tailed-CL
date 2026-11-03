"""Arm-independent support subsets and paired purpose-separated noise keys."""
import torch
from prototype_policy import build_state

def seed(stage,step,episode,purpose):return 74002+100000003*stage+100003*step+1009*episode+7919*purpose

def episode(d,stage,step,half=False):
 gen=torch.Generator().manual_seed(seed(stage,step,1,1));supports=[];ids=[]
 for c in range(d['C']):
  pool=d['Sids'][c];take=pool if not half else pool[torch.randperm(len(pool),generator=gen)[:max(1,len(pool)//2)]];assert len(take)==len(take.unique());ids+=take.tolist();supports.append(d['S'][torch.searchsorted(d['Sall'],take.to(d['Sall'].device))])
 assert not set(ids)&d['Bids_set'];ep=build_state(supports,d['rawcounts'],d['births'],stage);ep.update({'query':d['B'],'y':d['By'],'nB':d['nB'],'protected':d['protected'],'support_ids':ids,'stage':stage});return ep

def noise(ep,stage,step,which,purpose=2):
 g=torch.Generator(device=ep['state'].device).manual_seed(seed(stage,step,which,purpose));return torch.randn(8,len(ep['support']),4,generator=g,device=ep['state'].device)
