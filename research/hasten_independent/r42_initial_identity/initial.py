"""One fixed identity initialization; retain native random draws and later tasks."""
import torch

def install_initial(inc):
 original=inc.hyper_net.update_task
 def update(self):
  first=len(self.Image_Adapter)==0
  original(self)
  if first:
   before=torch.get_rng_state().clone();gpu=[v.clone() for v in torch.cuda.get_rng_state_all()]
   count=sum(p.numel() for p in self.parameters())
   with torch.no_grad():
    for adapter in [self.Image_Adapter[0],self.Text_Adapter[0]]:
     fc=adapter.fc[0];assert len(adapter.fc)==1 and isinstance(fc,torch.nn.Linear) and fc.weight.shape==(512,512)
     fc.weight.copy_(torch.eye(512,dtype=fc.weight.dtype,device=fc.weight.device));fc.bias.zero_()
     assert torch.equal(fc.weight,torch.eye(512,dtype=fc.weight.dtype,device=fc.weight.device)) and not torch.count_nonzero(fc.bias)
   assert count==sum(p.numel() for p in self.parameters())
   assert torch.equal(before,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
   self.initial_identity_applied=True
 inc.hyper_net.update_task=update
