"""No new parameters: class-known text prompts use adapters present at birth."""
import torch
def birth_text_encode(x,class_ids,adapters,init_cls,increment):
 assert x.ndim==2 and class_ids.ndim==1 and len(x)==len(class_ids) and class_ids.dtype==torch.long
 assert len(x)>0 and int(class_ids.min())>=0 and increment>0
 birth=torch.where(class_ids<init_cls,0,1+torch.div(class_ids-init_cls,increment,rounding_mode='floor'))
 assert int(birth.max())<len(adapters),'FUTURE_OR_UNARRIVED_TEXT_CLASS'
 out=torch.zeros_like(x)
 for task in sorted(set(birth.tolist())):
  ids=torch.nonzero(birth==task,as_tuple=True)[0];group=x[ids]
  mapped=torch.stack([adapters[j](group) for j in range(task+1)]).sum(0)
  out=out.index_copy(0,ids,mapped)
 return out
