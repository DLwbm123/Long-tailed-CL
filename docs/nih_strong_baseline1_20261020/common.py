"""Matched static NIH training baseline; no test path or download code."""
import csv,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def write(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rows(cfg):
 splits={};counts={};patients={}
 for split in ['train','val']:
  with (Path(cfg['author_labels'])/(split+'.csv')).open(newline='') as f:
   reader=csv.DictReader(f);assert reader.fieldnames==['id',*cfg['labels'],'subject_id'];r=list(reader)
  result=[];seen=set();pat=set();cc=np.zeros(20,dtype=int)
  for z in r:
   name=z['id'];assert Path(name).name==name and name.endswith('.png') and name not in seen;seen.add(name);values=[int(z[c]) for c in cfg['labels']];assert sum(values)==1 and set(values)<={0,1};label=values.index(1);assert int(name.split('_')[0])==int(z['subject_id']);pat.add(z['subject_id']);cc[label]+=1;result.append((str(Path(cfg['images'])/'images'/name),label,z['subject_id']))
  splits[split]=result;counts[split]=cc;patients[split]=pat
 assert len(splits['train'])==68058 and len(splits['val'])==300 and np.all(counts['val']==15);assert not patients['train']&patients['val'];assert not {z[0] for z in splits['train']}&{z[0] for z in splits['val']};assert counts['train'].tolist()==cfg['train_counts']
 return splits,dict(train_images=68058,val_images=300,train_patients=len(patients['train']),val_patients=len(patients['val']),train_counts=counts['train'].tolist(),val_counts=counts['val'].tolist(),patient_overlap=0,image_overlap=0,test_access=False)


class Images(torch.utils.data.Dataset):
 def __init__(self,items,backbone,train):
  self.items=items
  mean,std=([.5]*3,[.5]*3) if backbone=='imagenet' else ([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])
  geo=[T.RandomResizedCrop(224,scale=(.8,1.),ratio=(.9,1.1),interpolation=InterpolationMode.BICUBIC),T.RandomHorizontalFlip(.5)] if train else [T.Resize(256,interpolation=InterpolationMode.BICUBIC),T.CenterCrop(224)]
  self.transform=T.Compose([*geo,T.ToTensor(),T.Normalize(mean,std)])
 def __len__(self):return len(self.items)
 def __getitem__(self,i):
  path,label,_=self.items[i]
  with Image.open(path) as im:x=self.transform(im.convert('RGB'))
  return x,label,i


def worker_seed(_):
 torch.set_num_threads(1);seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)


class Model(nn.Module):
 def __init__(self,cfg,backbone):
  super().__init__();from safetensors.torch import load_file
  if backbone=='imagenet':
   import timm
   self.visual=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);state=load_file(cfg['imagenet_weight']);assert set(['head.weight','head.bias'])<=set(state);state={k:v for k,v in state.items() if not k.startswith('head.')};self.visual.load_state_dict(state,strict=True);initial=torch.randn(20,768)*.02
  else:
   import clip
   original=clip.model.build_model(load_file(cfg['clip_weight'])).float().eval()
   with torch.no_grad():
    tokens=clip.tokenize(['a photo of a '+c.replace('_',' ')+'.' for c in cfg['labels']]);text=F.normalize(original.encode_text(tokens),dim=-1);initial=F.normalize(text@original.visual.proj.T,dim=-1)
   self.visual=original.visual;self.visual.proj=None;del original
  self.weight=nn.Parameter(initial.clone());self.scale=cfg['classifier_scale']
 def forward(self,x):
  z=self.visual(x);return self.scale*(F.normalize(z.float(),dim=-1)@F.normalize(self.weight.float(),dim=-1).T)


def metrics(score,y,groups,counts,legacy_tail):
 pred=score.argmax(1);correct=pred==y;rec=[float(correct[y==c].mean()) for c in range(20)];head=np.flatnonzero(np.asarray(counts)>1000);mid=np.flatnonzero((np.asarray(counts)>=100)&(np.asarray(counts)<=1000));tail=np.flatnonzero(np.asarray(counts)<100);w=np.empty(len(y))
 for c in range(20):
  ids=np.flatnonzero(y==c);gs,ns=np.unique(groups[ids],return_counts=True);mapping=dict(zip(gs,ns));w[ids]=[1/(20*len(gs)*mapping[g]) for g in groups[ids]]
 return dict(BA=float(np.mean(rec)),accuracy=float(correct.mean()),PP_accuracy=float(w@correct),correct=int(correct.sum()),images=len(y),counts=[int((y==c).sum()) for c in range(20)],correct_by_class=[int(correct[y==c].sum()) for c in range(20)],recall=rec,paper_groups={name:float(np.mean(np.asarray(rec)[cs])) for name,cs in [('head',head),('middle',mid),('tail',tail)]},paper_group_mean=float(np.mean([np.mean(np.asarray(rec)[z]) for z in [head,mid,tail]])),legacy_tail_BA=float(np.mean(np.asarray(rec)[legacy_tail])),confusion=np.bincount(20*y+pred,minlength=400).reshape(20,20).tolist())


def lr_multiplier(step,total):
 import math
 warm=max(100,int(.01*total))
 if step<warm:return 1e-6+(1-1e-6)*(step+1)/warm
 return .1+.9*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))/2
