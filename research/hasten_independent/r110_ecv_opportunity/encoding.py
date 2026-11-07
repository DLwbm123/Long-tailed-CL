import math
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from .base.real_integration import load_generic,BOXES
from .common import guard

class Encoder:
    def __init__(self,cfg):
        self.visual,self.transform=load_generic(cfg)
        self.visual=self.visual.cuda().eval().requires_grad_(False)
    @torch.no_grad()
    def encode(self,cfg,rows,ids,counter,kind,allowed):
        assert set(ids)<=set(allowed),'FUTURE_OR_DISCARDED_IMAGE_ACCESS'
        out=[]
        for start in range(0,len(ids),16):
            guard(cfg);group=ids[start:start+16];images=[]
            for i in group:
                with Image.open(Path(cfg['image_root'])/rows[i]['relative_path']) as image:images.append(image.convert('RGB'))
            views=[]
            for l,t,r,b in BOXES:
                x=[]
                for image in images:
                    w,h=image.size;x.append(self.transform(image.crop((math.floor(w*l),math.floor(h*t),math.ceil(w*r),math.ceil(h*b)))))
                views.append(F.normalize(self.visual(torch.stack(x).cuda()).float(),dim=-1))
                counter[kind+'_encoder_rows']=counter.get(kind+'_encoder_rows',0)+len(group)
            out.append(torch.stack(views,1))
        return torch.cat(out)

