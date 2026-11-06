"""Only original generic frozen BiomedCLIP, current-stage TRAIN pixels or sealed VAL."""
import copy,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from common import guard
BOXES=[(0,0,1,1),(.125,.125,.875,.875),(0,0,.6,.6),(.4,0,1,.6),(0,.4,.6,1),(.4,.4,1,1)]
class Encoder:
    def __init__(self,c):
        import open_clip
        from open_clip.factory import _MODEL_CONFIGS
        spec=json.loads((Path(c['biomed_assets'])/'open_clip_config.json').read_text());mc=copy.deepcopy(spec['model_cfg']);mc['text_cfg'].update(hf_model_name=str(Path(c['biomed_assets'])/'text'),hf_tokenizer_name=c['biomed_assets'],hf_model_pretrained=False);_MODEL_CONFIGS['generic_branch']=mc
        base,_,tf=open_clip.create_model_and_transforms('generic_branch',pretrained=None,pretrained_hf=False,precision='fp32',device='cpu',**{f'image_{k}':v for k,v in spec['preprocess_cfg'].items()});state=torch.load(c['biomed_checkpoint'],map_location='cpu',weights_only=False);state=state.get('state_dict',state);key='text.transformer.embeddings.position_ids'
        if key in state and key not in base.state_dict():assert torch.equal(state[key],base.text.transformer.embeddings.position_ids);del state[key]
        base.load_state_dict(state,strict=True);self.visual=base.visual.eval().requires_grad_(False).cuda();self.tf=tf;del base,state
        assert not any('Random' in type(t).__name__ for t in tf.transforms)
    @torch.no_grad()
    def encode(self,c,rows,ids,counter,split,allowed=None):
        assert split in ['native','TRAIN','VAL']
        if split!='VAL':assert set(ids)<=set(allowed),'ILLEGAL_TRAIN_HISTORY_OR_FUTURE_ACCESS'
        else:assert (Path(c['root_output'])/'GLOBAL_LEARNING_SEAL.json').exists()
        out=[]
        for lo in range(0,len(ids),16):
            guard(c);ix=ids[lo:lo+16];im=[]
            for i in ix:
                with Image.open(Path(c['image_root'])/rows[i]['relative_path']) as src:im.append(src.convert('RGB'))
            vv=[]
            for box in BOXES:
                pix=[]
                for src in im:
                    w,h=src.size;l,t,r,b=box;pix.append(self.tf(src.crop((math.floor(w*l),math.floor(h*t),math.ceil(w*r),math.ceil(h*b)))))
                z=F.normalize(self.visual(torch.stack(pix).cuda()).float(),dim=-1);assert torch.isfinite(z).all();vv.append(z);counter[split+'_encoder_rows']=counter.get(split+'_encoder_rows',0)+len(ix)
            out.append(torch.stack(vv,1))
        return torch.cat(out)
