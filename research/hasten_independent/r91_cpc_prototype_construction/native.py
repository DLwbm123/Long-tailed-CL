"""Frozen original Biomed prefix/tail on physicalGPU1; TRAIN native only before seal."""
import os,json,copy,time,sys,torch,subprocess
from pathlib import Path
from PIL import Image
from core import *
from binding import *
from common import *

def assets(cfg):
 import open_clip
 from open_clip.factory import _MODEL_CONFIGS
 spec=json.loads((Path(cfg['biomed_assets'])/'open_clip_config.json').read_text());mc=copy.deepcopy(spec['model_cfg']);mc['text_cfg'].update(hf_model_name=str(Path(cfg['biomed_assets'])/'text'),hf_tokenizer_name=cfg['biomed_assets'],hf_model_pretrained=False);_MODEL_CONFIGS['fixed_cpc_biomedical']=mc
 model,_,tf=open_clip.create_model_and_transforms('fixed_cpc_biomedical',pretrained=None,pretrained_hf=False,precision='fp32',device='cpu',**{f'image_{k}':v for k,v in spec['preprocess_cfg'].items()});state0=torch.load(cfg['biomed_checkpoint'],map_location='cpu',weights_only=False);state0=state0.get('state_dict',state0);key='text.transformer.embeddings.position_ids'
 if key in state0 and key not in model.state_dict():assert torch.equal(state0[key],model.text.transformer.embeddings.position_ids.cpu());del state0[key]
 model.load_state_dict(state0,strict=True);visual=model.visual.eval().requires_grad_(False).cuda();del model,state0
 source=Path(cfg['source_root'])/'r77_distributional_support';sys.path.insert(0,str(source));from vision import NativeTail,adapt_native_tail,load_delta,frozen_prefix
 bundle=torch.load(source/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);tail=NativeTail(bundle).cuda();return visual,tail,tf,(adapt_native_tail,load_delta,frozen_prefix)

def pixels(cfg,rows,ids,tf):
 root=Path(cfg['image_roots']['HK']);xs=[]
 for i in ids:
  with Image.open(root/rows[i]['relative_path']) as im:xs.append(tf(im.convert('RGB')))
 return torch.stack(xs).cuda()

def run(cfg,counter):
 cpu=time.process_time();wall=time.time();visual,tail,tf,ops=assets(cfg);adapt,load,prefix=ops;d=load_stage(cfg,0);dc=json.loads((Path(cfg['source_root'])/'r77_distributional_support/HK/config.private.json').read_text())
 import csv
 with open(Path(cfg['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
 ids=[next(i for i in d['roles']['W'] if int(d['y'][d['index'][i]])==c) for c in [0,1]];x=pixels(cfg,rows,ids,tf)
 with torch.no_grad():
  tok=prefix(visual,x);base=unit(visual(x));tailbase=tail(tok);assert (base-tailbase).abs().max()<=2e-5
 adapt(tail);load(tail,d['head']['delta']);tail.eval().requires_grad_(False)
 with torch.no_grad():z=tail(tok).cpu();expected=d['z'][[d['index'][i] for i in ids]].float();error=float((z-expected).abs().max());assert error<=2e-5
 st,p0,_=episode(cfg,d,0,0);torch.manual_seed(74002);g=Builder();cand,r=g(st,p0);sc=score(expected.to(DT),cand);original=20*expected@d['head']['prototypes'].T;direct0=20*expected@p0.float().T
 assert torch.equal(sc[:,:,0].float(),direct0);assert torch.isfinite(sc).all()
 fullproto=d['head']['prototypes'].to(DT)[:,None].expand(-1,3,-1);assert torch.equal(score(expected.to(DT),fullproto)[:,:,0].float(),original)
 loss=F.cross_entropy(20*expected.to(DT)@cand[:,2].T,torch.tensor([0,1]));loss.backward();assert torch.isfinite(g.net[-1].weight.grad).all() and g.net[-1].weight.grad.abs().sum()>0
 saved={k:v.detach().clone() for k,v in g.state_dict().items()};other=Builder();other.load_state_dict(saved,strict=True);assert all(torch.equal(v,other.state_dict()[k]) for k,v in saved.items())
 before=time.time()
 with torch.no_grad():
  for _ in range(4):zz=tail(prefix(visual,x))
 torch.cuda.synchronize();encode_wall=time.time()-before
 own=[l for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines() if l.split(',')[0].strip()==str(os.getpid())];assert len(own)==1 and own[0].split(',')[1].strip()==cfg['GPU_UUID'] and os.environ['CUDA_VISIBLE_DEVICES']=='1';assert torch.cuda.max_memory_reserved()<12*2**30
 counter['engineering_real_encoder_rows']+=2*3+8 # visual reference, prefix/tail parity and repeated timing, distinct2
 rec={'passed':True,'real_distinct_TRAIN_rows':2,'SFT_saved_feature_maxabs':error,'NOOP_same_matmul_exact':True,'builder_gradient_finite_nonzero':True,'strict_restore':True,'physical_GPU':1,'own_PID':os.getpid(),'GPU_UUID':cfg['GPU_UUID'],'four_2row_full_frozen_encode_wall':encode_wall,'GPU_reserved_bytes':torch.cuda.max_memory_reserved(),'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'new_optimizer_steps':0,'VAL_TEST':0}
 write(Path(cfg['output'])/'NATIVE.json',rec);del visual,tail;torch.cuda.empty_cache();return rec
