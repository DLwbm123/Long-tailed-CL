"""Source-only original-pixel frozen-prefix exporter, no optimizer or future-stage access."""
import os,json,time,copy,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from common import write,guard
from vision import frozen_prefix,NativeTail,adapt_native_tail,delta_state,load_delta
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
 import open_clip
 from open_clip.factory import _MODEL_CONFIGS
 guard(cfg);job=cfg['job'];assets=Path(cfg['biomed_assets']);spec=json.loads((assets/'open_clip_config.json').read_text());mc=copy.deepcopy(spec['model_cfg']);mc['text_cfg'].update(hf_model_name=str(assets/'text'),hf_tokenizer_name=str(assets),hf_model_pretrained=False);_MODEL_CONFIGS['fixed_biomedical']=mc
 model,_,preprocess=open_clip.create_model_and_transforms('fixed_biomedical',pretrained=None,pretrained_hf=False,precision='fp32',device='cuda',**{f'image_{k}':v for k,v in spec['preprocess_cfg'].items()});s=torch.load(cfg['biomed_checkpoint'],map_location='cpu',weights_only=False);s=s.get('state_dict',s);key='text.transformer.embeddings.position_ids'
 if key in s:assert torch.equal(s[key],model.text.transformer.embeddings.position_ids.cpu());del s[key]
 model.load_state_dict(s,strict=True);del s;model.eval().requires_grad_(False);visual=model.visual;assert len(visual.trunk.blocks)==12 and visual.trunk.global_pool=='token';assert isinstance(visual.trunk.fc_norm,nn.Identity);assert list(visual.head._modules)==['drop','proj'] and visual.head.drop.p==0 and visual.head.proj.bias is None
 reference=torch.load(root/'reference.private.pt',map_location='cpu',weights_only=False);assert reference['ids']==[r['id'] for r in job['rows']];expected=reference['x'];assert expected.shape==(len(job['rows']),512)
 if job['split']=='val':assert job['TRAIN_sealed_all3seeds']
 export=root/'export';export.mkdir(exist_ok=True);meta=dict(id=job['id'],dataset=job['dataset'],split=job['split'],stage=job.get('stage'),ids=[r['id'] for r in job['rows']],labels=[r['label'] for r in job['rows']],dtype='float32',token_shape=[197,768],chunks=[],actual_encoded_rows=0,reference_max_abs=0.,prefix_tail_max_abs=None)
 if job['role']=='preflight':
  state={}
  for i,block in enumerate(list(visual.trunk.blocks)[10:]):
   for k,v in block.state_dict().items():state[f'blocks.{i}.{k}']=v.cpu()
  state.update({'norm.'+k:v.cpu() for k,v in visual.trunk.norm.state_dict().items()});state['proj.weight']=visual.head.proj.weight.cpu()
  bundle=dict(state=state,meta=dict(global_pool='token',fc_norm_identity=True,split=10,depth=2,width=768,tokens=197,output_dim=512,norm_eps=visual.trunk.norm.eps,block_norm_eps=[[b.norm1.eps,b.norm2.eps] for b in list(visual.trunk.blocks)[10:]],torch_version=torch.__version__))
  torch.save(bundle,export/'tail.private.pt');tail=NativeTail(bundle).cuda().eval()
 def read_image(r):
  rel=Path(r['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
  with Image.open(Path(cfg['image_roots'][job['dataset']])/rel) as im:return preprocess(im.convert('RGB'))
 pool=ThreadPoolExecutor(max_workers=4)
 for start in range(0,len(job['rows']),64):
  guard(cfg);rows=job['rows'][start:start+64];pixels=[]
  pixels=torch.stack(list(pool.map(read_image,rows))).cuda()
  with torch.no_grad():
   tokens=frozen_prefix(visual,pixels);h=tokens
   for block in list(visual.trunk.blocks)[10:]:h=block(h)
   base=F.normalize(visual.head(visual.trunk.forward_head(visual.trunk.norm(h))),dim=-1)
   if job['role']=='preflight':assert torch.allclose(base,F.normalize(model.encode_image(pixels),dim=-1),atol=2e-5,rtol=0)
   assert tokens.shape==(len(rows),197,768) and tokens.dtype==torch.float32 and torch.isfinite(tokens).all();err=float((base.cpu()-expected[start:start+len(rows)]).abs().max());assert err<=2e-5,('INPUT_REFERENCE_PARITY',err);meta['reference_max_abs']=max(meta['reference_max_abs'],err)
  if job['role']=='preflight':
   with torch.no_grad():out=tail(tokens);err2=float((out-base).abs().max());assert err2<=2e-5;meta['prefix_tail_max_abs']=err2
   params=adapt_native_tail(tail);tail.eval();z=tail(tokens);assert torch.allclose(z,base,atol=2e-5,rtol=0);F.cross_entropy(20*z@reference['text'].cuda().T,torch.tensor([r['label'] for r in rows],device='cuda')).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params) and max(float(p.grad.norm()) for p in params)>0
  k=len(meta['chunks']);name=f'{k:06d}.bin';tmp=export/(name+'.tmp');tokens.cpu().numpy().tofile(tmp);tmp.replace(export/name);np.save(export/f'{k:06d}.base.npy',base.cpu().numpy());meta['chunks'].append(dict(file=name,base=f'{k:06d}.base.npy',start=start,rows=len(rows),bytes=len(rows)*197*768*4));meta['actual_encoded_rows']+=len(rows);write(export/'META.json',meta);write(root/'STATUS.json',dict(phase='ENCODING',job_id=job['id'],encoded=meta['actual_encoded_rows'],total=len(job['rows']),time=time.time()));assert torch.cuda.max_memory_reserved()<12*2**30
 pool.shutdown()
 assert meta['actual_encoded_rows']==len(job['rows']);write(export/'SOURCE_READY.json',dict(rows=len(job['rows']),source_input_parity=True,no_optimizer=True,role=job['role']));write(root/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:(root/'FAILURE.private.txt').write_text(traceback.format_exc());raise
