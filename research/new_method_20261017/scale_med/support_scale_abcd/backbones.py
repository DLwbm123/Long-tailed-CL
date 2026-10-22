"""Reuse the qualified frozen-backbone loaders; no query access."""
import json
import torch
from torch.nn import functional as F
from core import NAMES,PARENTS
def get_backbone(kind,cfg,out,write,status,counts):
 if kind == "CLIP":
  import open_clip
  status(phase='LOADING_BACKBONE')
  backbone,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',precision='fp32',device='cuda',cache_dir=cfg['clip_cache'])
  backbone.eval().requires_grad_(False);versions={n:p._version for n,p in backbone.named_parameters()}
  tokenizer=open_clip.get_tokenizer('ViT-B-32')
  prompts=['a photo of a '+n+'.' for n in NAMES]+['an endoscopic image of '+n+'.' for n in PARENTS]
  with torch.no_grad():text=F.normalize(backbone.encode_text(tokenizer(prompts).cuda()).float(),dim=1)
  assert text.shape==(20,512) and torch.isfinite(text).all();counts['text_encoder_rows']+=20;counts['text_encoder_calls']+=1
  write('TEXT_SEAL.json',dict(prompts=prompts,rows=20,normalized=True))
 else:
  import open_clip
  status(phase='LOADING_BACKBONE')
  from assets import prepare
  assets,revision,checkpoint=prepare(out/'assets')
  from open_clip.factory import _MODEL_CONFIGS
  spec=json.loads((assets/'open_clip_config.json').read_text());mc=spec['model_cfg']
  mc['text_cfg']['hf_model_name']=str(assets/'text');mc['text_cfg']['hf_tokenizer_name']=str(assets)
  mc['text_cfg']['hf_model_pretrained']=False
  _MODEL_CONFIGS['fixed_biomedical']=mc
  backbone,_,preprocess=open_clip.create_model_and_transforms('fixed_biomedical',pretrained=None,pretrained_hf=False,precision='fp32',device='cuda',**{f'image_{k}':v for k,v in spec['preprocess_cfg'].items()})
  state=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if 'state_dict' in state:state=state['state_dict']
  key='text.transformer.embeddings.position_ids'
  if key in state:
   assert torch.equal(state[key],backbone.text.transformer.embeddings.position_ids.cpu())
   del state[key]  # OpenCLIP factory compatibility for transformers>=4.31; no learned weight removed.
  backbone.load_state_dict(state,strict=True);del state;checkpoint.close();del checkpoint
  write('BACKBONE_RECEIPT.json',dict(model='BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',revision=revision,embedding_dimension=512,strict_checkpoint_loading=True,preprocess=spec['preprocess_cfg'],open_clip_version=open_clip.__version__,no_separate_text_weights=True))
  backbone.eval().requires_grad_(False);versions={n:p._version for n,p in backbone.named_parameters()}
  tokenizer=open_clip.get_tokenizer('fixed_biomedical',context_length=256)
  prompts=['a photo of a '+n+'.' for n in NAMES]+['an endoscopic image of '+n+'.' for n in PARENTS]
  with torch.no_grad():text=F.normalize(backbone.encode_text(tokenizer(prompts).cuda()).float(),dim=1)
  assert text.shape==(20,512) and torch.isfinite(text).all();counts['text_encoder_rows']+=20;counts['text_encoder_calls']+=1
  write('TEXT_SEAL.json',dict(prompts=prompts,rows=20,normalized=True))
 return backbone,preprocess,text,versions
