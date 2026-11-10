"""CPU tokenizer/EOS receipts and endpoint-state audit; no model construction."""
import json,re,time
from pathlib import Path
import torch
from transformers import AutoProcessor
from .core import ARMS,write

def run(cfg):
 p=Path(cfg['output']);start=time.time();cpu=time.process_time();tok=AutoProcessor.from_pretrained(cfg['model'],local_files_only=True,min_pixels=224**2,max_pixels=224**2).tokenizer;model_cfg=json.loads((Path(cfg['model'])/'config.json').read_text());eos=model_cfg['eos_token_id'];assert isinstance(eos,int);normal=lambda x:re.sub(r'[\s_\-]+','',x.lower().strip());n=0;finished=0
 for arm in ARMS:
  for k in range(32):
   z=torch.load(p/f'ROLLOUT_{arm}_{k:03d}.private.pt',map_location='cpu',weights_only=True);t=z['t'];m=z['m'];d=z['done'];assert t.shape==m.shape==z['old'].shape==z['ref'].shape and len(t)==64;assert bool((m[:,:-1]>=m[:,1:]).all());active_eos=(t==eos)&m;assert torch.equal(active_eos.any(-1),d) and bool((active_eos.sum(-1)<=1).all())
   for i in range(len(t)):
    length=int(m[i].sum());assert length>0
    if d[i]:assert int(t[i,length-1])==eos
    else:assert length==24
    if length<t.shape[1]:assert bool((t[i,length:]==tok.pad_token_id).all())
   decoded=[normal(x) for x in tok.batch_decode(t.tolist(),skip_special_tokens=True,clean_up_tokenization_spaces=False)];assert decoded==z['text'];n+=len(t);finished+=int(d.sum())
 bindings=json.loads((p/'STATE_BINDINGS.json').read_text());assert len(bindings)==5 and {x['arm'] for x in bindings}==set(ARMS) and all(x['same_prefix_exact'] and x['initial_optimizer_empty'] and x['initial_sigma']==1. and x['anchor_same_initial_prefix'] for x in bindings)
 prefix=torch.load(Path(cfg['prefix_source'])/'COMMON_PREFIX_SFT.private.pt',map_location='cpu',weights_only=True);norms={}
 for a in ARMS:
  s=torch.load(p/f'STATE_1_{a}.private.pt',map_location='cpu',weights_only=True);norms[a]=sum(float((u-x).double().square().sum()+(v-y).double().square().sum()) for (u,v),(x,y) in zip(s['weights'],prefix))**.5;assert norms[a]>0 and s['updates']==32 and {int(v['step']) for v in s['optimizer']['state'].values()}=={32}
 tf=json.loads((p/'TF_COSTS.json').read_text());assert all(tf['qualification:'+k]['rows']==7680 for k in ['semantic_anchor','semantic_actor_eval','semantic_actor_backward'])
 write(p/'TOKEN_STATE_VERIFICATION.json',{'passed':True,'responses_independently_decoded':n,'EOS_finished_responses':finished,'EOS_mask_padding_consistency':True,'all_five_same_prefix_separate_optimizer_initial_state':True,'all_five_terminal_parameter_L2_from_prefix':norms,'all_five_terminal_Adam_step':32,'semantic_science_rows_each_kind':7680,'new_model_generation_encoder_optimizer_calls':0,'CPU_body_seconds':time.process_time()-cpu,'wall_body_seconds':time.time()-start,'tokenizer_processor_metadata_load_only_not_model_weights_or_image_encoding':True})
