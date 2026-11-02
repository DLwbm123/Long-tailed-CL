import os,json,time,copy,sys,subprocess,csv,traceback
from pathlib import Path
import torch
from PIL import Image
from adapters import Prompts,insert_text,image,text
from common import write,save,guard
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);cp=time.process_time();wt=time.time();torch.set_num_threads(4)
counts={'engineering_image_rows':0,'engineering_text_sequences':0,'distinct_TRAIN_rows':0,'formal_optimizer':0,'engineering_optimizer':0,'VAL_TEST':0}

def software():
 torch.manual_seed(74002);word=torch.randn(2,8,5);mask=torch.tensor([[1,1,1,0,0,0,0,0],[1,1,1,1,0,0,0,0]]);p=torch.randn(2,5);x,m=insert_text(word,mask,p);assert x.shape==word.shape and m.shape==mask.shape and torch.equal(x[:,0],word[:,0]);assert torch.equal(x[:,1:3],p[None].expand(2,-1,-1));assert bool((m[:,1:3]==1).all());n,m0=insert_text(word,mask,None);assert torch.equal(n,word) and torch.equal(m0,mask)
 try:insert_text(word,torch.ones_like(mask),p)
 except ValueError:pass
 else:raise AssertionError('nonpadding truncation accepted')
 write(root/'SOFTWARE.json',{'passed':True,'mask_shape_CLS_preserved':True,'nonpadding_truncation_rejected':True,'None_is_true_NOOP':True,'model_forward':0,'optimizer':0,'VAL_TEST':0})

def run():
 software();guard(cfg)
 import open_clip
 from open_clip.factory import _MODEL_CONFIGS
 spec=json.loads((Path(cfg['biomed_assets'])/'open_clip_config.json').read_text());mc=copy.deepcopy(spec['model_cfg']);mc['text_cfg'].update(hf_model_name=str(Path(cfg['biomed_assets'])/'text'),hf_tokenizer_name=cfg['biomed_assets'],hf_model_pretrained=False);name='fixed_biomedical_compat';_MODEL_CONFIGS[name]=mc
 model,_,tf=open_clip.create_model_and_transforms(name,pretrained=None,pretrained_hf=False,precision='fp32',device='cpu',**{f'image_{k}':v for k,v in spec['preprocess_cfg'].items()});state=torch.load(cfg['biomed_checkpoint'],map_location='cpu',weights_only=False);state=state.get('state_dict',state);key='text.transformer.embeddings.position_ids'
 if key in state and key not in model.state_dict():assert torch.equal(state[key],model.text.transformer.embeddings.position_ids.cpu());del state[key]
 model.load_state_dict(state,strict=True);del state;model.eval().requires_grad_(False).cuda();tokenizer=open_clip.get_tokenizer(name);texts=['a histopathology image','a medical image of tissue'];tokens=tokenizer(texts).cuda();assert tokens.shape==(2,256);pad=model.text.config.pad_token_id;assert pad is not None and pad==tokenizer.tokenizer.pad_token_id;assert bool((tokens[:,0]==tokenizer.tokenizer.cls_token_id).all());assert bool((tokens[:,-4:]==pad).all())
 with open(Path(cfg['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
 roles=json.loads((Path(cfg['source_root'])/'r90_evidence_prompt_program/ROLES.private.json').read_text());chosen=[];components=set()
 for i in sorted(roles['S']):
  if int(rows[i]['original_label']) not in cfg['order'][:13]:continue
  if rows[i]['identity_component'] in components:continue
  chosen.append(i);components.add(rows[i]['identity_component'])
  if len(chosen)==2:break
 assert len(chosen)==2;xs=[]
 for i in chosen:
  with Image.open(Path(cfg['image_roots']['HK'])/rows[i]['relative_path']) as im:xs.append(tf(im.convert('RGB')))
 images=torch.stack(xs).cuda();counts['distinct_TRAIN_rows']=2
 with torch.no_grad():
  vi=model.encode_image(images,normalize=False);counts['engineering_image_rows']+=2
  vn=image(model,images,None);counts['engineering_image_rows']+=2
  tt=model.encode_text(tokens,normalize=False);counts['engineering_text_sequences']+=2
  tn=text(model,tokens,None);counts['engineering_text_sequences']+=2
 ve=float((vi-vn).abs().max());te=float((tt-tn).abs().max());assert ve<=2e-5 and te<=2e-5
 torch.manual_seed(74002);prompts=Prompts().cuda();save(root/'initial_prompts.private.pt',prompts.state_dict());before={k:v.detach().clone() for k,v in prompts.state_dict().items()};vp=image(model,images,prompts.visual);counts['engineering_image_rows']+=2;tp=text(model,tokens,prompts.text);counts['engineering_text_sequences']+=2
 loss=-(torch.nn.functional.normalize(vp,dim=1)*torch.nn.functional.normalize(tp,dim=1)).sum(1).mean();assert torch.isfinite(loss);loss.backward();grads={k:float(p.grad.norm()) for k,p in prompts.named_parameters()};assert all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in prompts.parameters());assert all(p.grad is None for p in model.parameters());assert all(torch.equal(v,before[k]) for k,v in prompts.state_dict().items());other=Prompts().cuda();other.load_state_dict(before,strict=True);assert all(torch.equal(v,other.state_dict()[k]) for k,v in prompts.state_dict().items())
 with torch.no_grad():
  vz=image(model,images,torch.zeros_like(prompts.visual));counts['engineering_image_rows']+=2;tz=text(model,tokens,torch.zeros_like(prompts.text));counts['engineering_text_sequences']+=2
 own=[l for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines() if l.split(',')[0].strip()==str(os.getpid())];assert len(own)==1 and own[0].split(',')[1].strip()==cfg['GPU_UUID'];assert os.environ['CUDA_VISIBLE_DEVICES']=='1';reserved=torch.cuda.max_memory_reserved();assert reserved<12*2**30;torch.cuda.synchronize()
 rec={'passed':True,'vision_NOOP_maxabs':ve,'text_NOOP_maxabs':te,'token_context':tokens.shape[1],'bert_max_positions':model.text.config.max_position_embeddings,'pad_token_id':pad,'CLS_token_id':tokenizer.tokenizer.cls_token_id,'SEP_token_id':tokenizer.tokenizer.sep_token_id,'gradient_norms':grads,'all_base_frozen_no_grad':True,'strict_prompt_restore':True,'prompt_parameters':sum(p.numel() for p in prompts.parameters()),'text_prompt_parameters_delta_vs_R88_per_branch':1024,'zero_tokens_vision_maxabs_vs_NOOP':float((vz-vn).abs().max()),'zero_tokens_text_maxabs_vs_NOOP':float((tz-tn).abs().max()),'physical_GPU':1,'own_PID_GPU_binding':True,'GPU_reserved_bytes':reserved,'counts':counts,'CPU_seconds':time.process_time()-cp,'wall_seconds':time.time()-wt,'scope':'Interface and gradient compatibility only, no accuracy/efficiency/complete RLAP/GRPO efficacy claim.'}
 write(root/'NATIVE.json',rec);write(root/'STATUS.json',{'phase':'COMPLETE_ENGINEERING_ONLY','passed':True,'counts':counts})
try:run()
except BaseException:
 write(root/'FAILURE_COUNTS.private.json',counts);(root/'FAILURE.private.txt').write_text(traceback.format_exc());raise
