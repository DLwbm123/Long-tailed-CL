"""Matched frozen-encoder/readout diagnostic. No validation until all four heads seal."""
import os,json,time,ast,copy,math
from pathlib import Path
import torch
from heads import add_class,solve,check
CFG=json.loads(Path(os.environ['RUN_CONFIG']).read_text());OUT=Path(CFG['output'])
torch.set_num_threads(4);torch.set_num_interop_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def write(path,obj):
 p=OUT/path;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False));q.replace(p)
def status(**kw):write('STATUS.json',dict(time=time.time(),**kw))
def finite(x):assert torch.isfinite(x).all()
def selected(source,names,ns):
 tree=ast.parse(Path(source).read_text());tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in tree.body}==set(names)
 exec(compile(tree,'<pinned-reader-encoder>','exec'),ns);return ns

class ImageNetEncoder:
 def __init__(self,rows,spec,ds):
  import timm
  from safetensors.torch import load_file
  self.rows=rows;self.spec=spec;self.ds=ds
  self.model=timm.create_model(CFG['imagenet_model'],pretrained=False)
  self.model.load_state_dict(load_file(CFG['imagenet_checkpoint']),strict=True)
  self.preprocess=timm.data.create_transform(**timm.data.resolve_model_data_config(self.model),is_training=False)
  self.model.reset_classifier(0);assert self.model.global_pool=='token'
  self.model.eval().requires_grad_(False).cuda()
  write(ds+'/IMAGENET_RECEIPT.json',dict(model=CFG['imagenet_model'],revision=CFG['imagenet_revision'],strict_load=True,dimension=768,timm_version=timm.__version__,preprocess=timm.data.resolve_model_data_config(self.model),frozen=True,global_pool='token',adapter=False))
 @torch.no_grad()
 def encode(self,ids,phase):
  from PIL import Image
  pieces=[]
  for j in range(0,len(ids),32):
   batch=[]
   for i in ids[j:j+32]:
    with Image.open(Path(self.spec['image_root'])/self.rows[i]['relative_path']) as im:batch.append(self.preprocess(im.convert('RGB')))
   x=self.model(torch.stack(batch).cuda()).float().cpu();finite(x);assert x.shape[1]==768;pieces.append(x)
   assert torch.cuda.max_memory_reserved()<8*2**30
   if j%1024==0:status(dataset=self.ds,encoder='ImageNet',phase=phase,encoded=j+len(batch),total=len(ids))
  return torch.cat(pieces)
 def release(self):self.model=None;torch.cuda.empty_cache()

def score(x,y,head,tail):
 z=torch.cat([x.double(),torch.ones(len(x),1,dtype=torch.double)],1)
 logits={'NCM':x.double()@head['prototype'].T,'ridge':z@head['ridge']};out={}
 for name,s in logits.items():
  finite(s);pred=s.argmax(1);counts=torch.bincount(y,minlength=head['classes']);correct=torch.bincount(y[pred==y],minlength=head['classes']);rec=correct.double()/counts
  assert (counts>0).all();out[name]=dict(BA=float(rec.mean()*100),tail_BA=float(rec[[c for c in tail if c<head['classes']]].mean()*100),counts=counts.tolist(),correct=correct.tolist())
 return out

write('HEAD_ACCEPTANCE.json',check());all_results={}
for ds in ['NIH','ISIC','HK']:
 status(dataset=ds,phase='READ_TRAIN_METADATA')
 spec=json.loads(Path(CFG['configs'][ds]).read_text());seal=json.loads((Path(spec['output'])/'SPLIT_SEAL.json').read_text());order=seal['original_order'];stages=seal['stages']
 ns=dict(CFG=spec,ORDER=order,Path=Path,torch=torch)
 rows,tr,te=selected(spec['base_runner'],['read_rows'],ns)['read_rows']()
 y=torch.tensor([r['label'] for r in rows]);assert torch.bincount(y[tr]).tolist()==seal['train_counts'];assert torch.bincount(y[te]).tolist()==seal['eval_counts']
 tail=[i for i,n in enumerate(seal['train_counts']) if n<250]
 write(ds+'/SPLIT_SEAL.json',seal)
 # Reuse the exact frozen Biomed encoder implementation without executing its experiment.
 namespace=dict(CFG=spec,Path=Path,torch=torch,json=json,copy=copy,time=time,os=os,finite=finite,status=lambda **kw:status(dataset=ds,encoder='Biomed',**kw),write=lambda p,v:write(ds+'/'+p,v))
 bio=selected(CFG['biomed_encoder_source'],['EncoderCache'],namespace)['EncoderCache'](rows)
 assert Path(spec['biomed_checkpoint']).exists()
 train_heads={}
 for enc in ['Biomed','ImageNet']:
  engine=bio if enc=='Biomed' else ImageNetEncoder(rows,spec,ds)
  gram=None;means=[];heads=[];provenance=[]
  for stage,classes in enumerate(stages):
   ids=[i for i in tr if rows[i]['label'] in classes]
   if enc=='Biomed':
    cache=Path(spec['cache_root'] if ds=='NIH' else spec['shared_cache'])/f'stage_{stage}.private.pt'
    rec=torch.load(cache,map_location='cpu',weights_only=False);assert rec['ids']==ids
    x=torch.nn.functional.normalize(rec['x'],dim=1);finite(x)
    if stage==0:
     # A small current-source probe checks cached row mapping and encoder provenance.
     m=min(32,len(ids));fresh=torch.nn.functional.normalize(engine.encode(ids[:m],'CACHE_PROVENANCE'),dim=1)
     err=float((fresh-x[:m]).abs().max());assert err<2e-5,('cache encoder mismatch',err)
     write(ds+'/CACHE_PROVENANCE.json',dict(max_abs_error=err,rows_checked=m,all_stage_ids_checked=True,shared_read_only=True))
    del rec
   else:
    status(dataset=ds,encoder=enc,phase='ENCODE_TRAIN',stage=stage)
    x=torch.nn.functional.normalize(engine.encode(ids,'ENCODE_TRAIN'),dim=1)
    target=OUT/ds/'cache'/f'imagenet_stage_{stage}.private.pt';target.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(x=x,ids=ids),target)
   assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6)
   yy=y[ids];assert torch.bincount(yy,minlength=len(order))[classes].tolist()==[seal['train_counts'][c] for c in classes]
   for c in classes:
    assert c==len(means);g,m=add_class(x[yy==c]);gram=g if gram is None else gram+g;means.append(m)
   head=solve(gram,means);target=OUT/ds/'heads'/f'{enc}_{stage}.private.pt';target.parent.mkdir(parents=True,exist_ok=True);torch.save(head,target)
   restored=torch.load(target,map_location='cpu',weights_only=False)
   assert torch.equal(head['ridge'],restored['ridge']) and torch.equal(head['prototype'],restored['prototype']);heads.append(restored)
   provenance.append(dict(stage=stage,rows=len(ids),classes=classes,dimension=x.shape[1]))
  write(ds+'/'+enc+'_TRAIN_RECEIPT.json',dict(stages=provenance,validation_image_reads=0,all_heads_restore_exact=True))
  train_heads[enc]=heads;engine.release();del engine
 write(ds+'/ALL_HEADS_SEALED.json',dict(time=time.time(),encoders=['Biomed','ImageNet'],readouts=['NCM','ridge'],stages=len(stages),validation_image_reads=0))
 results=[]
 for enc in ['Biomed','ImageNet']:
  engine=bio if enc=='Biomed' else ImageNetEncoder(rows,spec,ds)
  x=torch.nn.functional.normalize(engine.encode(te,'ENCODE_VAL_AFTER_SEAL'),dim=1);yv=y[te]
  for stage,head in enumerate(train_heads[enc]):
   keep=yv<head['classes'];metrics=score(x[keep],yv[keep],head,tail)
   for m in metrics.values():assert m['counts']==seal['eval_counts'][:head['classes']]
   results.append(dict(encoder=enc,stage=stage,seen=head['classes'],lambda_value=head['lambda_value'],metrics=metrics))
  engine.release();del engine
 all_results[ds]=dict(results=results,split=seal)
 write(ds+'/RESULTS.json',all_results[ds]);write(ds+'/STATUS.json',dict(phase='COMPLETE',time=time.time()))
write('RESULTS.json',all_results);status(phase='COMPLETE')
