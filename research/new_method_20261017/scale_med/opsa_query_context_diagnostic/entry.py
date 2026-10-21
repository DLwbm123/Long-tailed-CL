"""Post-seal original-F query batch-context diagnostic; no training or learned-arm scoring."""
import json, os, sys, time, resource, traceback
from pathlib import Path
import torch
from PIL import Image, features
from torch.nn import functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
torch.set_num_threads(4)
def write(n,v):
 p=out/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def delta(a,b):
 d=(a-b).double();return dict(maxabs=float(d.abs().max()),RMS=float(d.square().mean().sqrt()))
def main():
 assert delta(torch.zeros(2),torch.ones(2))==dict(maxabs=1.,RMS=1.)
 sig=dict(Pillow=Image.__version__,jpg_abi=features.version('jpg'),libjpeg_turbo=features.version('libjpeg_turbo'))
 write('ENVIRONMENT.json',dict(decoder=sig,torch=torch.__version__,cuda=torch.version.cuda))
 assert sig==dict(Pillow='10.4.0',jpg_abi='6.2',libjpeg_turbo='3.0.3')
 prod=Path(cfg['producer_root']);sys.path.insert(0,str(prod))
 from train_base import setup,raw
 from geometry import transform
 plan=torch.load(prod/'prep/PLAN.private.pt',map_location='cpu',weights_only=False)
 seal=json.loads(Path(cfg['model_seal']).read_text());assert seal['all_arms_sealed'] and seal['models']==12
 qids=torch.where(plan['fold']!=0)[0];assert len(qids)==1226
 ids=torch.cat([qids[:32],qids[-10:]]);assert len(ids.unique())==42
 sourceids=plan['row_ids'][ids]
 rows=json.loads(Path(cfg['rows']).read_text());rows=[r for r in rows if r['label']<4]
 selected=[rows[i] for i in sourceids.tolist()]
 assert [r['identity_component'] for r in selected]==[plan['groups'][i] for i in ids.tolist()]
 assert torch.equal(torch.tensor([r['label'] for r in selected]),plan['y'][ids])
 root=Path(cfg['image_root']).resolve();allowed={(root/r['relative_path']).resolve() for r in selected}
 assert all(p.is_relative_to(root) for p in allowed)
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve()
   if p.is_relative_to(root):assert p in allowed,'UNAUTHORIZED_IMAGE'
 sys.addaudithook(guard)
 write('PLAN_SEAL.json',dict(rows=42,producer_fold=0,source_commit=cfg['source_commit'],query_excluded=False,models_presealed=True,cells=['first32','tail10','tail10_padded32']))
 staged=Path(cfg['matched'])/'pixels_0';index=json.loads((staged/'INDEX.json').read_text());assert index['rows']==3236
 samples={}
 for cell in index['chunks']:
  wanted=[i for i in sourceids.tolist() if cell['first']<=i<cell['first']+cell['rows']]
  if wanted:
   chunk=torch.load(staged/cell['file'],map_location='cpu',weights_only=False)
   for i in wanted:samples[i]=chunk[i-cell['first']].clone()
   del chunk
 x=torch.stack([samples[i] for i in sourceids.tolist()]);del samples
 model,_,_,_,_,pointwise=setup(cfg)
 from run_medical_v2 import transform as image_transform
 pre=image_transform(False);parts=[]
 for row in selected:
  with Image.open(root/row['relative_path']) as im:parts.append(pre(im.convert('RGB')))
 y=torch.stack(parts);del parts
 assert x.shape==y.shape==(42,3,224,224) and torch.isfinite(x).all() and torch.isfinite(y).all()
 write('INPUT_COMPARISON.json',dict(delta=delta(x,y),equal=torch.equal(x,y),decoder=sig))
 teacher=torch.load(prod/'prep/0.teacher.private.pt',map_location='cpu',weights_only=False)
 base=plan['cached_g'][ids];scores=transform(base,teacher['mapper'])@teacher['W'][:-1]+teacher['W'][-1]
 old=torch.load(Path(cfg['old_responses']),map_location='cpu',weights_only=False)
 assert torch.equal(old['y'],plan['y']) and old['groups']==plan['groups']
 reference_error=delta(scores,old['score']['F_out'][ids]);write('REFERENCE_LINEAGE.json',reference_error)
 assert reference_error['maxabs']<=1e-10
 versions={n:p._version for n,p in model.named_parameters()};matrices={};cells=[]
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True
 for name,tensor,keep in [('first32',y[:32],32),('tail10',y[-10:],10),('tail10_padded32',torch.cat([y[-10:],y[-10:-9].repeat(22,1,1,1)]),10)]:
  with torch.no_grad():
   o=raw(model,pointwise,tensor.cuda());g=F.normalize(F.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1).double(),dim=1).cpu()[:keep]
  assert torch.isfinite(g).all();matrices[name]=g
  cells.append(dict(name=name,input_rows=len(tensor),retained_rows=keep,cudnn_tf32=True,matmul_tf32=False,finite=True));write('CELL_RECEIPTS.json',cells)
 write('FEATURE_SEAL.json',dict(cells=cells,encoder_calls=3,encoder_images=74,unique_query_images=42,backward=0,updates=0,fits=0))
 comparisons={}
 for name,g in matrices.items():
  idx=slice(0,32) if name=='first32' else slice(32,42)
  score=transform(g,teacher['mapper'])@teacher['W'][:-1]+teacher['W'][-1]
  fe=delta(g,base[idx]);se=delta(score,scores[idx])
  comparisons[name]=dict(feature=fe,score=se,replay_gate=fe['maxabs']<=1e-6 and se['maxabs']<=1e-5)
 pairs={'tail10_vs_padded32':delta(matrices['tail10'],matrices['tail10_padded32'])}
 unchanged=all(p._version==versions[n] and not p.requires_grad for n,p in model.named_parameters());assert unchanged
 write('aggregate.json',dict(decision='QUERY_CONTEXT_DIAGNOSTIC_COMPLETE',comparisons=comparisons,pairs=pairs,parameters_unchanged=unchanged,method_success=False,performance_test=False))
start=time.time();code=1
try:main();code=0
except BaseException as e:
 write('WORKER_FAILURE.json',dict(type=type(e).__name__,no_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0));write('WORKER_EXIT.json',dict(exit_code=code))
raise SystemExit(code)
