"""Training-only provenance and duplicate-coordinate audit; never fits a predictor."""
import json,os,sys,time,resource
from pathlib import Path
import torch

def coverage(perm,d,half):
 counts=torch.bincount(perm[:d]%half,minlength=half)
 return dict(selected_coordinates=d,channel_count=half,absent_channels=int((counts==0).sum()),single_channels=int((counts==1).sum()),double_channels=int((counts==2).sum()))
def main(cfg):
 start=time.monotonic();out=Path(cfg['output']);ref=Path(cfg['reference']).resolve();allowed=ref/'inputs/0.private.pt';torch.set_num_threads(4);assert not torch.cuda.is_initialized()
 # Counter checks are independent of real features or labels.
 assert coverage(torch.tensor([0,3,1,4,2,5]),4,3)==dict(selected_coordinates=4,channel_count=3,absent_channels=1,single_channels=0,double_channels=2)
 bc=json.loads(Path(cfg['benchmark_config']).read_text());legacy=Path(bc['legacy_source']).resolve()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(ref):assert p==allowed,'only initial input bundle permitted'
  assert not p.is_relative_to(Path(bc['output']).resolve()),'test cache forbidden'
  for name in ['images','development_images']:assert not p.is_relative_to(Path(bc['datasets']['ISIC'][name]).resolve()),'image forbidden'
 sys.addaudithook(guard)
 for p in [ref/'inputs/1.private.pt',Path(bc['output'])/'ISIC.test.features.private.pt']:
  try:guard('open',(str(p),'rb'))
  except AssertionError:pass
  else:raise AssertionError('isolation probe')
 source=(legacy/'tools/run_multilabel.py').read_text();st=source.index('class ApartFeatures');en=source.index('\ndef extract(',st);constructor=source[st:en];loader=(legacy/'third_party/APART/utils/medical_v2.py').read_text();backbone=(legacy/'third_party/APART/backbone/vision_transformer_adapter_pool_a.py').read_text()
 provenance=dict(constructor_calls_Learner='self.learner = Learner(args)' in constructor,constructor_no_incremental_train='incremental_train(' not in constructor,constructor_no_optimizer_step='.step(' not in constructor,loader_uses_verified_pretrained='state = verified_state(path)' in loader,loader_requires_1000_class_head="state.pop('head.weight').shape == (1000, 768)" in loader,adapter_up_projection_zero_init='nn.init.zeros_(self.up_proj.weight)' in backbone and 'nn.init.zeros_(self.up_proj.bias)' in backbone,scope='source inspection of extractor/constructor/loader; not pretrained-corpus overlap certification')
 obj=torch.load(allowed,map_location='cpu',weights_only=False);x=obj['g'][~obj['valid']].double();assert x.shape==(2555,1536) and torch.isfinite(x).all();left,right=x[:,:768],x[:,768:];delta=left-right;exact=torch.equal(left,right);mx=float(delta.abs().max());rms=float(delta.square().mean().sqrt());perm=torch.randperm(1536,generator=torch.Generator().manual_seed(74010));counts={a:coverage(perm,d,768) for a,d in [('G',1535),('P',1375),('C',1318),('Full',1536)]}
 result=dict(source_commit=cfg['source_commit'],training_rows=2555,validation_rows_used=0,test_rows_used=0,new_fits=0,new_predictions=0,new_forwards=0,source=provenance,branch_half_exact=exact,branch_half_max_abs=mx,branch_half_RMS=rms,global_norm_max_error=float((x.norm(dim=1)-1).abs().max()),coordinate_coverage=counts,synthetic_counter_pass=True,access_guard_probes_pass=True,decision='DUPLICATE_BRANCH_CONFIRMED_INITIAL_TRAINING' if exact else 'UNRESOLVED_BRANCH_EQUIVALENCE',independent_rank_not_estimated=True,no_test_performance_claim=True,GPU_initialized=torch.cuda.is_initialized(),worker_wall_seconds=time.monotonic()-start,worker_CPU_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 (out/'aggregate.json').write_text(json.dumps(result,indent=2,allow_nan=False));(out/'status.json').write_text(json.dumps(dict(state='COMPLETE',decision=result['decision'])));print(result['decision'])
