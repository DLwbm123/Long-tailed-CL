import json,gc
from pathlib import Path
import torch
from train_base import setup
from binding import native
from solver import mapper,transform
from core_attribution import weights
from core_depth import PERM
from train import write
from evaluation import evaluate,points
from space import check_space
ARMS=['F','E','L','R','D']
def fit(a,b,alpha):
 h=a/4;h=h.clone();h.diagonal()[:-1].add_(alpha);w=torch.linalg.solve(h,b/4);res=float((h@w-b/4).abs().max());assert res<=1e-7 and torch.isfinite(w).all();return w,res

def run(cfg):
 out=Path(cfg['output']);training=Path(cfg['training_output']);assert json.loads((training/'ENCODER_SEAL.json').read_text())['future_frozen'];obj=torch.load(Path(cfg['reference'])/'inputs/0.private.pt',weights_only=False);tr=~obj['valid'];va=obj['valid'];y=obj['y'];gg=obj['groups'];train_groups=[gg[i] for i in torch.where(tr)[0].tolist()];devgroups=[gg[i] for i in torch.where(va)[0].tolist()];assert not set(train_groups)&set(devgroups);assert [int(tr.sum()),int(va.sum())]==[2555,681];q=weights(y[tr],train_groups);fits=[];sizes={};selected={};pred={};rows={};maps={}
 def extract(arm):
  cache=out/(arm+'.g.private.pt');seal=out/(arm+'.FEATURE_SEAL.json');started=out/(arm+'.ENCODING_STARTED.private.json');over=torch.load(training/(arm+'.encoder.private.pt'),weights_only=False)['bias_overrides']
  if seal.exists():return torch.load(cache,weights_only=False),over
  assert not started.exists(),'no silent repeat of unsealed encoding';write(started,dict(arm=arm,quota=3236));m,params,keys,ckeys,heads,pw=setup(cfg);values=[];nativeover=native(over)
  with torch.no_grad():
   for k in keys:params[k].copy_(nativeover[k].cuda())
   index=json.loads((Path(cfg['matched'])/'pixels_0/INDEX.json').read_text());assert index['rows']==3236;done=0
   for cell in index['chunks']:
    xx=torch.load(Path(cfg['matched'])/'pixels_0'/cell['file'],weights_only=False)
    for x in xx.split(32):values.append(torch.nn.functional.normalize(m(x.cuda()).cpu().double(),dim=1));done+=len(x);check_space(cfg['window_root']);write(out/'status.json',dict(state='RUNNING',arm=arm,phase='ENDPOINT_ENCODING',rows=done,total_rows=3236))
  g=torch.cat(values);assert list(g.shape)==[3236,1536] and torch.isfinite(g).all();check_space(cfg['window_root'],reserve=g.numel()*g.element_size()*3);torch.save(g.contiguous(),cache);write(seal,dict(arm=arm,rows=3236,one_pass=True,test_used=False));del m,params,heads;gc.collect();torch.cuda.empty_cache();return g,over
 for arm in ARMS:
  g,over=(obj['g'].double(),None) if arm=='F' else extract(arm);rows[arm]=len(g);mp=mapper(g[tr],q);maps[arm]=mp;z=transform(g,mp);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);a=x[tr].T@(x[tr]*q[:,None]);a=(a+a.T)/2;b=x[tr].T@(torch.nn.functional.one_hot(y[tr],6).double()*q[:,None]);options=[]
  for alpha in [.0001,.01,1.]:
   w,res=fit(a,b,alpha);ba=points((x[va]@w[:,:4]).argmax(1),y[va])['BA'];fits.append(dict(arm=arm,alpha=alpha,status='CERTIFIED',residual=res,development_BA=ba));options.append((ba,alpha,w))
  _,alpha,w=max(options,key=lambda v:(v[0],v[1]));selected[arm]=alpha;pred[arm]=(x[va]@w[:,:4]).argmax(1);write(out/(arm+'.SELECTION_SEAL.json'),dict(alpha=alpha,own_initial_validation_only=True,grid=[.0001,.01,1.],test_used=False))
  if arm=='F':assert alpha==.01 and points(pred[arm],y[va])['correct']==[55,2,22,430]
  else:
   ix=torch.triu_indices(len(a),len(a));artifact=dict(A=a[ix.unbind()].contiguous(),B=b.contiguous(),W=w.contiguous(),mapper=mp,PERM=PERM,bias_overrides=over,bias_representation='absolute_overrides',alpha=alpha,classes=4,source_commit=cfg['source_commit']);file=out/(arm+'.state.private.pt');check_space(cfg['window_root'],reserve=24*2**20);torch.save(artifact,file);sizes[arm]=file.stat().st_size;assert sizes[arm]<=19058800;back=torch.load(file,weights_only=False);assert torch.equal(back['W'],w) and all(torch.equal(v,back['bias_overrides'][k]) for k,v in over.items());rec=torch.zeros_like(a);rec[ix.unbind()]=back['A'];rec[ix[1],ix[0]]=back['A'];assert torch.equal(rec,a);re,_=fit(rec,back['B'],alpha);assert (re-w).abs().max()<=1e-10
  write(out/'partial.json',dict(fits=fits,alphas=selected,artifact_bytes=sizes,feature_rows=rows));del g,x,z,a,b
 write(out/'MODEL_SEAL.json',dict(fitted_endpoints=ARMS,total_fits=15,new_artifacts=4,alpha=selected,test_used=False,future_used=False));assert len(fits)==15 and len(sizes)==4;result=evaluate(pred,y[va],devgroups);result.update(fits=fits,artifact_bytes=sizes,alphas=selected,feature_rows=rows,source_commit=cfg['source_commit'],new_parameter_updates_during_fit=0);torch.save(dict(pred=pred,y=y[va],groups=devgroups),out/'development_predictions.private.pt');write(out/'aggregate.json',result);write(out/'status.json',dict(state='COMPLETE',decision=result['decision']))
