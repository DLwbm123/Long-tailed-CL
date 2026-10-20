import gc,json
from pathlib import Path
import torch
from binding import native
from geometry import PERM,transform,weights
from train_base import setup
from sampling import write
from data import initial,pixels
from space import check_space
from evaluation import points,evaluate

def run(cfg):
 root,out=Path(cfg['root']),Path(cfg['output'])
 seal=json.loads((root/'train/ENCODER_SEAL.json').read_text());assert seal['query_excluded_entire_producer'] and seal['steps_each']==800
 plan=torch.load(root/'prep/PLAN.private.pt',weights_only=False);base=initial(cfg);assert torch.equal(base['y'],plan['y']) and base['groups']==plan['groups']
 xx=pixels(cfg);fold=plan['fold'];tables={k:torch.empty(2555,4,dtype=torch.float64) for k in ['T_seen','T_out','F_seen','F_out']};sizes={};perfold={};foldscores={}
 for k in [0,1]:
  assert not (out/f'{k}.ENCODING_STARTED.private.json').exists(),'NO_REENCODING'
  write(out/f'{k}.ENCODING_STARTED.private.json',dict(rows=2555,development_used=False))
  ep=torch.load(root/'train'/f'{k}.encoder.private.pt',weights_only=False);t=torch.load(root/'prep'/f'{k}.teacher.private.pt',weights_only=False)
  m,params,keys,_,heads,_=setup(cfg)
  with torch.no_grad():
   bound=native(ep['bias_overrides'])
   for name in keys:params[name].copy_(bound[name].cuda())
   values=[]
   for start in range(0,len(xx),32):
    values.append(torch.nn.functional.normalize(m(xx[start:start+32].cuda()).cpu().double(),dim=1))
    write(out/'status.json',dict(state='RUNNING',fold=k,phase='ENCODING',rows=min(start+32,2555),total_rows=2555));check_space(cfg['window_root'])
   g=torch.cat(values)
  assert list(g.shape)==[2555,1536] and torch.isfinite(g).all();torch.save(g,out/f'{k}.g.private.pt');write(out/f'{k}.FEATURE_SEAL.json',dict(rows=2555,one_pass=True,query_forward_only_after_encoders_sealed=True,development_used=False))
  del m,params,heads,values;gc.collect();torch.cuda.empty_cache()
  z=transform(g,ep['mapper']);x=torch.cat([z,torch.ones(len(g),1,dtype=g.dtype)],1);support=fold==k;y=plan['y'];sg=[plan['groups'][i] for i in torch.where(support)[0]];q=weights(y[support],sg)/4
  a=x[support].T@(x[support]*q[:,None]);a=(a+a.T)/2;b=x[support].T@(torch.nn.functional.one_hot(y[support],6).double()*q[:,None]);ix=torch.triu_indices(len(a),len(a));w=torch.cat([ep['head'],torch.zeros(len(a),2,dtype=g.dtype)],1)
  artifact=dict(A=a[ix.unbind()].contiguous(),B=b,W=w,mapper=ep['mapper'],PERM=PERM,bias_overrides=ep['bias_overrides'],head_origin='joint_CE_not_ridge_solution_of_A_B',classes=4,source_commit=cfg['source_commit'])
  f=out/f'{k}.state.private.pt';check_space(cfg['window_root'],reserve=24*2**20);torch.save(artifact,f);sizes[str(k)]=f.stat().st_size;assert sizes[str(k)]<=19058800
  back=torch.load(f,weights_only=False);assert torch.equal(back['W'],w) and all(torch.equal(back['bias_overrides'][n],v) for n,v in ep['bias_overrides'].items())
  rec=torch.zeros_like(a);rec[ix.unbind()]=back['A'];rec[ix[1],ix[0]]=back['A'];assert torch.equal(rec,a)
  ts=z@w[:-1,:4]+w[-1,:4];replay=transform(g,back['mapper'])@back['W'][:-1,:4]+back['W'][-1,:4];assert torch.equal(ts,replay)
  fs=transform(base['g'],t['mapper'])@t['W'][:-1]+t['W'][-1]
  for family,score in [('T',ts),('F',fs)]:
   tables[family+'_seen'][support]=score[support];tables[family+'_out'][~support]=score[~support]
  foldscores[k]=(ts,fs)
 write(out/'MODEL_SEAL.json',dict(two_full_artifacts=sizes,two_teacher_fits=2,four_response_tables=['T_seen','T_out','F_seen','F_out'],development_used=False,test_used=False))
 # Only after all producers/heads/features are immutable, compute any risk.
 for k,(ts,fs) in foldscores.items():
  support=fold==k
  perfold[str(k)]={family+'_'+name:points(score[mask],plan['y'][mask]) for family,score in [('T',ts),('F',fs)] for name,mask in [('seen',support),('out',~support)]}
 result=evaluate(tables,plan['y'],plan['groups']);result.update(per_fold=perfold,artifact_bytes=sizes,teacher_residuals=[float(torch.load(root/'prep'/f'{k}.teacher.private.pt',weights_only=False)['residual']) for k in [0,1]],source_commit=cfg['source_commit'],feature_rows={'0':2555,'1':2555},new_head_fits_during_evaluation=0)
 torch.save(dict(score=tables,y=plan['y'],groups=plan['groups']),out/'responses.private.pt')
 write(out/'aggregate.json',result);write(out/'status.json',dict(state='COMPLETE',decision=result['decision']))
