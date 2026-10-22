"""R0: aggregate CPU-only analysis of pre-existing probability mixtures."""
import os,json,time,resource,traceback,itertools
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);torch.set_num_threads(4)
SEEDS=[74510,74511,74512];K=13;EPS=1e-6
COUNTS=dict(optimizer_steps=0,backward=0,image_encoder_calls=0,text_encoder_calls=0,new_model_downloads=0,ridge_solves=0)
def write(n,v):
 t=out/(n+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(out/n)
def metric(p,y,groups,uncertain=None,candidates=None):
 pred=p.argmax(1);ok=pred==y;den=torch.bincount(y,minlength=K);cor=torch.bincount(y[ok],minlength=K)
 assert p.shape==(len(y),K) and torch.isfinite(p).all() and (den>0).all()
 vals=p.topk(2,dim=1).values;result=dict(BA=float((cor.double()/den).mean()),counts=den.tolist(),correct=cor.tolist(),recall=(cor.double()/den).tolist(),confusion=torch.bincount(y*K+pred,minlength=K*K).reshape(K,K).tolist())
 details=[]
 for c in range(K):
  mask=y==c;gs=sorted({groups[i] for i in torch.where(mask)[0].tolist()});gc=[float(ok[torch.tensor([g==z for g in groups])].double().mean()) for z in gs]
  details.append(dict(class_id=c,known_groups=len(gs),group_recall=sum(gc)/len(gc),mean_top1=float(vals[mask,0].mean()),mean_true_probability=float(p[mask,c].mean()),mean_top2_margin=float((vals[mask,0]-vals[mask,1]).mean()),uncertain_rows=0 if uncertain is None else int(uncertain[mask].sum())))
 result['per_class']=details;result['group_BA']=sum(v['group_recall'] for v in details)/K
 if candidates is not None:
  possible=candidates.gather(1,y[:,None]).squeeze(1);certain=possible&(candidates.sum(1)==1)
  low=torch.bincount(y[certain],minlength=K);high=torch.bincount(y[possible],minlength=K)
  result['numerical_sensitivity_bounds']=dict(correct_lower=low.tolist(),correct_upper=high.tolist(),BA_lower=float((low.double()/den).mean()),BA_upper=float((high.double()/den).mean()))
 return result

def paired(a,b,y,uncertain=None):
 aa=a.argmax(1)==y;bb=b.argmax(1)==y;items=[]
 for c in range(K):
  mask=y==c;n=int(mask.sum());r=int((mask&~aa&bb).sum());h=int((mask&aa&~bb).sum());items.append(dict(class_id=c,n=n,rescued=r,harmed=h,net=r-h,BA_contribution_pp=100*(r-h)/n/K,uncertain_rows=0 if uncertain is None else int((mask&uncertain).sum())))
 direct=float(((torch.bincount(y[bb],minlength=K)-torch.bincount(y[aa],minlength=K)).double()/torch.bincount(y,minlength=K)).mean())*100
 assert abs(sum(v['BA_contribution_pp'] for v in items)-direct)<1e-10
 return dict(delta_BA_pp=direct,per_class=items)

def flow(n,a,m,y,u):
 flags=[p.argmax(1)==y for p in [n,a,m]];result=[]
 for c in range(K):
  mask=y==c;cells={}
  for bits in itertools.product([0,1],repeat=3):
   sel=mask.clone()
   for flag,bit in zip(flags,bits):sel&=flag if bit else ~flag
   cells[''.join(map(str,bits))]=int(sel.sum())
  assert sum(cells.values())==int(mask.sum())
  result.append(dict(class_id=c,counts_NCM_adapter_mix=cells,uncertain_adapter_rows=int((mask&u).sum())))
 return result

def recover(n,m):
 assert n.dtype==m.dtype==torch.float32
 pn=n.softmax(1);a=2*m.double()-pn.double();af=(2*m-pn).double();ad=2*m.double()-n.double().softmax(1)
 info=dict(probability_min=float(a.min()),probability_max=float(a.max()),max_row_sum_error=float((a.sum(1)-1).abs().max()),mix_replay_maxabs=float((.5*(a+pn.double())-m.double()).abs().max()),float32_recovery_maxabs=float((a-af).abs().max()),float64_softmax_recovery_maxabs=float((a-ad).abs().max()))
 write('NUMERICAL_LATEST.json',info)
 assert info['probability_min']>=-EPS and info['probability_max']<=1+EPS and info['max_row_sum_error']<=K*EPS
 assert info['float32_recovery_maxabs']<=EPS and info['float64_softmax_recovery_maxabs']<=EPS
 top=a.topk(2,dim=1).values;unc=(top[:,0]-top[:,1]<=2*EPS)|(a.argmax(1)!=af.argmax(1))|(a.argmax(1)!=ad.argmax(1))
 cand=a>=a.max(1,keepdim=True).values-2*EPS
 cand.scatter_(1,af.argmax(1)[:,None],True);cand.scatter_(1,ad.argmax(1)[:,None],True)
 info.update(uncertain_rows=int(unc.sum()),argmax_diff_float32=int((a.argmax(1)!=af.argmax(1)).sum()),argmax_diff_float64_softmax=int((a.argmax(1)!=ad.argmax(1)).sum()),minimum_top2_margin=float((top[:,0]-top[:,1]).min()))
 return pn.double(),a,m.double(),unc,cand,info

def analyse(data,y,groups):
 n,a,m,u,cand=data
 return dict(summary=dict(NCM=metric(n,y,groups),adapter=metric(a,y,groups,u,cand),mix=metric(m,y,groups)),paired=dict(adapter_vs_NCM=paired(n,a,y,u),mix_vs_NCM=paired(n,m,y),mix_vs_adapter=paired(a,m,y,u)),flow=flow(n,a,m,y,u))

def main():
 assert not torch.cuda.is_available(),'CUDA must be hidden'
 # Nontrivial reconstruction and correctness flow self-check, no optimizer/encoder.
 pn=torch.tensor([[.4,.35,.25]],dtype=torch.float32);pa=torch.tensor([[.25,.35,.4]],dtype=torch.float32);mix=.5*(pn+pa)
 assert int(pn.argmax())==0 and int(pa.argmax())==2 and int(mix.argmax())==1
 assert (2*mix.double()-pn.double()-pa.double()).abs().max()<EPS
 roots={k:Path(v) for k,v in cfg['inputs'].items()};plans={k:json.loads((r/'PLAN_SEAL.json').read_text()) for k,r in roots.items()};configs={k:json.loads((r/'config.private.json').read_text()) for k,r in roots.items()}
 assert plans['CLIP']['class_names']==plans['BiomedCLIP']['class_names']
 assert configs['CLIP']['prior_root']==configs['BiomedCLIP']['prior_root']
 prior=Path(configs['CLIP']['prior_root']);meta={role:[r for r in json.loads((prior/f'HK.{role}.rows.private.json').read_text()) if r['label']<K] for role in ['S_old','E','Qgate']}
 seenp=set();seeng=set();audit={}
 for role,expected in [('S_old',3140),('E',986),('Qgate',388)]:
  rr=meta[role];assert len(rr)==expected;paths={r['relative_path'] for r in rr};groups={r['identity_component'] for r in rr};assert len(paths)==len(rr) and not paths&seenp and not groups&seeng
  seenp|=paths;seeng|=groups;audit[role]=dict(images=[sum(r['label']==c for r in rr) for c in range(K)],known_groups=[len({r['identity_component'] for r in rr if r['label']==c}) for c in range(K)],new_unexposed_groups='UNKNOWN',patient_independence='UNKNOWN')
 for p in plans.values():assert p['support_group_counts']==audit['S_old']['known_groups']
 saved={k:torch.load(r/'responses.private.pt',map_location='cpu',weights_only=False) for k,r in roots.items()}
 result=dict(source_commit=cfg['source_commit'],class_names=plans['CLIP']['class_names'],data_audit=audit,models={},cross_backbone={},numerical={},zero_work_counts=COUNTS,scientific_scope='Descriptive saved-response diagnosis only; no method selection',row_binding='Shared metadata paths/code lineage, all saved aggregate counts/confusions and common OriginalF exact; no embedded row IDs',new_training_authorized=False)
 for role in ['E','Qgate']:assert torch.equal(saved['CLIP'][role]['OriginalF'],saved['BiomedCLIP'][role]['OriginalF'])
 all_data={}
 for tag,root in roots.items():
  original=json.loads((root/'aggregate.json').read_text());result['models'][tag]={};result['numerical'][tag]={};all_data[tag]={}
  for role in ['E','Qgate']:
   rr=meta[role];y=torch.tensor([r['label'] for r in rr]);gg=[r['identity_component'] for r in rr]
   for key,s in saved[tag][role].items():
    got=metric(s.double(),y,gg);ref=original['cohorts'][role]['summary'][key]
    assert got['correct']==ref['correct'] and got['counts']==ref['counts'] and got['confusion']==ref['confusion']
   result['models'][tag][role]={};result['numerical'][tag][role]={}
   for seed in SEEDS:
    n,a,m,u,cand,info=recover(saved[tag][role]['NCM'],saved[tag][role][f'{seed}.flat']);all_data[tag][role,seed]=(n,a,m,u,cand)
    result['models'][tag][role][str(seed)]=analyse((n,a,m,u,cand),y,gg);result['numerical'][tag][role][str(seed)]=info
  y=torch.tensor([r['label'] for role in ['E','Qgate'] for r in meta[role]]);gg=[r['identity_component'] for role in ['E','Qgate'] for r in meta[role]]
  result['models'][tag]['pooled']={}
  for seed in SEEDS:
   data=tuple(torch.cat([all_data[tag][role,seed][i] for role in ['E','Qgate']]) for i in range(5));all_data[tag]['pooled',seed]=data;result['models'][tag]['pooled'][str(seed)]=analyse(data,y,gg)
 for role in ['E','Qgate','pooled']:
  rr=meta[role] if role!='pooled' else meta['E']+meta['Qgate'];y=torch.tensor([r['label'] for r in rr]);result['cross_backbone'][role]={}
  for seed in SEEDS:
   x=all_data['CLIP'][role,seed];z=all_data['BiomedCLIP'][role,seed];result['cross_backbone'][role][str(seed)]={arm:paired(x[i],z[i],y,x[3]|z[3] if arm=='adapter' else None) for i,arm in enumerate(['NCM','adapter','mix'])}
 write('aggregate.json',result);write('VERIFICATION.json',dict(all_saved_counts_confusions_reproduced=True,common_OriginalF_scores_exact=True,metadata_disjoint=True,probability_clipping=False,renormalization=False,zero_work_counts=COUNTS,semantic_class_binding=True,patient_independence='UNKNOWN',selfcheck_third_decision=True));write('STATUS.json',dict(phase='COMPLETE'))
start=time.time();code=1
try:main();code=0
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());write('STATUS.json',dict(phase='FAILED'))
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_RESOURCE.json',dict(exit_code=code,wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,RSS_bytes=u.ru_maxrss*1024,zero_work_counts=COUNTS))
raise SystemExit(code)
