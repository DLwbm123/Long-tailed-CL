"""Analytic norms of affine adapter sums on sealed TRAIN means/prompts."""
import os,json,ast,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ref=Path(cfg['reference']);torch.set_num_threads(4);torch.set_num_interop_threads(2)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def affine(state,x,prefix,k):
 return torch.stack([F.linear(x,state[f'{prefix}.{i}.fc.0.weight'],state[f'{prefix}.{i}.fc.0.bias'])*state[f'{prefix}.{i}.log_alpha'].clamp(max=0).exp() for i in range(k)]).sum(0)
def main():
 rows=[];proofs=[]
 # Arithmetic check for a replicated adapter: sum grows, mean preserves norm.
 fixture={'Image_Adapter.0.fc.0.weight':torch.eye(2),'Image_Adapter.0.fc.0.bias':torch.zeros(2),'Image_Adapter.0.log_alpha':torch.tensor(0.)}
 for key,v in list(fixture.items()):fixture[key.replace('.0.','.1.',1)]=v.clone()
 x=torch.tensor([[1.,2.]])
 assert torch.equal(affine(fixture,x,'Image_Adapter',2),2*affine(fixture,x,'Image_Adapter',1))
 for ds in ['NIH','ISIC','HK']:
  source=ref/'linear64'/ds;dc=json.loads((source/'config.private.json').read_text());cache=Path(dc['cache_root'] if ds=='NIH' else dc['shared_cache'])
  if ds=='NIH':
   tree=ast.parse(Path(dc['base_runner']).read_text());v={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['ORDER','ORIGINAL_NAMES','LEAF_TEMPLATE']};names=[v['ORIGINAL_NAMES'][i] for i in v['ORDER']];template=v['LEAF_TEMPLATE'];sizes=[10,2,2,2,2,2]
  else:names=[dc['names'][i] for i in dc['order']];template=dc['leaf_template'];sizes=dc['task_sizes']
  for seed in [74002,74003,74004]:
   for stage in range(1,len(sizes)):
    write('STATUS.json',dict(phase='AFFINE_NORM_CHECK',dataset=ds,seed=seed,stage=stage,time=time.time()))
    previous=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage-1}.private.pt',map_location='cpu',weights_only=False);current=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage}.private.pt',map_location='cpu',weights_only=False)
    ps=previous['network'];cs=current['network'];old=previous['total'];saved={key:cs[key].clone() for key in cs if 'Adapter.' in key}
    bank=torch.load(cache/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False);idx=[bank['prompts'].index(template.format(n)) for n in names[:old]]
    prompts=cs['model.text_bank'][idx];assert torch.allclose(prompts.norm(dim=1),torch.ones(old),atol=2e-6)
    means=ps['fc.weight'][:old];assert means.shape==(old,512)
    for prefix,x in [('Image_Adapter',means),('Text_Adapter',prompts)]:
     for key in ps:
      if key.startswith(prefix+'.'):assert torch.equal(ps[key],cs[key]),'OLD_ADAPTER_CHANGED'
     before=affine(ps,x,prefix,stage);prefix_now=affine(cs,x,prefix,stage);after=affine(cs,x,prefix,stage+1);assert torch.equal(before,prefix_now)
     b=before.double().norm(dim=1);a=after.double().norm(dim=1);assert (b>1e-20).all() and (a>1e-20).all()
     ratio=a/b;sumerror=ratio.log().abs();meanerror=(ratio*stage/(stage+1)).log().abs()
     for i in range(old):rows.append(dict(dataset=ds,seed=seed,stage=stage,class_index=i,modality=prefix,old_modules=stage,new_modules=stage+1,sum_norm_ratio=float(ratio[i]),sum_abs_log_norm_change=float(sumerror[i]),counterfactual_mean_abs_log_norm_change=float(meanerror[i])))
    assert all(torch.equal(val,cs[key]) for key,val in saved.items());proofs.append(dict(dataset=ds,seed=seed,stage=stage,old_classes=old,old_prefix_exact=True,parameters_unchanged=True))
 assert len(rows)==990 and len(proofs)==36
 summary={}
 for ds in ['NIH','ISIC','HK']:
  summary[ds]={}
  for modality in ['Image_Adapter','Text_Adapter']:
   z=[]
   for seed in [74002,74003,74004]:
    rr=[r for r in rows if r['dataset']==ds and r['modality']==modality and r['seed']==seed];rat=torch.tensor([r['sum_norm_ratio'] for r in rr],dtype=torch.double);s=sum(r['sum_abs_log_norm_change'] for r in rr)/len(rr);m=sum(r['counterfactual_mean_abs_log_norm_change'] for r in rr)/len(rr)
    z.append(dict(seed=seed,rows=len(rr),median_sum_norm_ratio=float(rat.median()),sum_mean_abs_log_change=s,mean_mean_abs_log_change=m,relative_reduction=(s-m)/s))
   summary[ds][modality]=z
 eligible={mod:sum(z['median_sum_norm_ratio']>=1.05 and z['relative_reduction']>=.1 for z in summary['NIH'][mod]) for mod in ['Image_Adapter','Text_Adapter']}
 other=all(sum(z['relative_reduction'] for z in summary[ds][mod])/3>=0 for ds in ['ISIC','HK'] for mod in ['Image_Adapter','Text_Adapter'])
 passed=all(v>=2 for v in eligible.values()) and other
 write('RESULTS.json',dict(records=rows,proofs=proofs,summary=summary,optimizer_steps=0,classification_scores=0,VAL_reads=0,TEST_reads=0,no_new_random_draws=True))
 write('FEASIBILITY.json',dict(passed=passed,NIH_eligible_seeds_by_modality=eligible,other_domains_mean_reduction_nonnegative=other,not_performance_or_causality_evidence=True));write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
