"""Measure the original additive text path on sealed TRAIN states only."""
import os,json,ast,time,traceback
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ref=Path(cfg['reference']);torch.set_num_threads(4);torch.set_num_interop_threads(2)
def write(n,z):(out/n).write_text(json.dumps(z,indent=2,allow_nan=False))
def forward(s,x,k):
 return torch.stack([torch.nn.functional.linear(x,s[f'Text_Adapter.{j}.fc.0.weight'],s[f'Text_Adapter.{j}.fc.0.bias'])*s[f'Text_Adapter.{j}.log_alpha'].clamp(max=0).exp() for j in range(k)]).sum(0)
def main():
 rows=[];proofs=[]
 for ds in ['NIH','ISIC','HK']:
  source=ref/'linear64'/ds;dc=json.loads((source/'config.private.json').read_text())
  if ds=='NIH':
   tree=ast.parse(Path(dc['base_runner']).read_text());v={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['ORDER','ORIGINAL_NAMES','LEAF_TEMPLATE']};names=[v['ORIGINAL_NAMES'][i] for i in v['ORDER']];template=v['LEAF_TEMPLATE'];sizes=[10,2,2,2,2,2]
  else:names=[dc['names'][i] for i in dc['order']];template=dc['leaf_template'];sizes=dc['task_sizes']
  cache=Path(dc['cache_root'] if ds=='NIH' else dc['shared_cache'])
  for seed in [74002,74003,74004]:
   for stage in range(1,len(sizes)):
    write('STATUS.json',dict(phase='TEXT_DRIFT_PROBE',dataset=ds,seed=seed,stage=stage,time=time.time()))
    p=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage-1}.private.pt',map_location='cpu',weights_only=False);c=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage}.private.pt',map_location='cpu',weights_only=False);old=p['total'];ps=p['network'];cs=c['network']
    bank=torch.load(cache/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False);idx=[bank['prompts'].index(template.format(n)) for n in names[:old]];x=cs['model.text_bank'][idx].detach();assert x.shape==(old,512)
    for k,v in ps.items():
     if k.startswith('Text_Adapter.'):assert torch.equal(v,cs[k]),'FROZEN_TEXT_PARAMETER_CHANGED'
    previous=forward(ps,x,stage);prefix=forward(cs,x,stage);current=forward(cs,x,stage+1)
    assert torch.equal(previous,prefix);delta=current-previous
    w=cs[f'Text_Adapter.{stage}.fc.0.weight'].detach().clone().requires_grad_();bias=cs[f'Text_Adapter.{stage}.fc.0.bias'];alpha=cs[f'Text_Adapter.{stage}.log_alpha'].clamp(max=0).exp();extra=torch.nn.functional.linear(x,w,bias)*alpha
    grad=torch.autograd.grad((prefix.detach()+extra).square().mean(),w,retain_graph=True)[0];zero=torch.autograd.grad((prefix.detach()+0*extra).square().mean(),w)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0 and torch.count_nonzero(zero)==0
    relative=delta.double().norm(dim=1)/previous.double().norm(dim=1).clamp_min(1e-30);assert torch.isfinite(relative).all()
    for i in range(old):rows.append(dict(dataset=ds,seed=seed,stage=stage,class_index=i,relative_euclidean_text_change=float(relative[i])))
    proofs.append(dict(dataset=ds,seed=seed,stage=stage,old_classes=old,frozen_old_parameters_exact=True,prefix_output_exact=True,new_adapter_old_prompt_gradient_norm=float(grad.norm()),masked_new_gradient_zero=True))
 assert len(proofs)==36 and len(rows)==495
 ok=all(any(r['dataset']=='NIH' and r['seed']==seed and r['relative_euclidean_text_change']>1e-6 for r in rows) for seed in [74002,74003,74004])
 write('RESULTS.json',dict(records=rows,proofs=proofs,optimizer_steps=0,VAL_reads=0,TEST_reads=0,no_random_draws=True))
 write('FEASIBILITY.json',dict(passed=ok,cells=36,old_class_rows=495,not_performance_evidence=True,not_natural_loss_gradients=True))
 write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
