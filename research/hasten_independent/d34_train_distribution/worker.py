"""Full initial-task TRAIN distributions under sealed initial/final models."""
import os,json,time,traceback,sys,ast,csv
from pathlib import Path
import torch
from compat import load_upstream,install_feature_model
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ref=Path(cfg['reference'])
torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))

def main():
 # A domain config supplies only paths and fixed original model options.
 first=json.loads((ref/'linear64/NIH/config.private.json').read_text());cfg.update(first);cfg['output']=str(out);cfg['image_nonlinear']=False
 # compat reads RUN_CONFIG at import-time load; retain diagnostic output and linear arm.
 runtime=out/'runtime/model_config.private.json';runtime.write_text(json.dumps(cfg));os.environ['RUN_CONFIG']=str(runtime)
 inc,up,geo=load_upstream(cfg['official']);records=[]
 for ds in ['NIH','ISIC','HK']:
  source=ref/'linear64'/ds;dc=json.loads((source/'config.private.json').read_text())
  if ds=='NIH':
   module=ast.parse(Path(dc['base_runner']).read_text());constants={node.targets[0].id:ast.literal_eval(node.value) for node in module.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['ORDER','ORIGINAL_NAMES','LEAF_TEMPLATE']}
   dc.update(task_sizes=[10,2,2,2,2,2],names=constants['ORIGINAL_NAMES'],order=constants['ORDER'],leaf_template=constants['LEAF_TEMPLATE'])
  sizes=dc['task_sizes'];stages=len(sizes);names=[dc['names'][i] for i in dc['order']]
  cache=Path(dc['cache_root'] if ds=='NIH' else dc['shared_cache'])
  def get(stage,seed):
   state=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage}.private.pt',map_location='cuda',weights_only=False)
   bank=torch.load(cache/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False)
   install_feature_model(inc,bank['text'],bank['prompts'],bank['logit_scale'])
   net=inc.hyper_net(dict(convnet_type='cached',curvature=1.,mapping_mode='mlp',nullspace_energy_threshold=.95))
   for k in range(stage+1):net.update_fc(sum(sizes[:k+1]));net.update_task()
   net.to('cuda');net.load_state_dict(state['network'],strict=True);net.eval();return net,state
  @torch.no_grad()
  def score(net,means,n):
   prompts=[dc['leaf_template'].format(v) for v in names[:n]]
   text=net.model.encode_text(net.tokenizer(prompts).cuda());im=net.trans_hyper(net.Image_encode(means));tx=net.trans_hyper(net.Text_encode(text))
   scores=-net.logit_scale.exp()*geo.pairwise_dist(im,tx,net.get_curv());assert torch.isfinite(scores).all();return scores
  bank=torch.load(cache/'stage_0.private.pt',map_location='cpu',weights_only=False)
  with open(Path(dc['manifest'])/'train.csv') as f:train=list(csv.DictReader(f))
  labels=torch.tensor([dc['order'].index(int(train[i]['original_label'])) for i in bank['ids']]);init=sizes[0]
  assert all(train[i]['split']=='train' for i in bank['ids']) and set(labels.tolist())==set(range(init))
  x=bank['x'];x=torch.nn.functional.normalize(x,dim=1) if ds=='NIH' else x
  assert x.shape==(len(labels),512) and torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6)
  for seed in [74002,74003,74004]:
   results={};counts=torch.bincount(labels,minlength=init)
   for stage in [0,stages-1]:
    write('STATUS.json',dict(phase='FULL_TRAIN_PROBE',dataset=ds,seed=seed,stage=stage,time=time.time()))
    net,state=get(stage,seed);seen=state['total']
    for c in range(init):assert torch.allclose(x[labels==c].mean(0).cuda(),net.fc.weight[c],atol=2e-7), 'INITIAL_TRAIN_MEAN_PREPROCESS_MISMATCH'
    saved=[p.detach().clone() for p in net.parameters()]
    corr={k:torch.zeros(init,dtype=torch.long) for k in ['hyper_old','hyper_all','NCM_old','NCM_all']}
    with torch.no_grad():
     for offset in range(0,len(x),512):
      v=x[offset:offset+512].cuda();y=labels[offset:offset+512].cuda()
      prompts=[dc['leaf_template'].format(n) for n in names[:seen]]
      text=net.model.encode_text(net.tokenizer(prompts).cuda());im=net.trans_hyper(net.Image_encode(v));tx=net.trans_hyper(net.Text_encode(text))
      hyper=-net.logit_scale.exp()*geo.pairwise_dist(im,tx,net.get_curv())
      ncm=torch.nn.functional.normalize(v,dim=1)@torch.nn.functional.normalize(net.fc.weight[:seen],dim=1).T
      assert torch.isfinite(hyper).all() and torch.isfinite(ncm).all()
      scores={'hyper_old':hyper[:,:init],'hyper_all':hyper,'NCM_old':ncm[:,:init],'NCM_all':ncm}
      for k,z in scores.items():corr[k]+=torch.bincount(y[z.argmax(1)==y].cpu(),minlength=init)
    assert all(torch.equal(p,q) for p,q in zip(saved,net.parameters()))
    assert torch.cuda.max_memory_reserved()<8*2**30
    results[str(stage)]={k:v.tolist() for k,v in corr.items()}
    del net,state,saved;torch.cuda.empty_cache()
   for c in range(init):records.append(dict(dataset=ds,seed=seed,class_index=c,train_count=int(counts[c]),initial={k:v[c] for k,v in results['0'].items()},final={k:v[c] for k,v in results[str(stages-1)].items()}))
  del x,bank,train,labels
 summaries={}
 for ds in ['NIH','ISIC','HK']:
  ss=[]
  for seed in [74002,74003,74004]:
   rr=[v for v in records if v['dataset']==ds and v['seed']==seed];metrics={stage:{k:100*sum(v[stage][k]/v['train_count'] for v in rr)/len(rr) for k in ['hyper_old','hyper_all','NCM_old','NCM_all']} for stage in ['initial','final']}
   ss.append(dict(seed=seed,**metrics,old_only_change_pp=metrics['final']['hyper_old']-metrics['initial']['hyper_old'],new_competition_delta_pp=metrics['final']['hyper_all']-metrics['final']['hyper_old']))
  summaries[ds]=ss
 write('RESULTS.json',dict(records=records,summaries=summaries,optimizer_steps=0,validation_reads=0,test_reads=0,parameter_immutability=True,full_initial_task_TRAIN=True,independent_confirmation=False))
 write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
