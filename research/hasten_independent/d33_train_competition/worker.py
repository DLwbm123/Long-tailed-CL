"""Read-only TRAIN mean margins across sealed incremental boundaries."""
import os,json,time,traceback,sys,ast
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
  for seed in [74002,74003,74004]:
   for stage in range(1,stages):
    write('STATUS.json',dict(phase='TRAIN_MEAN_PROBE',dataset=ds,seed=seed,stage=stage,time=time.time()))
    prev,ps=get(stage-1,seed);cur,cs=get(stage,seed);old=ps['total'];seen=cs['total'];means=cs['network']['fc.weight'][:old].detach()
    assert torch.equal(means,ps['network']['fc.weight'])
    # Same input old means, each network's original text features.
    a=score(prev,means,old);bfull=score(cur,means,seen);b=bfull[:,:old];target=torch.arange(old,device='cuda')
    before=prev.state_dict();after=cur.state_dict();snap=[v.detach().clone() for v in after.values()]
    pa=a.argmax(1);pb=b.argmax(1);pc=bfull.argmax(1)
    def margin(s):
     mask=torch.eye(len(s),s.shape[1],device=s.device,dtype=torch.bool);other=s.masked_fill(mask,-torch.inf).max(1).values
     return s[target,target]-other
    ma,mb,mc=map(margin,[a,b,bfull])
    for c in range(old):
     records.append(dict(dataset=ds,seed=seed,stage=stage,class_index=c,old=old,seen=seen,before_correct=int(pa[c]==c),after_old_correct=int(pb[c]==c),after_all_correct=int(pc[c]==c),new_class_takeover=int(pb[c]==c and pc[c]!=c),before_margin=float(ma[c]),after_old_margin=float(mb[c]),after_all_margin=float(mc[c])))
    assert all(torch.equal(x,y) for x,y in zip(snap,after.values()));assert torch.cuda.max_memory_reserved()<8*2**30
    del prev,cur,ps,cs,before,after,snap;torch.cuda.empty_cache()
 summaries={}
 for ds in ['NIH','ISIC','HK']:
  seeds=[]
  for seed in [74002,74003,74004]:
   rr=[r for r in records if r['dataset']==ds and r['seed']==seed];n=len(rr)
   p={k:100*sum(r[k] for r in rr)/n for k in ['before_correct','after_old_correct','after_all_correct']}
   seeds.append(dict(seed=seed,rows=n,**p,drift_delta_pp=p['after_old_correct']-p['before_correct'],competition_delta_pp=p['after_all_correct']-p['after_old_correct'],new_class_takeovers=sum(r['new_class_takeover'] for r in rr)))
  summaries[ds]=dict(seeds=seeds,drift_visible_seeds=sum(r['drift_delta_pp']<=-1 for r in seeds),competition_visible_seeds=sum(r['competition_delta_pp']<=-1 for r in seeds))
 assert len(records)==3*(70+10+85)
 write('RESULTS.json',dict(records=records,summaries=summaries,optimizer_steps=0,validation_reads=0,test_reads=0,network_parameters_unchanged=True,old_means_unchanged=True,read_only_sealed_states=True,independent_confirmation=False))
 write('DECISION.json',dict(NIH_drift_priority=summaries['NIH']['drift_visible_seeds']>=2,NIH_competition_priority=summaries['NIH']['competition_visible_seeds']>=2,no_scientific_performance_claim=True))
 write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
