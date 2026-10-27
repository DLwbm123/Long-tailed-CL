"""Complete arrived-TRAIN paired orientation measurement of sealed R77 endpoints."""
from pathlib import Path
import os,json,time,torch,traceback
from common import write,guard
from data import train_rows,get_cache,Tokens,DataLoader,store
from folds import assign_folds
from vision import NativeTail,adapt_native_tail,load_delta,delta_state
from support import make_support,support_scores
from diagnostics import paired_stats
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());rc=json.loads(Path(cfg['root_config']).read_text());root=Path(rc['output']);out=Path(cfg['output']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
 rows,y=train_rows(cfg);counts=torch.bincount(y,minlength=len(cfg['order']));bundle=torch.load(root/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);paths=[];records=[];summaries=[];oldfolds={};totalrows=0
 with torch.no_grad():
  for stage,classes in enumerate(cfg['stages']):
   paths.extend(get_cache(rc,cfg,stage=stage));data=Tokens(paths);assert data.split=='train';N=len(data);C=sum(map(len,cfg['stages'][:stage+1]));nc=counts[:C];assert torch.equal(torch.bincount(data.labels,minlength=C),nc)
   fold=torch.tensor(assign_folds(data.labels.tolist(),[rows[i]['identity_component'] for i in data.ids]));fm=dict(zip(data.ids,fold.tolist()));assert all(fm[i]==b for i,b in oldfolds.items());oldfolds=fm
   for name in ['sft_reference','prototype_grpo','distribution_grpo']:
    write(out/'STATUS.json',dict(time=time.time(),phase='FLIPPED_TRAIN_ENDPOINT_FORWARD',stage=stage,model=name,completed_records=len(records),flipped_endpoint_forward_rows=totalrows))
    base=Path(rc['r77_run'])/'HK'
    if name=='sft_reference':
     h=torch.load(Path(rc['matched_reference_run'])/'HK/74002'/f'head{stage}.private.pt',map_location='cpu',weights_only=False)
     saved=torch.load(base/'74002'/f'SFT_SUPPORT{stage}.private.pt',map_location='cpu',weights_only=False)
    else:
     h=torch.load(base/name/'74002'/f'head{stage}.private.pt',map_location='cpu',weights_only=False);saved=h
    assert h['seen']==C and h['train_counts'].tolist()==nc.tolist()
    z=saved['features'].float();labels=saved['labels']
    if name=='sft_reference':
     merge_order=torch.cat([torch.where(fold==1)[0],torch.where(fold==0)[0]])
     assert torch.equal(labels,data.labels[merge_order])
     original_z=torch.empty_like(z);original_z[merge_order]=z;z=original_z;labels=data.labels
    assert torch.equal(labels,data.labels) and z.shape==(N,512) and torch.allclose(z.norm(dim=1),torch.ones(N),atol=2e-6)
    full=make_support(z,labels,C);err=float((full['prototypes']-h['prototypes']).abs().max());assert err<=2e-5
    m=NativeTail(bundle).cuda().eval();params=adapt_native_tail(m);load_delta(m,h['delta']);frozen={k:p.detach().cpu().clone() for k,p in m.named_parameters() if not p.requires_grad};parts=[]
    for tokens,old_original_base,yy,ix in DataLoader(data,batch_size=64):
     guard(cfg);zz=m(tokens.cuda());assert zz.shape==(len(yy),512) and torch.isfinite(zz).all();parts.append(zz.cpu());totalrows+=len(yy)
     assert torch.cuda.max_memory_reserved(0)<12*2**30 and torch.cuda.max_memory_reserved(1)<12*2**30
    flipped=torch.cat(parts);assert torch.allclose(flipped.norm(dim=1),torch.ones(N),atol=2e-6)
    pairedcos=(z.double()*flipped.double()).sum(1)
    bymode={}
    for readout in ['prototype','distribution']:
     original_scores=torch.empty(N,C);flipped_scores=torch.empty(N,C)
     for q in [0,1]:
      query=fold==q;support=fold!=q;assert not bool((query&support).any());su=make_support(z[support].cuda(),labels[support].cuda(),C)
      ii=torch.where(query)[0]
      for lo in range(0,len(ii),64):
       jj=ii[lo:lo+64];original_scores[jj]=support_scores(z[jj].cuda(),su,readout+'_').cpu();flipped_scores[jj]=support_scores(flipped[jj].cuda(),su,readout+'_').cpu()
     for policy in ['raw','natural']:
      prior=nc.float().log() if policy=='natural' else torch.zeros(C)
      ss=paired_stats(original_scores+prior,flipped_scores+prior,labels,C)
      for c in range(C):ss[c]['paired_embedding_cosine_mean']=float(pairedcos[labels==c].mean())
      bymode[readout+'_'+policy]=ss
    assert all(torch.equal(p.detach().cpu(),frozen[k]) for k,p in m.named_parameters() if not p.requires_grad);assert all(torch.equal(delta_state(m)[k],v) for k,v in h['delta'].items()) and all(p.grad is None for p in params)
    summaries.append(dict(stage=stage,model=name,seed=74002,seen=C,modes=bymode))
    records.append(dict(stage=stage,model=name,seed=74002,seen=C,TRAIN_rows=N,counts=nc.tolist(),flipped_model_forward_rows=N,saved_original_embedding_rows=N,optimizer_steps=0,VAL_TEST_reads=0,backwards=0,old_folds_preserved=True,original_supports_only=True,opposite_component_fold_support=True,original_weights_unchanged=True,LoRA_delta_unchanged=True,gradients_absent=True,full_original_prototype_parity_maxabs=err,trainable_parameter_architecture=49152));write(out/'CENSUS_RECORDS.json',records);write(out/'CLASS_SUMMARIES.json',summaries);del m,params,frozen,z,parts,h,saved,flipped;torch.cuda.empty_cache()
 assert len(records)==18 and totalrows==87651
 engine=store();cleanup=dict(encoded_unique_flipped_rows=engine.encoded,flipped_full_visual_parity_maxabs=engine.max_abs,retained_prefix_bytes=engine.ram_bytes,temporary_swap_bytes=engine.spill_bytes,remote_swap_bytes=engine.remote_spill_bytes,science_preserved=True,TRAIN_only=True);engine.index.clear();engine.ram_bytes=0;engine.disk_index.clear();engine.view_base.clear()
 for folder in [engine.spill_root,engine.remote_spill_root]:
  for f in folder.glob('prefix_*.npy'):f.unlink()
 write(out/'REGENERABLE_CACHE_CLEANUP.json',cleanup);write(out/'CENSUS_SEAL.json',dict(records=18,flipped_endpoint_forward_rows=totalrows,saved_original_embedding_rows=totalrows,paired_classifier_policy_routes=72,view_scoring_routes=144,optimizer_steps=0,backwards=0,VAL_TEST_reads=0,no_individual_features_or_predictions_written=True));write(out/'STATUS.json',dict(time=time.time(),phase='COMPLETE'))
if __name__=='__main__':
 try:main()
 except BaseException:(out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
