"""Full TRAIN-only census of saved endpoints; does not instantiate an optimizer."""
from pathlib import Path
import os,json,time,torch,traceback
from common import write,guard
from data import train_rows,get_cache,Tokens,DataLoader,store
from folds import assign_folds
from vision import NativeTail,adapt_native_tail,load_delta,delta_state
from diagnostics import summarize
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());rc=json.loads(Path(cfg['root_config']).read_text());root=Path(rc['output']);out=Path(cfg['output']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
 rows,y=train_rows(cfg);counts=torch.bincount(y,minlength=len(cfg['order']));bundle=torch.load(root/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);paths=[];records=[];summaries=[];oldfolds={};totalrows=0
 with torch.no_grad():
  for stage,classes in enumerate(cfg['stages']):
   write(out/'STATUS.json',dict(time=time.time(),phase='ARRIVED_TRAIN_CACHE',stage=stage));paths.extend(get_cache(rc,cfg,stage=stage));data=Tokens(paths);assert data.split=='train';N=len(data);C=sum(map(len,cfg['stages'][:stage+1]));nc=counts[:C];assert torch.equal(torch.bincount(data.labels,minlength=C),nc)
   folds=assign_folds(data.labels.tolist(),[rows[i]['identity_component'] for i in data.ids]);fm=dict(zip(data.ids,folds));assert all(fm[i]==b for i,b in oldfolds.items());oldfolds=fm;fold=torch.tensor(folds)
   for name in ['sft_reference','tars_grpo','macro_grpo']:
    write(out/'STATUS.json',dict(time=time.time(),phase='TRAIN_ENDPOINT_FORWARD',stage=stage,model=name,completed_records=len(records),model_TRAIN_forward_rows=totalrows))
    source=Path(rc['matched_reference_run'])/'HK/74002' if name=='sft_reference' else Path(rc['r75_run'])/'HK'/name/'74002'
    h=torch.load(source/f'head{stage}.private.pt',map_location='cpu',weights_only=False);assert h['seen']==C and h['train_counts'].tolist()==nc.tolist();m=NativeTail(bundle).cuda().eval();params=adapt_native_tail(m);load_delta(m,h['delta']);frozen={k:p.detach().cpu().clone() for k,p in m.named_parameters() if not p.requires_grad};parts=[]
    for tokens,base,yy,ix in DataLoader(data,batch_size=64):
     guard(cfg);z=m(tokens.cuda());assert z.shape==(len(yy),512) and torch.isfinite(z).all();parts.append(z.cpu());totalrows+=len(yy)
     assert torch.cuda.max_memory_reserved(0)<12*2**30 and torch.cuda.max_memory_reserved(1)<12*2**30
    z=torch.cat(parts);classes_stats,moments=summarize(z,data.labels,fold,C);err=float((moments['prototypes'].float()-h['prototypes']).abs().max());assert err<=2e-5
    assert all(torch.equal(p.detach().cpu(),frozen[k]) for k,p in m.named_parameters() if not p.requires_grad);assert all(torch.equal(delta_state(m)[k],v) for k,v in h['delta'].items()) and all(p.grad is None for p in params)
    file=out/name/f'moments{stage}.private.pt';file.parent.mkdir(parents=True,exist_ok=True);torch.save(moments,file)
    summaries.append(dict(stage=stage,model=name,seed=74002,seen=C,classes=classes_stats,scope='TRAIN endpoints, not independent holdout; hypothetical snapshot h, not historical oldpolicy'))
    records.append(dict(stage=stage,model=name,seed=74002,seen=C,TRAIN_rows=N,counts=nc.tolist(),optimizer_steps=0,VAL_TEST_reads=0,backwards=0,old_folds_preserved=True,opposite_component_fold_support=True,original_weights_unchanged=True,LoRA_delta_unchanged=True,gradients_absent=True,full_prototype_parity_maxabs=err,trainable_parameter_architecture=49152));write(out/'CENSUS_RECORDS.json',records);write(out/'CLASS_SUMMARIES.json',summaries);del m,params,frozen,z,parts,h,moments;torch.cuda.empty_cache()
   del data
 assert len(records)==18 and totalrows==87651
 engine=store();cleanup=dict(encoded_rows=engine.encoded,reference_max_abs=engine.max_abs,retained_prefix_bytes=engine.ram_bytes,temporary_swap_bytes=engine.spill_bytes,remote_swap_bytes=engine.remote_spill_bytes,science_preserved=True,TRAIN_only=True);engine.index.clear();engine.ram_bytes=0;engine.disk_index.clear()
 for folder in [engine.spill_root,engine.remote_spill_root]:
  for f in folder.glob('prefix_*.npy'):f.unlink()
 write(out/'REGENERABLE_CACHE_CLEANUP.json',cleanup);write(out/'CENSUS_SEAL.json',dict(records=18,TRAIN_endpoint_embedding_rows=totalrows,TRAIN_scoring_routes=36,optimizer_steps=0,backwards=0,VAL_TEST_reads=0,private_moments_preserved=True));write(out/'STATUS.json',dict(time=time.time(),phase='COMPLETE'))
if __name__=='__main__':
 try:main()
 except BaseException:(out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
