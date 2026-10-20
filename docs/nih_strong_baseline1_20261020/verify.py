"""Frozen-checkpoint re-inference and independent counts. No training or test access."""
import time
START=time.time()
import os,json,math,resource,gc
from pathlib import Path
import numpy as np
import torch
from common import rows,Images,Model,write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());torch.set_num_threads(cfg['CPU_threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);root=Path(cfg['output']);assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE';data,audit=rows(cfg);results=json.loads((root/'RESULTS.json').read_text());curves=json.loads((root/'LEARNING_CURVES.json').read_text());assert len(results)==4 and len(curves)==120;y=np.array([z[1] for z in data['val']]);checked=[];maxdiff=0.;deadline=cfg['hard_deadline_epoch'];recount=0
for item in results:
 arm=item['arm'];history=[h for h in curves if h['arm']==arm];assert len(history)==30 and [h['epoch'] for h in history]==list(range(1,31));assert item['best_epoch']==max(range(1,31),key=lambda ep:history[ep-1]['metrics']['BA']);assert item['updates']==30*math.ceil(68058/128);backbone=item['backbone'];loader=torch.utils.data.DataLoader(Images(data['val'],backbone,False,cfg['packed_cache']),batch_size=cfg['micro_batch_size'],shuffle=False,num_workers=0);adir=root/arm
 for tag,expected in [('epoch10',10),('best',item['best_epoch']),('last',30)]:
  assert time.time()<deadline-30;cp=torch.load(adir/(tag+'.private.pt'),map_location='cpu',weights_only=False);assert cp['epoch']==expected and cp['arm']==arm;model=Model(cfg,backbone);model.load_state_dict(cp['model'],strict=True);model.cuda().eval();scores=np.empty((300,20),dtype=np.float32)
  with torch.no_grad():
   for x,_,ids in loader:
    with torch.autocast('cuda',dtype=torch.bfloat16):s=model(x.cuda())
    scores[ids.numpy()]=s.float().cpu().numpy()
  saved=np.load(adir/f'val_epoch_{expected}.private.npz');maxdiff=max(maxdiff,float(np.abs(scores-saved['scores']).max()));assert np.array_equal(scores.argmax(1),saved['scores'].argmax(1));p=scores.argmax(1);cm=np.zeros((20,20),dtype=int)
  for truth,pred in zip(y,p):cm[truth,pred]+=1
  recall=np.diag(cm)/cm.sum(1);stored=history[expected-1]['metrics'];assert np.max(np.abs(recall-stored['recall']))<1e-12 and abs(float(recall.mean())-stored['BA'])<1e-12 and np.array_equal(cm,np.array(stored['confusion']));assert int(np.diag(cm).sum())==stored['correct'];checked.append(dict(arm=arm,checkpoint=tag,epoch=expected,BA=float(recall.mean()),correct=int(np.diag(cm).sum())));recount+=300;del cp,model;gc.collect();torch.cuda.empty_cache()
 assert np.load(adir/'val_epoch_30.private.npz')['scores'].shape==(300,20)
cost=json.loads((root/'costs.json').read_text());assert cost['arms_completed']==4 and cost['optimizer_updates']==4*30*math.ceil(68058/128) and cost['train_image_exposures']==4*30*68058 and cost['val_image_exposures']==4*30*300;size=sum(p.stat().st_size for p in root.parent.rglob('*') if p.is_file());assert size<cfg['new_storage_cap_bytes']-2000000000;assert maxdiff<1e-4;result=dict(state='PASS',predictions_recomputed=recount,checkpoints_checked=checked,max_score_error=maxdiff,patient_overlap=0,updates_and_coverage_consistent=True,predeclared_checkpoint_selection_consistent=True,new_output_bytes=size,verification_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved());write(root/'VERIFICATION.json',result);print(json.dumps(result),flush=True)
