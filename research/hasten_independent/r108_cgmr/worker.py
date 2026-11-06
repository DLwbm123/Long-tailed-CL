"""Native ->48-root global gate -> conditional180-endpoint trial, never a retry launcher."""
import os,json,time,csv,gc,traceback,copy
from pathlib import Path
import torch
from common import write,guard
from memory import from_views
from roles import Roles
from encoder import Encoder
from preflight import run as preflight
from pilot import run as pilot
from training import run as training
from constants import SEEDS
c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output']);counter={};cpu=time.process_time();wall=time.time();root=Path(c['root_output'])
def wait(ok,fail=None):
 while not (root/ok).exists():
  if fail and (root/fail).exists():return False
  guard(c);time.sleep(2)
 return True
try:
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert json.loads((root/'SOFTWARE_CHECK.json').read_text())['passed'];rows=list(csv.DictReader((Path(c['manifest'])/'train.csv').open()));assert len(rows)==6821;y=torch.tensor([c['order'].index(int(x['original_label'])) for x in rows]);assert torch.bincount(y,minlength=23).tolist()==c['train_counts'];comp={}
 for row in rows:
  k=row['identity_component'];comp.setdefault(k,row['original_label']);assert comp[k]==row['original_label'],'CROSS_LABEL_COMPONENT_REQUIRES_GLOBAL_REGISTRY_BINDING_REVIEW'
 first=[i for i in range(len(rows)) if int(y[i])<5];encoder=Encoder(c);write(r/'STATUS.json',{'phase':'NATIVE_FULL_A_B_ADMISSION','counts':counter});passed=preflight(c,encoder,rows,y,first,counter);write(r/'NATIVE_DONE.json',{'passed':passed,'counts':counter})
 if not wait('GLOBAL_NATIVE_ADMISSION.json','GLOBAL_NATIVE_FAIL.json'):write(r/'STATUS.json',{'phase':'NOT_ADMITTED_FULL_PLAN','counts':counter})
 else:
  roles=Roles();roles.arrive(first,rows);write(r/'ROLES.private.json',roles.role);write(r/'A_STARTED.private.json',{'start':time.time(),'original_start':c['start'],'deadline':c['deadline'],'actor_optimizer':0});write(r/'STATUS.json',{'phase':'A_FIRST_ARRIVAL_ENCODING','counts':counter});before=counter.get('TRAIN_encoder_rows',0);z=encoder.encode(c,rows,first,counter,'TRAIN',first);counter['A_encoder_rows']=counter['TRAIN_encoder_rows']-before;data=from_views(z,torch.tensor(first,device='cuda'),y[first].cuda());del z;gate=pilot(c,data,roles,rows,c['train_counts'][:5],counter);del data;gc.collect();torch.cuda.empty_cache()
  if not wait('GLOBAL_A_PASS.json','GLOBAL_A_FAIL.json'):write(r/'STATUS.json',{'phase':'COMPLETE_A_PRECHECK_NEGATIVE','counts':counter})
  else:
   write(r/'B_STARTED.private.json',{'start':time.time(),'fresh_seeds':SEEDS,'A_weights_carried':False});training(c,encoder,rows,y,roles,counter);write(r/'STATUS.json',{'phase':'WAIT_GLOBAL180_SEAL','counts':counter});wait('GLOBAL_LEARNING_SEAL.json')
   import evaluate
   vrows=list(csv.DictReader((Path(c['manifest'])/'val.csv').open()));assert len(vrows)==1698;z=encoder.encode(c,vrows,list(range(1698)),counter,'VAL')
   for seed in SEEDS:
    sc=copy.deepcopy(c);sc.update(seed=seed,output=str(r/f'seed{seed}'));evaluate.run(sc,encoder,counter,z)
   del z;write(r/'STATUS.json',{'phase':'COMPLETE_B','counts':counter})
 write(r/'WORKER_COMPLETE.json',{'counts':counter,'core_CPU_seconds':time.process_time()-cpu,'core_wall_seconds':time.time()-wall})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'FAILURE_COUNTS.json',counter);write(r/'STATUS.json',{'phase':'FAILED_PRESERVED','counts':counter});raise
finally:write(r/'COSTS.json',{'counts':counter,'core_CPU_seconds':time.process_time()-cpu,'core_wall_seconds':time.time()-wall,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'native_profile_included_once_in_worker':True,'preparation_import_SSH_document_delivery_CPU':'nonzero not fully sampled'})
