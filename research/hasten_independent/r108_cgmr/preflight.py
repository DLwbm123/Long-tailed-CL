"""Worst native shapes and full A+B admission before any scientific root."""
import time,json,os,subprocess,gc,tempfile
from pathlib import Path
import torch
from memory import from_views
from selection import FixedSelect
from proposals import generate
from features import describe
from policy import Head,update,behavior
from roles import Roles
from shadow import episode,root,feedback,gen
from learner import Learner,optimizer,learn,frozen
from common import write,guard

def run(c,encoder,rows,y,first,counter):
 r=Path(c['output']);ids=[]
 for cl in range(5):
  group=set()
  for i in first:
   if int(y[i])==cl and rows[i]['identity_component'] not in group:ids.append(i);group.add(rows[i]['identity_component'])
   if len(group)==6:break
 assert len(ids)==30;times=[]
 for _ in range(2):
  torch.cuda.synchronize();t=time.perf_counter();z=encoder.encode(c,rows,ids,counter,'native',first);torch.cuda.synchronize();times.append((time.perf_counter()-t)/30)
 N=23*48;zz=z[torch.arange(N,device='cuda')%30];fake=from_views(zz,torch.arange(N,device='cuda'),torch.arange(N,device='cuda')//48);rr=[];roles=Roles()
 for i in range(N):
  cl=i//48;j=i%48;role='S' if j<32 else 'FIT' if j<40 else 'AUDIT';component=f'fixture{cl}_{role}_{j if role=="S" else j//2}';rr.append({'identity_component':component,'original_label':str(cl)});roles.role[component]=role
 e,skip=episode(fake,roles,rr,[1000]*23,74002,2,native_max=True);assert skip is None and e['K']==23 and e['K0']==21;group=[];actor_seconds=[];last=None
 for repeat in range(2):
  torch.cuda.synchronize();t=time.perf_counter();ctx=root(e,counter,'native',c,native=True);assert ctx['actual_B']==256 and len(ctx['mems'])==9,'NATIVE_REQUIRES_NINE_DISTINCT_CANDIDATES';scores,raw=feedback(e,ctx,counter,'native',c)
  astart=time.perf_counter()
  for kind in ['CGMR_MAIN','REWARD_SHUFFLED','SUP_GREEDY']:
   head=Head(74002).cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);update(head,opt,ctx['x'],scores['FIT']['R'],kind,gen(8108,'cuda'),counter,'native');_,last=behavior(head,ctx['x'],kind,8109)
  torch.cuda.synchronize();actor_seconds.append(time.perf_counter()-astart);group.append(time.perf_counter()-t)
 # Actual native classifier+nonempty optimizer strict roundtrip, before global admission.
 m=ctx['m'];o=ctx['o'];xx,_,_=ctx['mems'][0].bags()
 with tempfile.TemporaryDirectory(dir=r) as d:
  p=Path(d)/'restore.pt';torch.save({'model':m.state_dict(),'opt':o.state_dict()},p);pack=torch.load(p,weights_only=False);other=Learner(74002).cuda();other.C=m.C;other.load_state_dict(pack['model'],strict=True);oo=optimizer(other);oo.load_state_dict(pack['opt'])
  with torch.no_grad():assert torch.equal(m(xx[:4]),other(xx[:4]))
  assert ctx['mems'][0].serialize(Path(d)/'max_memory.bin',e['counts'][:m.C],e['birth'][:m.C])==528568
  for a,b in zip(o.state.values(),oo.state.values()):
   for k,v in a.items():
    if torch.is_tensor(v):assert torch.equal(v,b[k])
 assert not any(p.requires_grad or p.grad is not None for p in encoder.visual.parameters())
 N=c['maximum_stage_parent_pool'];fp=from_views(z[torch.arange(N,device='cuda')%30],torch.arange(N,device='cuda'),torch.arange(N,device='cuda')%23);m=Learner(74002).cuda();m.arrive(fp,23);o=optimizer(m);teacher=frozen(m);step=[]
 for _ in range(2):
  torch.cuda.synchronize();t=time.perf_counter();learn(m,o,fp,4,74002,counter,teacher,64,'native');torch.cuda.synchronize();step.append((time.perf_counter()-t)/4)
 costs=[];proposals={}
 for B in [64,128,256]:
  for repeat in range(2):
   torch.cuda.synchronize();t=time.perf_counter();base,cover,info=FixedSelect(fp,23,B).choose('BALANCED',gen(74002,'cuda'));ms,es,au=generate(fp,base,cover,gen(74003,'cuda'),counter);x,fa=describe(fp,ms,es,m,[N//23+1]*23,[0]*23,4,fp.parents.tolist());torch.cuda.synchronize();elapsed=time.perf_counter()-t
  costs.append(elapsed);proposals[str(B)]={'seconds_second':elapsed,'candidates':len(ms),'max_shape_parent_pool':N}
 # Includes cost of each root, selection/features on largest pool, all formal updates, all encoding.
 first_images=len(first);encoder_images=first_images+3*6821+1698
 estimate=group[1]*(24+720)+max(costs)*(24+720+90)+step[1]*27000+times[1]*encoder_images+600
 projection=2*estimate+7200;gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);own=any(s.split(',')[0].strip()==str(os.getpid()) and c['GPU_UUID'] in s for s in gpu.splitlines());assert own
 passed=projection<c['deadline']-time.time() and torch.cuda.max_memory_reserved()<12*2**30
 out={'passed':passed,'own_physical_device_verified':True,'max_native_C':23,'root_C0':21,'root_B':256,'root_distinct_candidates':9,'native_distinct_first_TRAIN_images':30,'native_encoder_passes':2,'repeated_features_synthetic_labels_only_engineering_fixture':True,'native_fixture_forced_BALANCED_base_only_to_profile_nine_action_maximum':True,'native_complete_root_seconds_second':group[1],'native_three_heads_two_updates_seconds_second':actor_seconds[1],'formal_classifier_step_seconds_second':step[1],'max_pool_selection_feature_profiles':proposals,'sixview_encoder_seconds_per_parent_second':times[1],'A_plus_B_roots_per_stream_upper':744,'A_plus_B_classifier_optimizer_per_stream_upper':315672,'per_stream_formal_classifier_updates':27000,'safety':2,'reserve_seconds':7200,'projection_seconds_per_stream':projection,'remaining_original_clock_seconds':c['deadline']-time.time(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'frozen_encoder_no_grad':True,'native_classifier_optimizer_strict_restore':True,'native_counts':dict(counter),'last_synthetic_behavior_without_probabilities':{k:v for k,v in last.items() if k!='probability'},'no_scientific_forward_optimizer_VAL_TEST_yet':True,'all_fixture_parameters_discarded':True}
 write(r/'BUDGET.json',out);del fake,ctx,fp,ms,x,head,opt,m,o,teacher,other,oo,xx,pack,raw,scores;gc.collect();torch.cuda.empty_cache();guard(c);return passed
