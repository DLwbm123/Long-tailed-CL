import copy,json,time,resource
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import counts
from program import Dictionary,PROGRAMS,unit
from data import train_rows,loader
from common import write,guard,save_torch
from decision import ARMS,Policy,outcomes,state_tree,standardized,kl,group_loss,exact_group_constants,exact_group_loss,sample

def safe(cfg):
 guard(cfg);assert torch.cuda.max_memory_reserved()<12*2**30,'GPU_RESERVED_LIMIT'
def load_roles(cfg):
 rows,y=train_rows(cfg);roles=json.loads(Path(cfg['output'],'ROLES.private.json').read_text());sets=[set(roles[k]) for k in ['S','W','F']];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i));assert set.union(*sets)==set(range(len(rows)))
 for c in range(23):assert all(any(int(y[i])==c for i in s) for s in sets)
 return rows,y,roles

def encode_ids(cfg,base,dictionary,rows,ids,tf,program):
 out=[]
 with torch.no_grad():
  for x,j in loader(cfg,rows,ids,tf):safe(cfg);out.append(dictionary(base,x.cuda(),program).cpu().numpy())
 return np.concatenate(out)
def select_memory(cfg,ids,features,y,roles):
 lookup={i:k for k,i in enumerate(ids)};memory={}
 for c in sorted(set(int(y[i]) for i in ids)):
  parts={}
  for name in ['S','W','F']:
   pool=sorted(i for i in roles[name] if i in lookup and int(y[i])==c);z=features[[lookup[i] for i in pool]];mu=z.mean(0);mu/=max(np.linalg.norm(mu),1e-8);rank=np.argsort(((z-mu)**2).sum(1),kind='stable');parts[name]=[pool[k] for k in rank]
  chosen={name:parts[name][:cap] for name,cap in zip(['S','W','F'],[8,6,6])};left=min(20,sum(map(len,parts.values())))-sum(map(len,chosen.values()))
  for name in ['S','W','F']:
   take=min(left,len(parts[name])-len(chosen[name]));chosen[name]+=parts[name][len(chosen[name]):len(chosen[name])+take];left-=take
  assert left==0 and all(chosen.values());flat=sum(chosen.values(),[]);assert len(flat)==len(set(flat))==min(20,sum(map(len,parts.values())));memory[c]=chosen
 return memory

def fit_dictionary(cfg,base,tf,dictionary,rows,y,roles,new,basefeatures,memory):
 r=Path(cfg['output']);position={i:k for k,i in enumerate(new)};sids=[i for i in roles['S'] if int(y[i])<13];assert len(sids)==1955
 support=[i for c in range(13) for i in memory[c]['S']];labels=y[support].cuda();opt=torch.optim.AdamW(dictionary.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);cost=[]
 for epoch in range(cfg['epochs']):
  wall=time.time();cpu=time.process_time();banks=[]
  for k,program in enumerate(PROGRAMS):
   if k==0:
    features=basefeatures[[position[i] for i in support]];counts.C['NOOP_feature_cache_hits']+=len(support)
   else:features=encode_ids(cfg,base,dictionary,rows,support,tf,program)
   z=torch.tensor(features,device='cuda');banks.append(unit(torch.stack([z[labels==c].mean(0) for c in range(13)])))
  save_torch(r/f'dictionary_bank_epoch{epoch+1}.private.pt',{'epoch':epoch+1,'support_ids':support,'banks':torch.stack(banks).cpu(),'counts':dict(counts.C)})
  opt.zero_grad(set_to_none=True);seen=0;accum=0;total=0.
  for images,j in loader(cfg,rows,sids,tf,shuffle=True,seed=cfg['seed']+epoch):
   safe(cfg);images=images.cuda();yy=y[j].cuda();den=min(cfg['batch_size'],len(sids)-(seen//cfg['batch_size'])*cfg['batch_size'])
   for k,program in enumerate(PROGRAMS):
    if k==0:
     z=torch.tensor(basefeatures[[position[int(i)] for i in j]],device='cuda');counts.C['NOOP_feature_cache_hits']+=len(j)
    else:z=dictionary(base,images,program)
    counts.C['classifier_score_rows']+=len(j);loss=F.cross_entropy(20*z@banks[k].T,yy);assert torch.isfinite(loss);total+=float(loss.detach())*len(j)/16
    if loss.requires_grad:(loss*len(j)/(16*den)).backward()
   seen+=len(j);accum+=len(j)
   if accum==cfg['batch_size'] or seen==len(sids):
    opt.step();opt.zero_grad(set_to_none=True);accum=0;counts.C['formal_visual_optimizer_steps']+=1
    save_torch(r/'dictionary_active.private.pt',{'epoch':epoch+1,'seen':seen,'dictionary':dictionary.state_dict(),'optimizer':opt.state_dict(),'counts':dict(counts.C)})
  assert seen==1955
  record={'epoch':epoch+1,'rows':seen,'mean_uniform_program_CE':total/seen,'wall_seconds':time.time()-wall,'CPU_seconds':time.process_time()-cpu};cost.append(record);write(r/'DICTIONARY_COST.json',cost)
  torch.save({'epoch':epoch+1,'dictionary':dictionary.state_dict(),'optimizer':opt.state_dict(),'support_ids':support,'prototype_banks':torch.stack(banks).cpu(),'counts':dict(counts.C)},r/f'dictionary_epoch{epoch+1}.private.pt');write(r/'STATUS.json',{'phase':'DICTIONARY_FIT','epoch_complete':epoch+1,'counts':dict(counts.C)})
 assert counts.C['formal_visual_optimizer_steps']==1230
 dictionary.requires_grad_(False).eval();write(r/'DICTIONARY_SEALED.json',{'epoch':10,'snapshots':10,'VAL_reads':0,'counts':dict(counts.C)})

def all_features(cfg,stage,needed):
 r=Path(cfg['output']);pieces=[];ids=[]
 for s in range(stage+1):
  meta=json.loads((r/f'arrival_s{s}.private.json').read_text());positions=[j for j,i in enumerate(meta['ids']) if i in needed]
  if positions:
   mm=np.load(r/f'arrival_s{s}.private.npy',mmap_mode='r');pieces.append(np.array(mm[:,positions],copy=True));ids.extend(meta['ids'][j] for j in positions);del mm
 assert set(ids)==set(needed) and len(ids)==len(set(ids));counts.C['program_feature_cache_rows']+=16*len(ids)
 return np.concatenate(pieces,axis=1),{i:j for j,i in enumerate(ids)}
def score_table(features,position,support,query,n):
 proto=np.stack([np.stack([features[k,[position[i] for i in support[c]]].mean(0) for c in range(n)]) for k in range(16)]);proto/=np.maximum(np.linalg.norm(proto,axis=2,keepdims=True),1e-8)
 queryfeatures=features[:,[position[i] for i in query]];scores=np.matmul(queryfeatures,proto.transpose(0,2,1));counts.C['classifier_score_rows']+=16*len(query);return scores,proto

def context(cfg,features,pos,memory,y,birth,stage,context_index,Fids):
 n=len(memory);old=sum(map(len,cfg['stages'][:stage]));cap=[1,2,None][context_index%3] if context_index>=0 else None;offset=context_index//3 if context_index>=0 else 0;support={}
 for c in range(n):
  pool=memory[c]['S'];number=len(pool) if cap is None else min(cap,len(pool));support[c]=[pool[(offset+i)%len(pool)] for i in range(number)];assert len(set(support[c]))==number
 Wids=[i for c in range(n) for i in memory[c]['W']];wscores,proto=score_table(features,pos,support,Wids,n);wr=outcomes(wscores,y[Wids].numpy(),n,old,cfg['train_counts']);states=state_tree(wscores,y[Wids].numpy(),wr,n,old,cfg['train_counts'],birth,np.array([len(support[c]) for c in range(n)]),np.array([len(memory[c]['W']) for c in range(n)]),stage)
 scores,_=score_table(features,pos,support,Fids,n);fr=outcomes(scores,y[Fids].numpy(),n,old,cfg['train_counts']);return states,fr,wr,proto,support,Wids

def fit_policies(cfg,stage,memory,features,pos,rows,y,roles,birth,policies,optimizers):
 r=Path(cfg['output']);newclasses=cfg['stages'][stage];Fids=[i for c in range(len(memory)) for i in ( [j for j in roles['F'] if int(y[j])==c] if c in newclasses else memory[c]['F'])];logs={a:[] for a in ARMS};tables=[]
 for group in range(cfg['contexts']):
  safe(cfg);states,fr,wr,proto,support,Wids=context(cfg,features,pos,memory,y,birth,stage,group,Fids);R=fr['reward'];cost=standardized(fr['cost']);reward=standardized(R);uniforms=np.random.default_rng(cfg['seed']+1000*stage+group).random((cfg['group_size'],2));tables.append({'context':group,'states':states,'F':fr,'W':wr,'support':support,'W_ids':Wids,'F_ids':Fids})
  save_torch(r/'active_context.private.pt',tables[-1])
  for arm in ARMS:
   model=policies[arm];opt=optimizers[arm]
   with torch.no_grad():oldjoint,oldslots,_=model.tree(states);oldjoint=oldjoint.clone();oldslots=oldslots.clone()
   actions=sample(oldslots,uniforms) if arm in ['G','H'] else None;co=exact_group_constants(oldjoint,R,cfg['group_size']) if arm=='X' else None;losses=[]
   for step in range(cfg['updates_per_context']):
    write(r/'ACTIVE_POLICY.private.json',{'stage':stage,'context':group,'arm':arm,'update':step,'counts':dict(counts.C)})
    joint,slots,logjoint=model.tree(states)
    if arm=='C':objective=(joint*cost).sum()
    elif arm=='E':objective=-(joint*reward).sum()
    elif arm=='X':objective=exact_group_loss(slots,oldslots,oldjoint,co)
    else:objective=group_loss(slots,oldslots,actions,R,arm=='H')
    loss=objective+cfg['controller_kl']*kl(joint,logjoint);assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();counts.C['formal_controller_optimizer_steps']+=1;losses.append(float(loss.detach()))
   logs[arm].append({'context':group,'oldjoint':oldjoint,'oldslots':oldslots,'actions':actions,'losses':losses,'multinomial_count_cases':None if co is None else co[-1],'exact_group_constants':co,'assigned_rewards':None if actions is None else (np.roll(R[actions.numpy()],-1) if arm=='H' else R[actions.numpy()]).tolist()})
  save_torch(r/'policies_active.private.pt',{'stage':stage,'context_complete':group+1,'policy':{a:policies[a].state_dict() for a in ARMS},'optimizer':{a:optimizers[a].state_dict() for a in ARMS},'groups':logs,'counts':dict(counts.C)})
  write(r/'STATUS.json',{'phase':'POLICY_FIT','stage':stage,'context_complete':group+1,'counts':dict(counts.C)})
 states,fr,wr,proto,support,Wids=context(cfg,features,pos,memory,y,birth,stage,-1,Fids);distributions={}
 for arm in ARMS:
  with torch.no_grad():joint,_,_=policies[arm].tree(states);distributions[arm]=joint.numpy()
  torch.save({'stage':stage,'arm':arm,'policy':policies[arm].state_dict(),'optimizer':optimizers[arm].state_dict(),'groups':logs[arm],'counts':dict(counts.C)},r/f's{stage}_{arm}.private.pt')
 distributions['U']=np.ones(16)/16;distributions['NOP']=np.eye(16)[0];a=int(np.argmax(wr['reward'][[0,4,8,12]]));b=int(np.argmax(wr['reward'][4*a:4*a+4]));distributions['WG']=np.eye(16)[4*a+b]
 all_memory_proto=np.stack([features[0,[pos[i] for role in memory[c].values() for i in role]].mean(0) for c in range(len(memory))]);all_memory_proto/=np.maximum(np.linalg.norm(all_memory_proto,axis=1,keepdims=True),1e-8)
 with (r/f'memory_s{stage}.private.json').open('w') as f:json.dump({str(c):memory[c] for c in memory},f)
 torch.save({'stage':stage,'proto':proto,'all_memory_proto':all_memory_proto,'support':support,'W_ids':Wids,'F_ids':Fids,'states':states,'distributions':distributions,'context_tables':tables,'memory':copy.deepcopy(memory),'counts':dict(counts.C)},r/f'deployment_s{stage}.private.pt')

def train(cfg,base,tf,sealed_boundary=False):
 start=time.time();cpu=time.process_time();r=Path(cfg['output']);rows,y,roles=load_roles(cfg);torch.manual_seed(cfg['seed']);dictionary=Dictionary().cuda();memory={};birth=[];policies={};optimizers={}
 for a in ARMS:
  torch.manual_seed(cfg['seed']);policies[a]=Policy();optimizers[a]=torch.optim.AdamW(policies[a].parameters(),lr=cfg['controller_lr'],weight_decay=0)
 for stage,classes in enumerate(cfg['stages']):
  safe(cfg);new=[i for i,c in enumerate(y.tolist()) if c in classes]
  write(r/'STATUS.json',{'phase':'ARRIVAL','stage':stage,'counts':dict(counts.C)})
  if stage==0 and sealed_boundary:
   # Full dictionary phase and arrival cache were sealed before the accounting failure.
   audit=json.loads((r/'COUNTER_FAILURE_BOUNDARY_AUDIT.private.json').read_text());assert audit['boundary_audit_passed'] and audit['controller_updates']==audit['VAL_TEST_reads']==0
   z=torch.load(r/'dictionary_epoch10.private.pt',map_location='cpu',weights_only=False);assert z['epoch']==10
   dictionary.load_state_dict(z['dictionary'],strict=True);dictionary.requires_grad_(False).eval()
   meta=json.loads((r/'arrival_s0.private.json').read_text());assert meta['ids']==new
   cached=np.load(r/'arrival_s0.private.npy',mmap_mode='r');assert cached.shape==(16,len(new),512)
   memory.update(select_memory(cfg,new,cached[0],y,roles));assert [i for c in range(13) for i in memory[c]['S']]==z['support_ids'];del cached,z
   write(r/'BOUNDARY_RESTORED.private.json',{'dictionary_epoch':10,'new_visual_optimizer_steps':0,'new_encoder_rows':0,'controller_optimizer_restored':False,'counts':dict(counts.C)})
  else:
   basefeatures=encode_ids(cfg,base,dictionary,rows,new,tf,(0,0));memory.update(select_memory(cfg,new,basefeatures,y,roles))
   if stage==0:fit_dictionary(cfg,base,tf,dictionary,rows,y,roles,new,basefeatures,memory)
   array=np.empty((16,len(new),512),dtype=np.float32);array[0]=basefeatures
   for k,program in enumerate(PROGRAMS[1:],1):array[k]=encode_ids(cfg,base,dictionary,rows,new,tf,program)
   np.save(r/f'arrival_s{stage}.private.npy',array,allow_pickle=False);write(r/f'arrival_s{stage}.private.json',{'stage':stage,'ids':new,'labels':y[new].tolist()});del array,basefeatures
  birth.extend([stage]*len(classes))
  needed={i for c in memory for role in memory[c].values() for i in role};needed.update(i for i in roles['F'] if int(y[i]) in classes);features,pos=all_features(cfg,stage,needed)
  try:fit_policies(cfg,stage,memory,features,pos,rows,y,roles,birth,policies,optimizers)
  except BaseException:
   save_torch(r/f'failure_policies_s{stage}.private.pt',{'stage':stage,'policy':{a:policies[a].state_dict() for a in ARMS},'optimizer':{a:optimizers[a].state_dict() for a in ARMS},'counts':dict(counts.C)})
   raise
  del features,pos
 assert counts.C['formal_controller_optimizer_steps']==3840
 write(r/'ALL_STATES_SEALED.json',{'dictionary_epochs':10,'policy_snapshots':30,'deployment_states':6,'VAL_reads':0,'TEST_reads':0,'counts':dict(counts.C)})
 write(r/'TRAIN_COMPLETE.json',{'counts':dict(counts.C),'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 return dictionary
