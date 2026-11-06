"""Six independent real histories; own-pool roots, fresh networks after pilot gate."""
import copy,json,time,gc
from pathlib import Path
import torch
from common import guard,write,save
from memory import from_views,join,BUDGET_BYTES
from learner import Learner,optimizer,learn,frozen
from selection import FixedSelect
from proposals import generate,key
from features import describe
from policy import Head,update,behavior
from shadow import episode,root,feedback,gen
from constants import ARMS,LEARNED,SEEDS

def run(c,encoder,rows,y,roles,counter):
 r=Path(c['output']);all_audit=[]
 for seed in SEEDS:
  sr=r/f'seed{seed}';sr.mkdir();models={a:Learner(seed).cuda() for a in ARMS};opts={a:optimizer(m) for a,m in models.items()};memories={a:None for a in ARMS};heads={a:Head(seed).cuda() for a in LEARNED};ho={a:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001) for a,h in heads.items()};counts=[];birth=[];seed_before=counter.get('formal_classifier_optimizer',0)
  for t,classes in enumerate(c['stages']):
   guard(c);folder=sr/f's{t}';folder.mkdir();lo=len(counts);C=lo+len(classes);ids=[i for i in range(len(rows)) if lo<=int(y[i])<C];roles.arrive(ids,rows);write(r/'ROLES.private.json',roles.role);write(r/'STATUS.json',{'phase':'B_CURRENT_ENCODING','seed':seed,'stage':t,'counts':counter});z=encoder.encode(c,rows,ids,counter,'TRAIN',ids);data=from_views(z,torch.tensor(ids,device='cuda'),y[ids].cuda());del z;counts.extend([int((y==cl).sum()) for cl in classes]);birth.extend([t]*len(classes));stage_meta=[]
   for arm in ARMS:
    guard(c);m=models[arm];o=opts[arm];previous=memories[arm];pool=join(previous,data);assert set(pool.pid.tolist())<=set(ids)|(set(previous.pid.tolist()) if previous is not None else set())
    if arm in LEARNED:
     for j in range(16):
      rootseed=seed*100000+t*10000+j*101+9100;e,skip=episode(pool,roles,rows,counts,rootseed,j)
      if skip:stage_meta.append(dict(arm=arm,root=j,**skip));continue
      ctx=root(e,counter,'B_shadow',c);scores,raw=feedback(e,ctx,counter,'B_shadow',c);up=update(heads[arm],ho[arm],ctx['x'],scores['FIT']['R'],arm,gen(rootseed+60000,'cuda'),counter,'B');_,bh=behavior(heads[arm],ctx['x'],arm,rootseed+70000);record={'candidate_keys':ctx['candidate_keys'],'anchor_keys':ctx['anchor_keys'],'arm':arm,'root':j,'candidate_count':len(ctx['mems']),'coverage':e['coverage'],'actual_B':ctx['actual_B'],'C0':ctx['C0'],'maximum_age':e['prefix_phases']-1,'edits':ctx['edits'],'proposal_audit':ctx['proposal_audit'],'scores':scores,'update':up,'behavior':bh};stage_meta.append(record);write(folder/'META.private.json',stage_meta);write(r/'ACTIVE_COUNTS.private.json',{'phase':'B_SHADOW','seed':seed,'stage':t,'arm':arm,'root':j,'counts':counter});del e,ctx,scores,raw
    teacher=frozen(m) if m.C else None;m.arrive(data,C);before=counter.get('formal_classifier_optimizer',0);learn(m,o,pool,200,seed*100000+t*100+1,counter,teacher,64,'formal')
    base,cover,info=FixedSelect(pool,C,256).choose('BALANCED' if arm=='BALANCED' else 'ZERO',gen(seed*100000+t*100+30000,'cuda'))
    if arm in ['BASE_ZERO','BALANCED']:
     M=base;choice=0;proposal_count=1;edits=[{'type':'NOOP','operations':0,'removed_tokens':0,'replaced_parents':0}];bh={'noop_probability':1.,'KL_vs_prior':0.,'TV_vs_prior':0.,'candidate_entropy':0.,'draw32_counts':[32]}
    else:
     cand,edits,pa=generate(pool,base,cover,gen(seed*100000+t*100+31000,'cuda'),counter);x,fa=describe(pool,cand,edits,m,counts,birth,t,ids);choice,bh=behavior(heads.get(arm),x,arm,seed*100000+t*100+32000,gen(seed*100000+t*100+33000,'cuda'));M=cand[choice];proposal_count=len(cand);del cand,x,fa,pa
    learn(m,o,M,100,seed*100000+t*100+2,counter,teacher,64,'formal');assert counter['formal_classifier_optimizer']-before==300;bs=M.serialize(folder/f'{arm}_MEMORY.private.bin',counts,birth);memories[arm]=M
    save(folder/f'{arm}_MODEL.private.pt',{'model':m.state_dict(),'optimizer':o.state_dict(),'seen':C,'stage':t,'arm':arm,'observed_counts':counts.copy(),'birth':birth.copy(),'memory_bytes':bs,'formal_updates':300})
    restored=Learner(seed).cuda();pk=torch.load(folder/f'{arm}_MODEL.private.pt',weights_only=False);restored.C=C;restored.load_state_dict(pk['model'],strict=True);oo=optimizer(restored);oo.load_state_dict(pk['optimizer']);xx,_,_=M.bags()
    with torch.no_grad():assert torch.equal(m(xx[:2]),restored(xx[:2]))
    assert all(torch.isfinite(v).all() for v in restored.state_dict().values())
    record={'seed':seed,'stage':t,'arm':arm,'seen':C,'memory_bytes':bs,'retained_parents':len(M.parents),'global_tokens':len(M.gidx),'local_tokens':len(M)-len(M.gidx),'selected_candidate':choice,'candidate_count':proposal_count,'selected_edit':edits[choice],'behavior':bh,'old_views_from_own_previous_memory_only':True,'strict_restore':True,'per_class_parent_coverage':torch.bincount(M.y[M.gidx],minlength=C).cpu().tolist()};all_audit.append(record);write(r/'MEMORY_AND_ACCESS_AUDIT.private.json',all_audit);write(r/'ACTIVE_COUNTS.private.json',{'phase':'B_FORMAL','seed':seed,'stage':t,'arm':arm,'counts':counter});del teacher,restored,oo,pk,xx,pool
   save(folder/'POLICY.private.pt',{'heads':{k:h.state_dict() for k,h in heads.items()},'optimizers':{k:o.state_dict() for k,o in ho.items()},'seed':seed,'stage':t});write(folder/'STAGE_SEAL.json',{'classifier_endpoints':6,'policy_heads':3,'seed':seed,'stage':t,'VAL_TEST':0});del data;gc.collect();torch.cuda.empty_cache()
  assert counter['formal_classifier_optimizer']-seed_before==9000;write(sr/'LEARNING_SEAL.json',{'classifier_endpoints':30,'policy_heads':15,'formal_classifier_updates':9000,'VAL_TEST':0});del models,opts,memories,heads,ho;gc.collect();torch.cuda.empty_cache()
 write(r/'B_LEARNING_SEAL.json',{'classifier_endpoints':90,'policy_heads':45,'formal_classifier_updates':27000,'counts':counter,'VAL_TEST':0})
