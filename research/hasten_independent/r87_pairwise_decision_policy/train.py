import os,json,time,torch,subprocess
from pathlib import Path
from source import load_stage,subset
from decision import inputs,outcomes,Policy,uniform,joint,summary
from fit import optimize
from common import guard,write

def main(cfg):
 r=Path(cfg['output']);torch.set_num_threads(4);states={};snapshots=[];steps=0;fitcost=[];context_count=0
 def born(fold,direction,stream,arm):
  key=(fold,direction,stream,arm)
  if key not in states:
   torch.manual_seed(74002+1000000+100*fold+10*direction+stream);m=Policy();o=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=0);states[key]=(m,o)
  return states[key]
 for stage in range(6):
  for fold in range(2):
   v=load_stage(cfg,stage,fold)
   for direction,tag in enumerate(['Qa','Qb']):
    guard(cfg);ix=subset(v,tag);ctx=inputs(v['q']['arms']['NCM']['logits'][ix],v['proto'],v['support'],v['ages']);out=outcomes(ctx,v['q']['labels'][ix],v['eligible'],v['protected']);assert all(out['count'][c]>0 for c in v['eligible'])
    context_name=f'context_s{stage}f{fold}d{direction}.private.pt';torch.save(dict(stage=stage,fold=fold,direction=direction,train_tag=tag,ids=[i for i,b in zip(v['q']['ids'],ix) if b],actions=ctx['actions'],features=ctx['x1'][:,:11],mask=ctx['mask'],outcome=out),r/context_name);context_count+=1
    for stream in range(cfg['streams']):
     for arm in ['CE','E','G','H']:
      guard(cfg);write(r/'STATUS.json',dict(phase='POLICY_FIT',stage=stage,fold=fold,direction=direction,stream=stream,arm=arm,formal_optimizer_steps=steps));m,opt=born(fold,direction,stream,arm);cpu=time.process_time();wall=time.time();f=optimize(m,opt,ctx,out,arm,cfg,74002+2000000+10000*stage+100*fold+10*direction+stream);steps+=f['steps'];name=f'policy_s{stage}f{fold}d{direction}r{stream}_{arm}.private.pt'
      torch.save(dict(stage=stage,fold=fold,direction=direction,stream=stream,arm=arm,model_state=m.state_dict(),optimizer_state=opt.state_dict(),fit=f,training_context=context_name,cumulative_stage_optimizer_steps=(stage+1)*cfg['groups']*cfg['epochs']),r/name);snapshots.append(name);fitcost.append(dict(stage=stage,fold=fold,direction=direction,stream=stream,arm=arm,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-wall,optimizer_steps=f['steps'],legal_actions=len(ctx['actions']),legal_trajectories=int(ctx['mask'].sum()),distribution_evaluations=cfg['groups']*cfg['epochs']+1+(cfg['groups'] if arm in ['G','H'] else 0)))
 assert len(snapshots)==288 and steps==36864 and context_count==24
 write(r/'ALL_POLICIES_SEALED.json',dict(policy_snapshots=288,contexts=24,formal_controller_optimizer_steps=steps,heldout_labels_scored=0,visual_optimizer_steps=0,encoder_classifier_forward_rows=0,VAL_TEST_reads=0,time=time.time()));write(r/'FIT_COST.json',fitcost)
 # No held-out outcome evaluation occurs above this seal.
 records=[];counter=0;eval_calls=0
 for stage in range(6):
  for fold in range(2):
   v=load_stage(cfg,stage,fold)
   for direction,tag in enumerate(['Qb','Qa']):
    for evaltag in [tag,'full']:
     guard(cfg);ix=subset(v,evaltag);ctx=inputs(v['q']['arms']['NCM']['logits'][ix],v['proto'],v['support'],v['ages']);eligible=list(range(v['n'])) if evaltag=='full' else v['eligible'];prot=[c for c in eligible if c<sum(map(len,cfg['stages'][:stage])) or cfg['train_counts'][c]<250];out=outcomes(ctx,v['q']['labels'][ix],eligible,prot);tail=[a<250 for a in cfg['train_counts'][:v['n']]];counter+=int(ctx['mask'].sum())*len(out['eligible'])
     # Save finite counterfactual basis, not dense per-trajectory per-example logits.
     torch.save(dict(stage=stage,fold=fold,direction=direction,query=evaltag,ids=[i for i,b in zip(v['q']['ids'],ix) if b],actions=ctx['actions'],features=ctx['x1'][:,:11],mask=ctx['mask'],outcome=out),r/f'eval_s{stage}f{fold}d{direction}_{evaltag}.private.pt')
     refs={}
     for ref,key in [('NCM','NCM'),('balanced','balanced')]:
      pred=v['q']['arms'][key]['prediction'][ix];yy=v['q']['labels'][ix];cnt=torch.bincount(yy,minlength=v['n']).double();cor=torch.bincount(yy[pred==yy],minlength=v['n']).double();cc=[c for c in eligible if cnt[c]>0];tc=[c for c in cc if tail[c]];refs[ref]=dict(counts=cnt.tolist(),expected_correct=cor.tolist(),modal_correct=cor.tolist(),expected_macro=float((cor[cc]/cnt[cc]).mean()*100),expected_tail=float((cor[tc]/cnt[tc]).mean()*100),modal_macro=float((cor[cc]/cnt[cc]).mean()*100),modal_tail=float((cor[tc]/cnt[tc]).mean()*100))
     for stream in range(cfg['streams']):
      result=dict(refs);result['U']=summary(ctx,out,uniform(ctx),tail);saved={}
      for arm in ['CE','E','G','H']:
       h=torch.load(r/f'policy_s{stage}f{fold}d{direction}r{stream}_{arm}.private.pt',map_location='cpu',weights_only=False);m=Policy();m.load_state_dict(h['model_state']);m.eval()
       with torch.no_grad():p=m.distribution(ctx)
       eval_calls+=1;result[arm]=summary(ctx,out,p,tail);saved[arm]=dict(p1=p[0],p2=p[1],joint=joint(p))
      torch.save(saved,r/f'distributions_s{stage}f{fold}d{direction}r{stream}_{evaltag}.private.pt');records.append(dict(stage=stage,fold=fold,direction=direction,stream=stream,query='heldout' if evaltag==tag else 'full',heldout_tag=tag,arms=result))
 write(r/'RECORDS.private.json',records);write(r/'TRAIN_COMPLETE.json',dict(policy_snapshots=288,formal_controller_optimizer_steps=steps,policy_fit_cells=288,records=len(records),heldout_policy_distribution_evaluations=eval_calls,shared_counterfactual_class_trajectory_evaluations=counter,encoder_classifier_forward_rows=0,visual_optimizer_steps=0,VAL_TEST_reads=0,fullQ_is_descriptive=True,time=time.time()))
if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
