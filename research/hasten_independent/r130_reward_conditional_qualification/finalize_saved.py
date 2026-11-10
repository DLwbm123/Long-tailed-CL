"""CPU-only saved verification; zero new model/feature/generation calls."""
import json,time,csv,re
from pathlib import Path
import torch
from .core import ARMS,write,outcome,fixtures

def run(cfg):
 begin=time.time();cpu=time.process_time();p=Path(cfg['output']);exit=json.loads((p/'CONTROL_EXIT.json').read_text());assert exit['exit_code']==0 and exit['worker']['exit_code']==0
 curves=json.loads((p/'TRAINING_CURVES.json').read_text());evals=json.loads((p/'EVALUATIONS.json').read_text());cost=json.loads((p/'COSTS.json').read_text());count=cost['counts'];assert len(curves)==160 and len(evals)==6
 train=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()));ys=[int(x['original_label']) for x in train];vy=[int(x['original_label']) for x in val]
 names={int(x['original_label']):x['official_class_name'].replace('-', ' ') for x in train}
 def normalized(x):return re.sub(r'[\s_\-]+','',x.lower().strip())
 tape=torch.load(p/'QUERY_TAPE.private.pt',map_location='cpu',weights_only=True);current=[i for i,c in enumerate(ys) if c in range(5,10)];g=torch.Generator().manual_seed(12474006);expected=torch.tensor(current)[torch.randint(len(current),(32,8),generator=g)];assert torch.equal(tape,expected)
 for arm in ARMS:
  rows=[r for r in curves if r['arm']==arm];assert [r['update'] for r in rows]==list(range(32));sigma=1.
  for k,r in enumerate(rows):
   z=torch.load(p/f'ROLLOUT_{arm}_{k:03d}.private.pt',map_location='cpu',weights_only=True);assert z['labels'].tolist()==[ys[i] for i in tape[k].tolist()]
   b=z['b'];m=z['m'];gold=[normalized(names[ys[i]]) for i in tape[k].tolist() for _ in range(8)];actual=torch.tensor([float(normalized(t)==y and bool(d)) for t,y,d in zip(z['text'],gold,z['done'])]).reshape_as(b);assert torch.equal(actual,b)
   drift=(((z['old']-z['ref'])*m).sum(-1)/m.sum(-1).clamp_min(1)).clamp_min(0);h=torch.exp(-20*drift).reshape_as(b);reward=b+.5*h;ns=.999*sigma+.001*float(reward.std(unbiased=False))
   assert torch.allclose(z['h'],h,atol=1e-6,rtol=1e-5) and abs(ns-r['sigma_after'])<1e-7 and abs(sigma-r['sigma_before'])<1e-9
   assert torch.allclose(z['a'],(reward-reward.mean(-1,keepdim=True))/(ns+1e-6),atol=2e-6) and torch.allclose(z['ac'],(b-b.mean(-1,keepdim=True))/(ns+1e-6),atol=2e-6)
   bound_classes=json.loads((p/'PROMPT_BINDINGS.json').read_text())['1']['classes'];assert bound_classes==list(range(10))
   assert int(b.sum())==r['correct_responses'] and r['nonzero_update'] and r['parameter_L2']>0;sigma=ns
  state=torch.load(p/f'STATE_1_{arm}.private.pt',map_location='cpu',weights_only=True);assert state['updates']==32 and {int(x['step']) for x in state['optimizer']['state'].values()}=={32} and abs(state['sigma']-sigma)<1e-8
 classes=[]
 for r in evals:
  z=torch.load(p/f'VAL_1_{r["arm"]}.private.pt',map_location='cpu',weights_only=True);ids=z['ids'];assert ids==[i for i,c in enumerate(vy) if c in range(10)] and len(ids)==964
  correct=[0]*23;den=[0]*23
  for i,pred in zip(ids,z['predictions']):den[vy[i]]+=1;correct[vy[i]]+=int(pred==vy[i])
  assert correct==r['correct'] and den==r['denominator'];assert abs(sum(correct[c]/den[c] for c in range(10))/10-r['macro'])<1e-12
  counts=[sum(x==c for x in ys) for c in range(23)]
  for key,subset in [('head',[c for c in range(10) if counts[c]>=250]),('tail',[c for c in range(10) if counts[c]<250]),('tiny',[c for c in range(10) if counts[c]<25])]:
   expected_metric=sum(correct[c]/den[c] for c in subset)/len(subset) if subset else None
   assert (r[key] is None and expected_metric is None) or abs(r[key]-expected_metric)<1e-12
  assert abs(sum(correct)/sum(den)-r['micro'])<1e-12
  for c in range(10):classes.append({'arm':r['arm'],'class_id':c,'correct':correct[c],'denominator':den[c],'accuracy':correct[c]/den[c]})
 seal=json.loads((p/'STAGE_1_ALL_MODELS_SEALED.json').read_text())
 for r in evals:
  if r['arm'] in ARMS:assert r['evaluation_started_epoch']>=seal['epoch']
 access=json.loads((p/'ACCESS.private.json').read_text())
 for r in access:
  labels=ys if r['split']=='TRAIN' else vy;allowed=set(range(5) if r['stage']==0 else range(5,10)) if r['split']=='TRAIN' else set(range(5) if r['stage']==0 else range(10));assert all(labels[i] in allowed for i in r['ids'])
  if r['split']=='VAL':assert r['all_stage_models_sealed'] and r['purpose']=='isolated_audit'
 prior_cost=json.loads((p/'attempt_1/COSTS.json').read_text()) if (p/'attempt_1/COSTS.json').exists() else None
 expected_current=18584 if prior_cost else 18680
 assert count['generation_responses']==expected_current and count['post_RL_responses']==10240 and count['probe_responses']==64 and count['menu_responses']==2496 and count['VAL_responses']==5784 and count['engineering_responses']==(0 if prior_cost else 96)
 assert count['post_completed_updates']==count['post_optimizer_attempts']==160 and count['engineering_optimizer_attempts']==(0 if prior_cost else 5) and count['probe_optimizer_attempts']==1 and count['qualification_CE_queries']==256 and count['SFT_updates']==0 and count['TRAIN_encoder_rows']==(0 if prior_cost else 4) and count['VAL_encoder_rows']==0
 assert json.loads((p/'GRADIENT_PROBE.json').read_text())['counterfactual_discarded_actor_anchor_restored']
 result=outcome(evals,[sum(x==c for x in ys) for c in range(23)]);result['all_arms_32_nonzero_steps']=True;result['qualification_passed']=result['performance_qualification_passed'];write(p/'FINAL_RESULTS.json',result)
 with (p/'CLASS_RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)
 proof={'passed':True,'new_model_calls':0,'response_counts_exact_this_attempt':count['generation_responses'],'all_attempt_generated_responses':count['generation_responses']+(prior_cost['counts']['generation_responses'] if prior_cost else 0),'all_five_optimizer_steps':32,'same_tape_current_TRAIN_access_and_seals':True,'independent_reward_CTAN_recomputation':True,'class_rows':len(classes),'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-begin,'fixtures':fixtures()};write(p/'FINAL_VERIFICATION.json',proof)
 report=['# R130 单转换机制资格结果','',f'资格门通过 {sum(result["gates"].values())}/{len(result["gates"])}；非原完整九门。','', '| arm | macro % | tail % | old % | new % |','|---|---:|---:|---:|---:|']
 for a in ARMS:
  r=next(r for r in evals if r['arm']==a);oldnew=result['old_new'][a];report.append(f'| {a} | {100*r["macro"]:.4f} | {100*r["tail"]:.4f} | {100*oldnew["old_macro"]:.4f} | {100*oldnew["new_macro"]:.4f} |')
 report+=['','仅固定首转换32步开发资格，原R125/R128失败不改。B接近A不证明等价或全由CTAN。有限字符串条件分布不保证旧图风险；标签CE对照及全部失败/费用保留。',f'本成功attempt生成{count["generation_responses"]}，全部attempt生成{proof["all_attempt_generated_responses"]}；失败/native/probe费用另列，不按正常预算抹去重试。无新SFT/VAL编码，CPU保存核验0新模型调用。','准备shell/SSH/Git CPU未全面仪器，未知非0；COSTS/CPU软件/native/保存验证分别保留，wholeworkerCPU和body不重复相加。']
 (p/'REPORT_ZH.md').write_text('\n'.join(report)+'\n');write(p/'ALL_ATTEMPT_COSTS.json',{'worker_successful_attempt':cost,'failed_attempt_1':prior_cost,'failed_controller_exit':json.loads((p/'attempt_1/CONTROL_EXIT.json').read_text()) if prior_cost else None,'controller_exit':exit,'software':json.loads((p/'CPU_SOFTWARE_RUN_COSTS.json').read_text()),'saved_verification':proof,'original_clock_not_reset':True,'INTEGRITY_PULSES':json.loads((p/'INTEGRITY_PULSES.json').read_text()) if (p/'INTEGRITY_PULSES.json').exists() else [],'shell_SSH_Git_CPU_unknown_not_zero':True})
