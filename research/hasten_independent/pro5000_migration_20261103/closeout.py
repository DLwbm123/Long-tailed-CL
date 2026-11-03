"""Saved aggregate closeout, no model imports or new scientific computation."""
import argparse,json,csv,math,time
from pathlib import Path

def run(root,kind,counts):
 start=time.process_time();wt=time.time();src=root/'results_pro5000';rows=json.loads((src/'STAGE_METRICS.json').read_text());a=json.loads((src/'ANALYSIS.json').read_text());exit=json.loads((src/'CONTROL_EXIT_1.json').read_text());train=json.loads((src/'TRAIN_COMPLETE.json').read_text());ev=json.loads((src/'EVAL_COMPLETE.json').read_text());ver=json.loads((src/'VERIFICATION.json').read_text());old=json.loads((root/'MIGRATION_STOP_RECEIPT.private.json').read_text());assert exit['exit_code']==0 and ver['passed'] and ev['all_heads_sealed_before_VAL'];assert len(rows)==6 and ev['counts']['TEST_rows']==0 and train['counts']['VAL_rows']==0
 arms=list(rows[0]['arms']);expected=31260 if kind=='r95' else 7294;assert ev['counts']['formal_optimizer_steps']==expected;assert ver['saved_endpoints']==(18 if kind=='r95' else 12);assert sum(rows[-1]['arms'][arms[0]]['counts'])==1698
 stage=[];classes=[];groups=[];maxerr=0.;microerr=0.
 for s in rows:
  n=s['classes'];oldn=0 if s['stage']==0 else rows[s['stage']-1]['classes']
  for arm,m in s['arms'].items():
   cnt=m['counts'];cor=m['correct'];assert len(cnt)==len(cor)==n and all(0<=a<=b for a,b in zip(cor,cnt));rec=[100*a/b if b else 0 for a,b in zip(cor,cnt)];eligible=[c for c in range(n) if cnt[c]>0];tail=[c for c in eligible if counts[c]<250];macro=sum(rec[c] for c in eligible)/len(eligible);ta=sum(rec[c] for c in tail)/len(tail);maxerr=max(maxerr,abs(macro-m['macro']),abs(ta-m['tail']));microerr=max(microerr,abs(100*sum(cor)/sum(cnt)-m['micro']));assert maxerr<1e-10 and microerr<1e-5
   stage.append([s['stage'],arm,m['micro'],m['macro'],m['tail']])
   for c in range(n):classes.append([s['stage'],arm,c,cnt[c],cor[c],rec[c],('old' if c<oldn else 'new'),('tail' if counts[c]<250 else 'head')])
   masks={'old':[c for c in eligible if c<oldn],'new':[c for c in eligible if c>=oldn],'head':[c for c in eligible if counts[c]>=250],'tail':tail}
   for age in ['old','new']:
    for freq in ['head','tail']:masks[age+'_'+freq]=[c for c in masks[age] if c in masks[freq]]
   for name,ix in masks.items():
    ba=None if not ix else sum(rec[c] for c in ix)/len(ix);micro=None if not ix else 100*sum(cor[c] for c in ix)/sum(cnt[c] for c in ix);groups.append([s['stage'],arm,name,len(ix),sum(cnt[c] for c in ix),sum(cor[c] for c in ix),ba,micro])
    if name in m['groups']:
     v=m['groups'][name];assert (ba is None)==(v is None);assert ba is None or abs(ba-v)<1e-10
 def dump(name,header,data):
  with (root/name).open('w') as f:w=csv.writer(f,lineterminator="\n");w.writerow(header);w.writerows(data)
 dump('STAGE_METRICS.csv',['stage','arm','micro','macro','tail'],stage);dump('CLASS_STAGE.csv',['stage','arm','class','VAL_count','correct','recall','age','frequency'],classes);dump('GROUP_STAGE.csv',['stage','arm','group','eligible_classes','VAL_count','correct','macro','micro'],groups)
 if kind=='r95':
  d=lambda ref,s,k:rows[s]['arms']['FULL'][k]-rows[s]['arms'][ref][k]
  globalpass=all(sum(d(ref,s,k) for s in range(6))/6>1e-8 for ref in ['CE_WEIGHT','UNIFORM'] for k in ['macro','tail']);stagepass=sum(all(d(ref,s,k)>1e-8 for ref in ['CE_WEIGHT','UNIFORM'] for k in ['macro','tail']) for s in range(6))>=5;finalpass=all(d('NCM',5,k)>1e-8 for k in ['macro','tail']);tiny=[rows[-1]['arms']['FULL']['recall'][c]-rows[-1]['arms']['NCM']['recall'][c] for c in [1,2,12]];tinypass=all(x>=-1e-8 for x in tiny) and sum(x>1e-8 for x in tiny)>=2;gate=globalpass and stagepass and finalpass and tinypass;assert gate==a['necessary_gate']
 else:
  mean={k:sum(s['arms']['JOINT'][k]-s['arms']['SFT'][k] for s in rows)/6 for k in ['macro','tail','micro']};final={k:rows[-1]['arms']['JOINT'][k]-rows[-1]['arms']['SFT'][k] for k in mean};assert all(abs(mean[k]-a['mean_JOINT_minus_internal_SFT_pp'][k])<1e-10 and abs(final[k]-a['final_delta_pp'][k])<1e-10 for k in mean);stagepass=sum(rows[s]['arms']['JOINT']['macro']-rows[s]['arms']['SFT']['macro']>=-1e-8 and rows[s]['arms']['JOINT']['tail']-rows[s]['arms']['SFT']['tail']>=-1e-8 for s in range(6))>=4;gate=mean['macro']>=.5 and final['macro']>=.5 and final['tail']>=1 and mean['tail']>=-1e-8 and stagepass and all(t['harm']==0 for t in a['tiny']);assert gate==a['performance_pilot_gate'];assert ev['counts']['GRPO_optimizer_steps']==2084
 summary={'status':'COMPLETE_NEGATIVE_GATE','gate':gate,'completed_stages':6,'TEST_rows':0,'formal_updates':expected,'saved_endpoints':ver['saved_endpoints'],'mean':{arm:{k:sum(s['arms'][arm][k] for s in rows)/6 for k in ['micro','macro','tail']} for arm in arms},'final':{arm:{k:rows[-1]['arms'][arm][k] for k in ['micro','macro','tail']} for arm in arms},'analysis':a,'new_forward_optimizer_VAL_TEST':0};(root/'RESULT_SUMMARY.json').write_text(json.dumps(summary,indent=2))
 cost={'source_stop':'USER_AUTHORIZED_MIGRATION_NOT_ENGINEERING_FAILURE','source_last_committed_counts':old['active_before_stop']['counts'],'source_worker':old['controller_exit'],'target_worker':exit,'worker_CPU_sum_both_servers':old['controller_exit']['worker_CPU_seconds']+exit['worker_CPU_seconds'],'target_train_core_included_in_worker':train,'target_evaluation_counts':ev['counts'],'other_preparation_CPU_nonzero_not_fully_sampled':True,'native_profile_CPU_included_do_not_add_again':True,'all_history_retained':True};(root/'COSTS.json').write_text(json.dumps(cost,indent=2))
 check={'passed':True,'scope':'saved JSON aggregate and receipt recomputation; reuse existing tensor verification','all_stage_class_metrics_recomputed':True,'macro_tail_maxabs_error_pp':maxerr,'micro_FP32_rounding_maxabs_error_pp':microerr,'original_gate_recomputed':gate,'new_encoder_classifier_policy_forward':0,'new_optimizer':0,'new_VAL_TEST_reads':0,'CPU_seconds':time.process_time()-start,'wall_seconds':time.time()-wt};(root/'CLOSEOUT_VERIFICATION.json').write_text(json.dumps(check,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('kind',choices=['r95','r96']);p.add_argument('template',type=Path);args=p.parse_args();run(args.root,args.kind,json.loads(args.template.read_text())['train_counts'])
