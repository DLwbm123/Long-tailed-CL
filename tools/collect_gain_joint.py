"""Anonymous fixed-candidate diagnosis; no model, images or sample scores leave NAS."""
import copy,json,subprocess
from pathlib import Path
from gain_joint_production import CANDIDATES,HEADS
from concm_statistical_metrics import FIELDS
from collect_intervention_gain import dump,finite,delta,csvwrite,table,EXTRA,JOBS
VERSION='gain_joint_predictive_v2'

def validate(s):
    finite(s);state=s['program'];lock=s['lock'];sha=s['source_commit']
    if state['active'] or state['status'] not in ('COMPLETE','INCOMPLETE'):raise ValueError('Not terminal')
    if state['status']=='INCOMPLETE':return False
    if len(state['costs'])!=8 or len(state['cpu_costs'])!=4 or state['failures'] or {(z['job'],z['operation']) for z in state['costs']}!={(n,o) for n in JOBS for o in ('preflight','evaluate')}:raise ValueError('Cost coverage differs')
    if any(z['returncode'] or not z['process_started'] or z['gpu'] not in (0,1) for z in state['costs']) or any(z['returncode'] or not z['process_started'] or z['gpu'] is not None or z['operation']!='prepare' for z in state['cpu_costs']):raise ValueError('Process scope differs')
    total=images=0
    for n in JOBS:
        d=lock['data'][n.split('_')[0]];prep=s['prepared'][n];run=s['runs'][n];f=s['preflights'][n];ds=n.split('_')[0];arm=n[len(ds)+1:]
        if prep['status']!='PREPARED' or prep['source_commit']!=sha or prep['optimizer_updates'] or prep['medical_images_read'] or len(prep['stages'])!=len(d['task_sizes']):raise ValueError('Preparation coverage differs')
        if run['status']['status']!='COMPLETE' or f['status']!='PASS' or not f['parameters_unchanged'] or not f['CPU_GPU_scores_close'] or not f['guard_checked'] or f['classification_metrics_read'] or f['real_validation_images']!=64:raise ValueError('Real preflights differ')
        for a in (run,f):
            if a['source_commit']!=sha or a['version']!=VERSION or a['optimizer_updates'] or a['policy_updates'] or a['train_accessed'] or a['test_accessed']:raise ValueError('Provenance differs')
        if set(run['metrics'])!=set(HEADS) or len(run['stages'])!=len(d['task_sizes']):raise ValueError('Seven heads required')
        for task,(a,st) in enumerate(zip(prep['stages'],run['stages']),1):
            seen=d['order'][:sum(d['task_sizes'][:task])]
            if a['task']!=task or st['task']!=task or set(a['variants'])!=set(CANDIDATES) or set(st['heads'])!=set(HEADS) or set(st['raw_heads'])!=set(HEADS) or a['native_solve']['relative_residual']>1e-8:raise ValueError('Stage coverage/solve differs')
            for v in CANDIDATES:
                md=dict(dataset=ds,control=arm,control_source=lock['controls'][arm]['source'],order=d['order'],seen=seen,seed=74002,task=task,source=sha,variant=v,version=VERSION,control_steps=d['cumulative_steps'][task-1],optimizer_updates=0)
                if a['variants'][v]['metadata']!=md or a['variants'][v]['mechanical']['relative_residual']>1e-8:raise ValueError('Sidecar source differs')
                total+=1
            for v in HEADS:
                z=st['heads'][v]
                if not z['published_control_exact'] or not z['encoder_state_unchanged'] or z['paired']['new']['harmed'] or z['margins']['new']['applied_delta']['negative'] or not z['guard_diagnostic']['new_logits_exact'] or task==1 and z['paired']['all']['changed']:raise ValueError('Guard failed')
                for report in (z,st['raw_heads'][v]):
                    for m in report['metrics'].values():
                        if m['seen']!=seen or m['per_class_n']!={str(c):d['val_counts'][str(c)] for c in seen}:raise ValueError('Bad class denominator')
                    for pair in report['paired'].values():
                        if pair['changed']!=pair['rescued']+pair['harmed']+pair['wrong_to_different_wrong'] or pair['net_correct']!=pair['rescued']-pair['harmed']:raise ValueError('Counts differ')
                ref=lock['control_references'][n]['stages'][task-1]
                if any(z['metrics']['OFF'][k]!=ref[k] for k in ('seen','per_class_n','per_class_recall','accuracy')):raise ValueError('OFF parity differs')
            for v,selection in a['selections'].items():
                scale=selection['chosen_scale'];expected=st['heads'][CANDIDATES[(.125,.25,.5,1.).index(scale)]]['metrics']['ON'] if scale else st['heads'][v]['metrics']['OFF']
                if st['heads'][v]['metrics']['ON']!=expected:raise ValueError('Selection changed after val')
        images+=run['validation_image_exposures']
    if total!=72 or images!=15966:raise ValueError('Exposure/sidecars differ')
    return True

def estimates(job,task,head,report,mechanical):
    seen=report['metrics']['ON']['seen'];paired=report['paired'];actual=sum((paired[str(c)]['rescued']-paired[str(c)]['harmed'])/paired[str(c)]['n'] for c in seen)/len(seen);rows=[];classes=[]
    for kind in ('point','predictive'):
        m=mechanical[kind]
        for z in m['scenarios']:
            center=z['known_net'] if not z['unknown_mass'] else None
            rows.append(dict(job=job,task=task,head=head,estimator=kind,scenario=z['scenario'],known_only_estimated_net=z['known_net'],full_estimated_net=center,actual_macro_net=actual,numerical_error=z['numerical_error'],unknown_mass=z['unknown_mass'],lower=z['net_lower'],upper=z['net_upper'],full_estimate_minus_actual=None if center is None else center-actual,known_only_minus_actual=z['known_net']-actual,known_only_bias_not_full_population_bias=center is None,actual_below_interval=actual<z['net_lower'],actual_above_interval=actual>z['net_upper'],numerical_or_model_interval_not_empirical_confidence=True))
            predicted={a['label']:a for a in z['classes']}
            for i,c in enumerate(seen):
                a=paired[str(c)];b=predicted.get(i);zero=m['exact_identity']
                classes.append(dict(job=job,task=task,head=head,estimator=kind,scenario=z['scenario'],label=c,n=a['n'],rescued=a['rescued'],harmed=a['harmed'],actual_rescue=a['rescued']/a['n'],actual_harm=a['harmed']/a['n'],actual_net=a['net_correct']/a['n'],known_estimated_rescue=0. if zero else b['known_rescue'],known_estimated_harm=0. if zero else b['known_harm'],known_estimated_net=0. if zero else b['known_net'],unknown_mass=0. if zero else b['unknown_mass']))
    return rows,classes

def report(s,r):
    complete=validate(s);out=r/'results';out.mkdir(exist_ok=False);main=[];contrasts=[];classes=[];paired=[];margins=[];confusions=[];bias=[];riskclasses=[];selections=[];guard=[];rawmain=[];interactions=[]
    def metricrow(ds,arm,v,m):return dict(dataset=ds,control=arm,head=v,**{k:None if m.get(k) is None else 100*m[k] for k in FIELDS+EXTRA})
    for n,run in s['runs'].items():
        ds=run['dataset'];arm=run['control'];data=s['lock']['data'][ds];off=run['control_metrics'];main.append(metricrow(ds,arm,'OFF',off));allm=dict(OFF=off,**{v:run['metrics'][v]['guarded'] for v in HEADS})
        for v in HEADS:
            main.append(metricrow(ds,arm,v,allm[v]));rawmain.append(metricrow(ds,arm,v,run['metrics'][v]['raw']));contrasts.append(dict(job=n,contrast=v+'−OFF',delta_pp=delta(allm[v],off)))
        for a,b in [('JOINT_POINT_SELECTED','LEGACY_SELECTED'),('JOINT_PRED_SELECTED','LEGACY_SELECTED'),('JOINT_PRED_SELECTED','JOINT_POINT_SELECTED')]:contrasts.append(dict(job=n,contrast=a+'−'+b,delta_pp=delta(allm[a],allm[b])))
        for v,m in allm.items():contrasts.append(dict(job=n,contrast=v+'−AIR',delta_pp=delta(m,s['lock']['AIR_references'][ds]),BA_gap_reduction_pp=100*(m[FIELDS[0]]-off[FIELDS[0]])))
        for st,prep in zip(run['stages'],s['prepared'][n]['stages']):
            task=st['task'];reference=st['heads']['JOINT_PRED_SELECTED'];seen=reference['metrics']['OFF']['seen']
            for c in seen:
                a=reference['paired'][str(c)];classes.append(dict(job=n,task=task,head='OFF',readout='original',label=c,train_n=data['train_counts'][str(c)],n=a['n'],recall=reference['metrics']['OFF']['per_class_recall'][str(c)],rescued=0,harmed=0,net_correct=0,wrong_to_different_wrong=0))
            for v in HEADS:
                for mode,a in [('guarded',st['heads'][v]),('raw',st['raw_heads'][v])]:
                    for c in seen:
                        z=a['paired'][str(c)];classes.append(dict(job=n,task=task,head=v,readout=mode,label=c,train_n=data['train_counts'][str(c)],n=z['n'],recall=a['metrics']['ON']['per_class_recall'][str(c)],rescued=z['rescued'],harmed=z['harmed'],net_correct=z['net_correct'],wrong_to_different_wrong=z['wrong_to_different_wrong']))
                    for g,z in a['paired'].items():paired.append(dict(job=n,task=task,head=v,readout=mode,group=g,**z))
                    for g,z in a['margins'].items():
                        for kind,value in z.items():margins.append(dict(job=n,task=task,head=v,readout=mode,group=g,margin=kind,**value))
                confusions.extend(dict(job=n,task=task,head=v,**z) for z in st['heads'][v]['class_confusion']);guard.append(dict(job=n,task=task,head=v,**st['heads'][v]['guard_diagnostic']))
            for v in CANDIDATES:
                a,b=estimates(n,task,v,st['heads'][v],prep['variants'][v]['mechanical']);bias.extend(a);riskclasses.extend(b)
            selections.append(dict(job=n,task=task,selections=prep['selections']))
    if complete:
        for ds in ('ISIC','HK'):
            b=s['runs'][ds+'_base'];p=s['runs'][ds+'_local_transport']
            for v in HEADS:
                x=delta(p['metrics'][v]['guarded'],p['control_metrics']);y=delta(b['metrics'][v]['guarded'],b['control_metrics']);it={k:None if x[k] is None or y[k] is None else x[k]-y[k] for k in x};contrasts.append(dict(job=ds,contrast='('+v+'_LT−OFF_LT)−('+v+'_base−OFF_base)',delta_pp=it));interactions.append(dict(dataset=ds,head=v,delta_pp=it))
            contrasts.append(dict(job=ds,contrast='JOINT_PRED_SELECTED_LT−JOINT_PRED_SELECTED_base',delta_pp=delta(p['metrics']['JOINT_PRED_SELECTED']['guarded'],b['metrics']['JOINT_PRED_SELECTED']['guarded'])))
        if len(main)!=32 or len(contrasts)!=88 or len(bias)!=432:raise ValueError('Complete fixed comparison required')
    air=[metricrow(ds,'reference','AIR',m) for ds,m in s['lock']['AIR_references'].items()];gpu=sum(z['residence_seconds'] for z in s['program']['costs']);cpu=sum(z['residence_seconds'] for z in s['program']['cpu_costs']);ledger=dict(s['ledger']);assert r.name not in ledger;history=83019.61353103025+sum(ledger.values());ledger[r.name]=gpu
    audit=dict(status='PASS' if complete else 'INCOMPLETE',source_commit=s['source_commit'],reporting_source_commit=s['reporting_source_commit'],design_source_commit=s['lock']['design_source_commit'],direction_source_commit=s['lock']['design_parameters']['direction_source_commit'],round_gpu_process_residence_seconds=gpu,cpu_preparation_process_residence_seconds=cpu,cpu_production_all_attempt_process_residence_seconds=s['cpu_checks']['all_CPU_attempt_residence_seconds'],CPU_PRODUCTION_CHECK=s['cpu_checks'],cost_records=s['program']['costs'],cpu_cost_records=s['program']['cpu_costs'],failures=s['program']['failures'],closed_round_cost_ledger=ledger,historical_closed_gpu_process_residence_seconds=history,cumulative_gpu_process_residence_seconds=history+gpu,whole_round_wall_seconds=s['program'].get('whole_round_wall_seconds'),source_stage_checkpoints=18 if complete else None,sidecars=72 if complete else None,main_rows=32 if complete else None,contrasts=88 if complete else None,model_candidate_scenario_rows=len(bias),primary_control='local_transport',main_head='JOINT_PRED_SELECTED',optimizer_updates=0,policy_updates=0,test_accessed=False,train_accessed=False,validation_retuning=False,automatic_retry=False,prior_results_reused_cost=0,numerical_probability_interval_not_clinical_confidence=True,previous_design_CPU_residence_seconds=29.468065921217203,previous_design_not_recharged=True,engineering_events=s['engineering_events'])
    dump(out/'AUDIT.json',audit);dump(out/('RESULTS.json' if complete else 'PARTIAL_RESULTS.json'),dict(rows=main+air,raw_rows=rawmain,runs=s['runs'],prepared=s['prepared'],program=s['program'],preflights=s['preflights'],missing_jobs=[n for n in JOBS if n not in s['runs']]));dump(out/'CONTRIBUTIONS.json',dict(contrasts=contrasts,interactions=interactions));dump(out/'SELECTIONS.json',selections)
    if s.get('partial_stages'):dump(out/'PARTIAL_STAGE_RESULTS.json',s['partial_stages'])
    for f,a in [('MAIN.csv',main+air),('RAW_MAIN.csv',rawmain),('STAGE_CLASS_RECALLS.csv',classes),('PAIRED_COUNTS.csv',paired),('MARGINS.csv',margins),('CLASS_CONFUSION.csv',confusions),('ESTIMATE_BIAS.csv',bias),('CLASS_ESTIMATES.csv',riskclasses),('GUARD_COUNTS.csv',guard)]:csvwrite(out/f,a)
    lines=['# GAIN 同状态四固定候选完整诊断','',f'状态{audit["status"]}。固定JOINT_PRED_SELECTED与local_transport；四步长不按val更换。0optimizer/0policy/0train/test封存。全部候选共享编码器，模型选择先于val。','Gaussian点模型与Student-t预测模型只是假设模型；数值区间不等于医学统计置信度。未知质量>0时完整中心与偏差留null；known-only减实际总净值只描述部分估计差，不能称全总体估计偏差。','单seed/order反复development val，非临床、患者泛化或统计显著证据。']
    table(lines,['dataset','control','head']+list(FIELDS+EXTRA),main+air);lines+=['','## 全部贡献(pp，forget下降有利)'];table(lines,['job','contrast']+list(FIELDS+EXTRA),[dict(job=a['job'],contrast=a['contrast'],**a['delta_pp']) for a in contrasts]);lines+=['','## 四候选实际净值、模型估计及不确定质量'];table(lines,list(bias[0]) if bias else [],bias);lines+=['','## 各stage逐类全部正负'];table(lines,list(classes[0]) if classes else [],classes);lines+=['','完整RAW/守卫、新旧混淆、边界间隔与各类救回/破坏见上述CSV与RESULTS。n<5连续margin抑制，离散计数完整保留。','方向判定须看四候选实际宏净值；正候选未准入才支持准入保守，不能从identity直接断言方向无效。方向实际均负与模型中心失配、未知质量/积分半宽导致拒绝应分别报告。',f'GPU本轮{gpu:.9f}s；CPU四准备{cpu:.9f}s；生产CPU全部尝试{s["cpu_checks"]["all_CPU_attempt_residence_seconds"]:.9f}s。pipeline包含于驻留不相加；历史设计CPU29.468065921s不重计。',f'闭合历史GPU{history:.9f}s，本轮后{history+gpu:.9f}s；旧控制/AIR/旧GAIN复用费用0。失败全部计入。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit

def collect(round_dir):
    r=Path(round_dir);rt=r/'local_runtime'
    if (r/'results').exists() or (rt/'DELIVERY.private.json').exists():raise ValueError('Already collected')
    h=json.loads((rt/'HANDOFF.private.json').read_text());code='import json\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try:args=(proc/'cmdline').read_bytes().split(bytes([0]))
  except (FileNotFoundError,PermissionError):continue
  assert b'/tmp/q185w.py' not in args and b'/tmp/q185c.py' not in args
s=dict(program=p,prepared={},runs={},preflights={},partial_stages={})
for n in ['ISIC_base','ISIC_local_transport','HK_base','HK_local_transport']:
 path=root/'prepared'/n/'PREPARED.json'
 if path.exists():s['prepared'][n]=json.loads(path.read_text())
 for key,folder in [('runs','runs'),('preflights','preflight')]:
  path=root/'JOINT'/folder/n/'RESULTS.json'
  if path.exists():
   a=json.loads(path.read_text())
   if key=='runs':a['status']=json.loads((path.parent/'STATUS.json').read_text())
   s[key][n]=a
for n in ['ISIC_base','ISIC_local_transport','HK_base','HK_local_transport']:
 if n not in s['runs']:
  path=root/'JOINT'/'runs'/n/'PARTIAL_STAGE_RESULTS.json'
  if path.exists():s['partial_stages'][n]=json.loads(path.read_text())
print(json.dumps(s,allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError('Private collection receipt failed')
    s=json.loads(x.stdout);s.update(source_commit=h['source_commit'],reporting_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),ledger=json.loads(Path('docs/gain1_20261013/results/AUDIT.json').read_text())['closed_round_cost_ledger'],cpu_checks=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events']);dump(rt/'SNAPSHOT_ANONYMOUS.json',s);return report(s,r)
if __name__=='__main__':print(json.dumps(collect(json.load(__import__('sys').stdin)['round_dir']),indent=2))
