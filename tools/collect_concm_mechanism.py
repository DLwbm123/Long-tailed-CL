"""Terminal anonymous diagnostic collection. No Torch or old checkpoint access."""
import json,subprocess,csv
from pathlib import Path
HEADS=('MPC','REG_ZERO','IDENTITY','ETF_FULL','ETF_DISC','MPC_DSM')
JOBS=[d+'_'+a for d in ('ISIC','HK') for a in ('base','local_transport')]
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def validate(s):
    p=s['program'];lock=s['lock']
    if p['active']:raise ValueError('Still active')
    if p['status']!='COMPLETE':return False
    expected={(n,op) for n in JOBS for op in ('preflight','evaluate')}
    if len(p['costs'])!=8 or {(z['job'],z['operation']) for z in p['costs']}!=expected or len(p['cpu_costs'])!=4 or p['failures']:raise ValueError('Cost coverage differs')
    if any(z['returncode'] or not z['process_started'] or z['gpu'] not in (0,1) for z in p['costs']):raise ValueError('GPU scope failed')
    for n in JOBS:
        a=s['prepared'][n]
        if a['status']!='PREPARED' or a['source_commit']!=s['source_commit'] or len(a['stages'])!=1 or set(a['stages'][0]['variants'])!=set(HEADS):raise ValueError('CPU seal incomplete')
        r=s['runs'][n];f=s['preflights'][n];d=lock['data'][n.split('_')[0]]
        if r['status']['status']!='COMPLETE' or f['status']!='PASS' or not f['parameters_unchanged'] or not f['CPU_GPU_scores_close'] or f['classification_metrics_read']:raise ValueError('Preflight/result incomplete')
        for v in (r,f):
            if v['source_commit']!=s['source_commit'] or v['optimizer_updates'] or v['policy_updates'] or v['train_accessed'] or v['test_accessed'] or v['version']!='concm_final_mechanism_diagnostic_v1':raise ValueError('Provenance/scope differs')
        if set(r['reports'])!=set(HEADS):raise ValueError('Heads missing')
        ref=lock['control_references'][n]['stages'][-1]
        for v,a in r['reports'].items():
            for mode in ('OFF','ON'):
                m=a['metrics'][mode]
                if m['seen']!=d['order'] or m['per_class_n']!={str(c):d['val_counts'][str(c)] for c in d['order']}:raise ValueError('Denominator differs')
                for z in a['paired'].values():
                    if z['changed']!=z['rescued']+z['harmed']+z['wrong_to_different_wrong'] or z['net_correct']!=z['rescued']-z['harmed']:raise ValueError('Count identity differs')
            for key in ('accuracy','per_class_n','per_class_recall','seen'):
                if a['metrics']['OFF'][key]!=ref[key]:raise ValueError('OFF parity differs')
            if a['mechanical']['solver']['relative_residual']>1e-8:raise ValueError('Solve failed')
            if v=='IDENTITY' and a['paired']['all']['changed']:raise ValueError('Identity changed labels')
            if v in ('MPC','ETF_FULL','MPC_DSM'):
                previous=s['prior']['runs'][{'ETF_FULL':'DSM'}.get(v,v)][n]['stages'][-1]['metrics']['ON']
                if a['metrics']['ON']!=previous:raise ValueError('Previous diagnostic head parity differs')
    return True

def report(s,r):
    complete=validate(s);out=r/'results';out.mkdir(exist_ok=False);rows=[];classes=[];paired=[];contrasts=[]
    for n,run in s['runs'].items():
        d=s['lock']['data'][n.split('_')[0]];old=d['order'][:-d['task_sizes'][-1]];new=d['order'][-d['task_sizes'][-1]:]
        for v,a in run['reports'].items():
            m=a['metrics']['ON'];off=a['metrics']['OFF'];p=a['paired']['new']
            rows.append(dict(job=n,head=v,BA=100*m['balanced_accuracy'],delta_BA=100*(m['balanced_accuracy']-off['balanced_accuracy']),Acc=100*m['accuracy'],delta_Acc=100*(m['accuracy']-off['accuracy']),tail=100*m['tail_recall'],F1=100*m['macro_F1'],old=100*sum(m['per_class_recall'][str(c)] for c in old)/len(old),new=100*sum(m['per_class_recall'][str(c)] for c in new)/len(new),new_rescued=p['rescued'],new_harmed=p['harmed'],new_margin_delta=a['margins']['new']['applied_delta']['mean'],normalized_new_margin_delta=a['normalized_fixed_competitor_margin']['new']['delta']['mean']))
            for c,count in m['per_class_n'].items():classes.append(dict(job=n,head=v,label=c,n=count,off_correct=round(count*off['per_class_recall'][c]),on_correct=round(count*m['per_class_recall'][c]),**{k:a['paired'][c][k] for k in ('rescued','harmed','wrong_to_different_wrong')}))
            for g,z in a['paired'].items():paired.append(dict(job=n,head=v,group=g,**z))
        for on,off in [('ETF_FULL','REG_ZERO'),('ETF_DISC','ETF_FULL'),('MPC_DSM','MPC'),('MPC_DSM','ETF_FULL')]:
            a=next(z for z in rows if z['job']==n and z['head']==on);b=next(z for z in rows if z['job']==n and z['head']==off)
            contrasts.append(dict(job=n,contrast=on+'−'+off,delta_pp={k:a[k]-b[k] for k in ('BA','Acc','tail','F1','old','new')},new_paired=run['head_comparisons'][on+'−'+off]['paired']['new']))
    cost=sum(z['residence_seconds'] for z in s['program']['costs']);cpu=sum(z['residence_seconds'] for z in s['program']['cpu_costs']);ledger=dict(s['ledger']);assert r.name not in ledger
    history=83019.61353103025+sum(ledger.values());ledger[r.name]=cost
    audit=dict(status='PASS' if complete else 'INCOMPLETE',source_commit=s['source_commit'],optimizer_updates=0,train_accessed=False,test_accessed=False,automatic_retry=False,round_gpu_process_residence_seconds=cost,cpu_preparation_process_residence_seconds=cpu,CPU_PRODUCTION_CHECK=s['cpu_checks'],cost_records=s['program']['costs'],cpu_cost_records=s['program']['cpu_costs'],failures=s['program']['failures'],whole_round_wall_seconds=s['program'].get('whole_round_wall_seconds'),historical_closed_gpu_process_residence_seconds=history,cumulative_gpu_process_residence_seconds=history+cost,closed_round_cost_ledger=ledger,prior_results_reused_cost=0,engineering_events=s['engineering_events'])
    dump(out/'AUDIT.json',audit);dump(out/('RESULTS.json' if complete else 'PARTIAL_RESULTS.json'),dict(rows=rows,runs=s['runs'],prepared=s['prepared'],program=s['program'],preflights=s['preflights']));dump(out/'CONTRIBUTIONS.json',contrasts)
    for file,data in [('MAIN.csv',rows),('CLASS_COUNTS.csv',classes),('PAIRED_COUNTS.csv',paired)]:
        if data:
            with (out/file).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    lines=['# MPC/DSM 同状态最小机制诊断','',f'状态：{audit["status"]}。四个最终原checkpoint，六个固定诊断头，0 optimizer/0policy/0训练图像/test封存。最终状态诊断不提供新的平均增量或遗忘。',
      'REG_ZERO与ETF_FULL固定同一原γ；IDENTITY为原解析最优头目标（浮点readout原封保留）；ETF_DISC只改变前瞻指定的判别范数缩放。公共分量不影响argmax，但计入原ETF全头尺度。MPC供体权重只支持方向/混淆一致性核对，没有独立真实中心估计。',
      '所有结果为重复development val诊断，不是官方ConCM、训练收益或临床/test证据；不按本结果调参或自动选择头。','', '|job|head|BA%|ΔBA pp|Acc%|old%|new%|新纠正|新破坏|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for z in rows:lines.append(f'|{z["job"]}|{z["head"]}|{z["BA"]:.6f}|{z["delta_BA"]:+.6f}|{z["Acc"]:.6f}|{z["old"]:.6f}|{z["new"]:.6f}|{z["new_rescued"]}|{z["new_harmed"]}|')
    lines+=['','## 正则化/方向/尺度贡献','', '|job|contrast|BA|Acc|tail|F1|old|new|','|---|---|---:|---:|---:|---:|---:|---:|']
    for z in contrasts:lines.append('|'+z['job']+'|'+z['contrast']+'|'+'|'.join(f'{v:+.6f}' for v in z['delta_pp'].values())+'|')
    lines+=['', '同状态头间新类纠正/破坏：']+[z['job']+' '+z['contrast']+' '+json.dumps(z['new_paired'],ensure_ascii=False) for z in contrasts]
    lines+=['','## 全头、判别及公共范数','', '|job|head|full|disc|common|公共能量%|γ|','|---|---|---:|---:|---:|---:|---:|']
    for n,run in s['runs'].items():
        for v,a in run['reports'].items():
            z=a['mechanical']['norms'];lines.append(f'|{n}|{v}|{z["full_norm"]:.9f}|{z["discriminative_norm"]:.9f}|{z["common_norm"]:.9f}|{100*z["common_energy_fraction"]:.6f}|{a["mechanical"]["structure_coefficient"]:.9g}|')
        lines+=['',f'### {n} 原头与供体核对','',json.dumps(run['reports']['MPC']['mechanical']['reference_norms'],ensure_ascii=False),'','|class|n|主要供体(权重/方向cos)|有效供体数|错误类供体质量|错向权重重合|','|---|---:|---|---:|---:|---:|']
        d=s['lock']['data'][n.split('_')[0]]
        for z in run['donor_summary']:
            donors='; '.join(str(a['donor'])+' ('+f'{a["weight"]:.4f}/'+('undefined' if a['shift_alignment'] is None else f'{a["shift_alignment"]:.4f}')+')' for a in z['top3']);over='无错误' if z['confusion_weight_overlap'] is None else f'{z["confusion_weight_overlap"]:.6f}'
            lines.append(f'|{z["label"]}|{d["train_counts"][str(z["label"])]}|{donors}|{z["donor_effective_count"]:.4f}|{z["weight_on_actual_wrong_classes"]:.6f}|{over}|')
    lines+=['','## 全部类纠正和破坏','', '|job|head|class|val n|OFF正确|ON正确|纠正|破坏|错转错|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for z in classes:lines.append('|'+ '|'.join(str(z[k]) for k in z)+'|')
    lines+=['','## 费用',f'GPU各8次实际驻留总{cost:.9f}秒，CPU四次准备{cpu:.9f}秒；CPU fixture驻留{s["cpu_checks"]["process_residence_seconds"]:.9f}秒包含pipeline{s["cpu_checks"]["pipeline_seconds"]:.9f}秒，不相加。并行墙钟{s["program"].get("whole_round_wall_seconds")}秒。闭合历史{history:.9f}秒，本轮后{history+cost:.9f}秒。复用旧结果计费0。全部独立记录及错误见AUDIT。',
      '完整旧新混淆、原始/几何归一化margin正零负、31类分母/纠正破坏见RESULTS/CSV。极少n类连续值抑制，missing不填0；供体向错误类的重合不等于独立中心风险估计。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit

def collect(round_dir):
    r=Path(round_dir);rt=r/'local_runtime'
    if (r/'results').exists() or (rt/'DELIVERY.private.json').exists():raise ValueError('Already collected/delivered')
    h=json.loads((rt/'HANDOFF.private.json').read_text())
    code='import json,subprocess\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try:args=(proc/'cmdline').read_bytes().split(b'\\0')
  except (FileNotFoundError,PermissionError):continue
  assert b'/tmp/q171w.py' not in args and b'/tmp/q171c.py' not in args
s=dict(program=p,prepared={},runs={},preflights={})
for n in ['ISIC_base','ISIC_local_transport','HK_base','HK_local_transport']:
 path=root/'prepared'/n/'PREPARED.json'
 if path.exists():
  a=json.loads(path.read_text())
  for z in a.get('stages',[]):z.pop('donors',None)
  s['prepared'][n]=a
 for key,folder in [('runs','runs'),('preflights','preflight')]:
  path=root/'MECHANISM'/folder/n/'RESULTS.json'
  if path.exists():
   a=json.loads(path.read_text())
   if key=='runs':a['status']=json.loads((path.parent/'STATUS.json').read_text())
   s[key][n]=a
print(json.dumps(s,allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    s=json.loads(x.stdout);s.update(source_commit=h['source_commit'],lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),prior=json.loads(Path('docs/concm_statistical1_20261013/results/RESULTS.json').read_text()),ledger=json.loads(Path('docs/concm_statistical1_20261013/results/AUDIT.json').read_text())['closed_round_cost_ledger'],cpu_checks=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events'])
    # Never copy private donor arrays or sample-level data from NAS.
    dump(rt/'SNAPSHOT_ANONYMOUS.json',{k:v for k,v in s.items() if k!='prior'});return report(s,r)
if __name__=='__main__':print(json.dumps(collect(json.load(__import__('sys').stdin)['round_dir']),indent=2))
