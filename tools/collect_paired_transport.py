"""One collection of anonymous metrics/scalars only after terminal active-empty state."""
import json
from pathlib import Path
import subprocess
from summarize_paired_transport import report,NEW


def collect(round_dir):
    r=Path(round_dir);rt=r/'local_runtime';h=json.loads((rt/'HANDOFF.private.json').read_text())
    if (r/'results').exists() or (rt/'DELIVERY.private.json').exists():raise ValueError('No repeat collection')
    code='import json\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\nNEW='+repr(NEW)+'\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
program={k:p[k] for k in ('status','phase','started','ended','elapsed_seconds','costs','active','whole_round_wall_seconds','resource_wait_seconds') if k in p}
launch=json.loads((root/'LAUNCH.private.json').read_text())
program['preflight_gate_wait_wall_seconds']=p['started']-launch['launched_at'] if 'started' in p else None
program['resource_admission_revised_before_GPU']=launch.get('resource_only',False)
program['failures']=[{k:v for k,v in a.items() if k in ('job','operation','gpu','returncode','residence_seconds')} for a in p['failures']]
def scalar(v):return {k:x for k,x in v.items() if type(x) in (float,int,bool)}
runs={};preflights={}
for ds in ('ISIC','HK'):
 for arm in NEW:
  name=ds+'_'+arm;folder=root/'runs'/name
  if (folder/'STATUS.json').exists():
   raw=json.loads((folder/'STATUS.json').read_text());v=dict(status={k:x for k,x in raw.items() if k in ('status','steps','stages','policy_updates','max_solve_residual','peak_gpu_bytes')})
   if (folder/'metrics.json').exists():v['metrics']=json.loads((folder/'metrics.json').read_text())
   if (folder/'diagnostics.json').exists():
    diagnostics=[]
    for d in json.loads((folder/'diagnostics.json').read_text()):
     keep=scalar(d)
     for k in ('activation','controller','paired_transport','boundary_paired_transport'):
      if k in d:keep[k]=scalar(d[k])
     if 'boundary_competition' in d:keep['boundary_solve']=scalar(d['boundary_competition']['solve'])
     diagnostics.append(keep)
    v['scalar_diagnostics']=diagnostics
   runs[name]=v
  folder=root/'preflight'/name
  if (folder/'STATUS.json').exists():
   st=json.loads((folder/'STATUS.json').read_text())
   f=json.loads((folder/'PREFLIGHT.json').read_text()) if (folder/'PREFLIGHT.json').exists() else {}
   v=scalar(f);v['receipt_status']=f.get('status');v['worker_status']=st['status']
   complete=st['status']=='COMPLETE' and f.get('status')=='PASS' and all(k in f for k in ('optimizer_updates','parameters_unchanged','test_accessed','pmt','synthetic_highest_bank','max_solve_residual'))
   v['receipt_complete']=complete;v['status']='PASS' if complete else 'INCOMPLETE'
   for key in ('pmt','adapter_pool_gradients','synthetic_highest_bank','batch_solve','activation'):v[key]=scalar(f.get(key,{}))
   preflights[name]=v
print(json.dumps(dict(program=program,runs=runs,preflights=preflights),allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    s=json.loads(x.stdout);s.update(round_name=r.name,source_commit=h['source_commit'],lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),
        cpu_checks=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),control_matching=json.loads((r/'CONTROL_MATCH.json').read_text()),
        engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events'],
        closed_round_cost_ledger={p.parent.parent.name:json.loads(p.read_text())['round_gpu_process_residence_seconds'] for p in Path('docs').glob('*/results/AUDIT.json') if 'round_gpu_process_residence_seconds' in json.loads(p.read_text())})
    old=json.loads(Path('docs/dat_hao_isic_hk2_20261012/results/RESULTS.json').read_text())
    for ds in ('ISIC','HK'):
        for arm in ('base','local_transport','AIR'):s['runs'][ds+'_'+arm]=old['runs'][ds+'_'+arm]
    (rt/'SNAPSHOT_ANONYMOUS.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    if s['program']['status']=='COMPLETE':return report(s,r/'results')
    out=r/'results';out.mkdir();cost=sum(c['residence_seconds'] for c in s['program']['costs'])
    ledger=s['closed_round_cost_ledger'].copy();ledger[r.name]=cost
    a=dict(status='INCOMPLETE',source_commit=h['source_commit'],complete_comparison=False,test_accessed=False,
        formal_adapter_updates=sum(v['status'].get('steps',0) for name,v in s['runs'].items() if any(name.endswith('_'+arm) for arm in NEW) and 'status' in v),
        real_zero_update_preflights=sum(v['status']=='PASS' for v in s['preflights'].values()),
        dispatched_preflights=len(s['preflights']),required_preflights=6,policy_updates=0,
        failures=s['program']['failures'],cost_records=s['program']['costs'],
        round_gpu_process_residence_seconds=cost,closed_round_cost_ledger=ledger,
        cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),
        whole_round_wall_seconds=s['program'].get('whole_round_wall_seconds'),
        resource_wait_seconds=s['program'].get('resource_wait_seconds'),
        cpu_checks=s['cpu_checks'],engineering_events=s['engineering_events'])
    (out/'AUDIT.json').write_text(json.dumps(a,indent=2)+'\n');(out/'PARTIAL_RESULTS.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    (out/'REPORT_ZH.md').write_text('# PMT未完整结束\n\n匿名部分聚合和全部费用保留；需私有核对错误并补去标识原因/缺失范围。missing不填0，不自动retry，不声称完整配对效果。\n')
    return a
