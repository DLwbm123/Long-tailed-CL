"""Only anonymous aggregates; no images, scores, memory IDs, weights or raw logs."""
import json
from pathlib import Path
import subprocess
from summarize_paper_medical import report


def collect(round_dir):
    r=Path(round_dir);runtime=r/'local_runtime';h=json.loads((runtime/'HANDOFF.private.json').read_text())
    if (r/'results').exists() or (runtime/'DELIVERY.private.json').exists():raise ValueError('No repeat collection')
    code='import json\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
program={k:p[k] for k in ('status','phase','started','ended','elapsed_seconds','active','costs','whole_round_wall_seconds','resource_wait_seconds') if k in p}
program['failures']=[{k:v for k,v in c.items() if k in ('job','operation','gpu','returncode','residence_seconds')} for c in p.get('failures',[])]
runs={};preflights={}
for ds in ('ISIC','HK'):
 for method in ('DAT_paper','Hao_paper'):
  name=ds+'_'+method;folder=root/'runs'/name
  if (folder/'STATUS.json').exists():
   raw=json.loads((folder/'STATUS.json').read_text());s={k:v for k,v in raw.items() if k in ('status','phase','steps','analytic_batches','stages','policy_updates','test_accessed')};s['error_present']=bool(raw.get('error'));v=dict(status=s)
   for filename,key in [('metrics.json','metrics'),('MODEL_COST.json','model_cost')]:
    if (folder/filename).exists():v[key]=json.loads((folder/filename).read_text())
   if (folder/'diagnostics.json').exists():v['scalar_diagnostics']=[{k:x for k,x in d.items() if type(x) in (int,float,bool)} for d in json.loads((folder/'diagnostics.json').read_text())]
   runs[name]=v
  folder=root/'preflight'/name
  if (folder/'PREFLIGHT.json').exists():preflights[name]=json.loads((folder/'PREFLIGHT.json').read_text())
print(json.dumps(dict(program=program,runs=runs,preflights=preflights),allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    s=json.loads(x.stdout);s.update(lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),round_name=r.name,source_commit=h['source_commit'],
        cpu_checks=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events'],
        closed_round_cost_ledger={p.parent.parent.name:json.loads(p.read_text())['round_gpu_process_residence_seconds'] for p in Path('docs').glob('*/results/AUDIT.json') if 'round_gpu_process_residence_seconds' in json.loads(p.read_text())})
    previous=json.loads(Path('docs/air_apart_isic_hk1_20261012/results/RESULTS.json').read_text())
    for name,v in previous['runs'].items():s['runs'][name]=v
    s['resource_comparison']=json.loads((r/'REPORT_COMPARISON_SPEC.json').read_text())
    s['reporting_cpu_checks']=json.loads((r/'CPU_REPORT_CHECK.json').read_text())
    s['reporting_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    sources={'base':'module_additions1_20261009','local_transport':'module_local2_20261010','residual_drift':'module_drift10_20261011','local_transport_drift':'module_drift10_20261011','AIR':'air_apart_isic_hk1_20261012','APART_full':'air_apart_isic_hk1_20261012'}
    s['cost_sources']={m:json.loads(Path('docs',name,'results/RESULTS.json').read_text())['costs'] for m,name in sources.items()}
    for m in ('DAT_paper','Hao_paper'):s['cost_sources'][m]=s['program']['costs']
    (runtime/'SNAPSHOT_ANONYMOUS.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    if s['program']['status']=='COMPLETE':return report(s,r/'results')
    out=r/'results';out.mkdir();cost=sum(c['residence_seconds'] for c in s['program']['costs']);ledger=s['closed_round_cost_ledger'];ledger[r.name]=cost
    audit=dict(status='INCOMPLETE',source_commit=h['source_commit'],complete_comparison=False,test_accessed=False,
        round_gpu_process_residence_seconds=cost,cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),closed_round_cost_ledger=ledger)
    (out/'PARTIAL_RESULTS.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n');(out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('# DAT/Hao未完整结束\n\n匿名部分聚合、全部成功与失败费用保留。具体去标识原因需私有核对后补充；missing不填0，不以单臂声称完整比较，不自动重跑。\n')
    return audit
