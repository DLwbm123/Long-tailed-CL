"""Collect only anonymous aggregate external-method results, diagnostics and costs."""
import json
from pathlib import Path
import subprocess
import sys

from summarize_external_methods import report


def collect(round_dir):
    r=Path(round_dir);h=json.loads((r/'local_runtime/HANDOFF.private.json').read_text())
    if (r/'results').exists() or (r/'local_runtime/DELIVERY.private.json').exists():
        raise ValueError('No repeat collection/delivery')
    code='import json\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
program={k:p[k] for k in ('status','phase','started','ended','elapsed_seconds','active','costs','whole_round_wall_seconds','resource_wait_seconds') if k in p}
program['failures']=[{k:v for k,v in c.items() if k in ('job','operation','gpu','returncode','residence_seconds')} for c in p.get('failures',[])]
runs={};preflights={}
for ds in ('ISIC','HK'):
 for method in ('AIR','APART_full'):
  name=ds+'_'+method;folder=root/'runs'/name
  if (folder/'STATUS.json').exists():
   raw=json.loads((folder/'STATUS.json').read_text());s={k:v for k,v in raw.items() if k in ('status','phase','steps','analytic_batches','stages','policy_updates','test_accessed')}
   s['error_present']=bool(raw.get('error'));v=dict(status=s)
   for filename,key in [('metrics.json','metrics'),('MODEL_COST.json','model_cost')]:
    if (folder/filename).exists():v[key]=json.loads((folder/filename).read_text())
   if (folder/'diagnostics.json').exists():
    v['scalar_diagnostics']=[{k:x for k,x in d.items() if type(x) in (int,float,bool) or (k=='active_lrs' and isinstance(x,list) and len(x)==2 and all(type(z) in (int,float) for z in x))} for d in json.loads((folder/'diagnostics.json').read_text())]
   runs[name]=v
  folder=root/'preflight'/name
  if (folder/'PREFLIGHT.json').exists():preflights[name]=json.loads((folder/'PREFLIGHT.json').read_text())
print(json.dumps(dict(program=program,runs=runs,preflights=preflights),allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    s=json.loads(x.stdout);s.update(lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),round_name=r.name,
        source_commit=h['source_commit'],cpu_checks=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),
        closed_round_cost_ledger={f.parent.parent.name:json.loads(f.read_text())['round_gpu_process_residence_seconds']
            for f in Path('docs').glob('*/results/AUDIT.json') if 'round_gpu_process_residence_seconds' in json.loads(f.read_text())})
    controls=json.loads(Path('docs/module_drift10_20261011/results/RESULTS.json').read_text())
    for name,v in controls['runs'].items():s['runs'][name]=v
    snapshot=r/'local_runtime/SNAPSHOT_ANONYMOUS.json';snapshot.write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    if s['program']['status']=='COMPLETE':return report(s,r/'results')
    out=r/'results';out.mkdir();cost=sum(c['residence_seconds'] for c in s['program']['costs'])
    ledger=s['closed_round_cost_ledger'];ledger[r.name]=cost
    audit=dict(status='INCOMPLETE',source_commit=h['source_commit'],complete_paired_comparison=False,
        round_gpu_process_residence_seconds=cost,cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),
        closed_round_cost_ledger=ledger,failed_subprocess_cost_records=sum(c['returncode']!=0 for c in s['program']['costs']),test_accessed=False)
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'PARTIAL_RESULTS.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    (out/'REPORT_ZH.md').write_text('# AIR/APART未完整结束\n\n全部匿名部分聚合与成功/失败成本保留；缺失不填0、不以单臂称配对成功。私有核对错误后补充去标识具体原因和未完成范围；不自动retry。\n')
    return audit


if __name__=='__main__':print(json.dumps(collect(json.load(sys.stdin)['round_dir']),indent=2))
