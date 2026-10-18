"""Summarize both immutable attempts without loading private features or heads."""
import csv
import json
from pathlib import Path


def main():
    root=Path('docs/linear_feasibility1_20261018')
    collected=json.loads((root/'local_runtime/COLLECTED_FINAL.private.json').read_text())
    f=collected['files'];original=f['PROGRAM.json'];compact=f['PROGRAM_COMPACT.json']
    assert original['active'] is None and compact['active'] is None
    assert len(collected['ps'].splitlines()) == 1 and not collected['candidate_exists']
    assert f['solve.json']['solver_status']=='memlimit'
    assert f['solve_compact.json']['solver_status']=='timelimit'
    assert f['solve_compact.json']['solutions']==0
    assert f['verify_compact.json']['status']=='UNRESOLVED_NO_WITNESS'
    attempts=[dict(attempt=name,**v) for name,p in [('dense',original),('compact',compact)] for v in p['costs']]
    cpu=sum(v['CPU_only_residence_seconds'] for v in attempts)
    user=sum(v['CPU_user_seconds'] for v in attempts);system=sum(v['CPU_system_seconds'] for v in attempts)
    prior=json.loads(Path('docs/support_query_tradeoff1_20261018/COSTS.json').read_text())
    solver=f['solve.json']['solver_seconds']+f['solve_compact.json']['solver_seconds']
    elapsed=compact['finished_at']-original['started_at']
    assert solver<=10500 and elapsed<=11700
    assert all(v['GPU_process_residence_seconds']==0 for v in attempts)
    costs=dict(CPU_only_residence_seconds=cpu,CPU_user_seconds=user,CPU_system_seconds=system,
               GPU_process_residence_seconds=0.,peak_worker_RSS_MiB=max(v['peak_RSS_MiB'] for v in attempts),
               compact_peak_worker_RSS_MiB=max(v['peak_RSS_MiB'] for v in compact['costs']),
               combined_solver_seconds=solver,original_elapsed_to_terminal_seconds=elapsed,
               supervisor_wall_seconds_separate=[original['supervisor_wall_seconds'],compact['supervisor_wall_seconds']],
               prior_CPU_residence_seconds=prior['cumulative_CPU_residence_seconds'],
               prior_GPU_residence_seconds=prior['cumulative_GPU_residence_seconds'],
               cumulative_CPU_residence_seconds=prior['cumulative_CPU_residence_seconds']+cpu,
               cumulative_GPU_residence_seconds=prior['cumulative_GPU_residence_seconds'],
               preparation_installation_collection_seconds=None,per_job=attempts)
    results=dict(status='UNRESOLVED_NO_WITNESS_NO_CERTIFICATE',quotas=[329,23,202,2297,2171,36],
                 counts=[375,23,208,2630,2254,36],attempts=[dict(attempt=n,**f[p]) for n,p in [('dense','solve.json'),('compact','solve_compact.json')]],
                 engineering_checks=[f['unit.json'],f['unit_compact.json']],verification=f['verify_compact.json'],
                 all_processes_retired=True,candidate_heads=0,independently_verified_witnesses=0,infeasibility_certified=False,
                 query_access=False,E_access=False,GPU_used=False,
                 original_worker_cleanup_termination_after_durable_result=True,
                 compact_configured_solver_seconds=9000,compact_solver_soft_limit_overrun_seconds=f['solve_compact.json']['solver_seconds']-9000,
                 combined_solver_within_original_budget=True,elapsed_within_original_budget=True,
                 production_family_or_representation_insufficiency_proven=False)
    for name,value in [('COSTS.json',costs),('RESULTS.json',results)]:
        (root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    with (root/'COSTS.csv').open('w') as fo:
        writer=csv.DictWriter(fo,fieldnames=list(attempts[-1]),lineterminator='\n')
        writer.writeheader();writer.writerows(attempts)
    print(json.dumps(dict(CPU_seconds=cpu,GPU_seconds=0,solver_seconds=solver,elapsed_seconds=elapsed,cumulative_CPU=costs['cumulative_CPU_residence_seconds'])))


if __name__=='__main__':main()
