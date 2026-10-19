"""Export aggregate tables and cumulative cost ledger; never load private predictions."""
import csv,json
from pathlib import Path
p=Path(__file__).parent;dst=p/'results/completed';cfg=json.loads((p/'config.json').read_text());r=json.loads((dst/'results.json').read_text());cost=json.loads((dst/'costs.json').read_text());vc=json.loads((dst/'verification_process_cost.json').read_text());wg=json.loads((dst/'WEIGHTED_GROUPS.json').read_text());terminal=json.loads((p/'local_runtime/resume_terminal.private.json').read_text());vf=json.loads((p/'local_runtime/verification_failure1.private.json').read_text())
with (dst/'summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['scope','arm','BA_percent','class_patient_equal_accuracy_percent','head_BA_percent','middle_BA_percent','tail_BA_percent','correct_images'])
 for name,row in [('pooled',r)]+[(f'outer_{i}',a) for i,a in enumerate(r['outer'])]:
  for arm,m in row['metrics'].items():w.writerow([name,arm,100*m['BA'],100*m['PP_accuracy'],100*m['frequency_BA']['head'],100*m['frequency_BA']['middle'],100*m['frequency_BA']['tail'],m['correct']])
with (dst/'per_class.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['class_id','frequency','arm','images','correct','recall_percent'])
 for i,c in enumerate(cfg['order']):
  frequency=next(n for n,m in cfg['frequency_bins'].items() if c in m)
  for arm,m in r['metrics'].items():w.writerow([c,frequency,arm,m['counts'][i],m['correct_by_class'][i],100*m['recall'][i]])
with (dst/'rescue_harm.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['comparison','rescue','harm','BA_rescue_pp','BA_harm_pp','PP_rescue_pp','PP_harm_pp'])
 for name,a in wg['contrasts'].items():w.writerow([name,a['rescue'],a['harm'],100*a['BA_rescue'],100*a['BA_harm'],100*a['PP_rescue'],100*a['PP_harm']])
ledger=dict(cost);ledger.update(resume_worker_body_seconds=cost['wall_seconds'],resume_process_and_guard_seconds=terminal['wall_seconds'],successful_verification_process_seconds=vc['process_wall_seconds'],direct_group_audit_process_seconds=wg['process_wall_seconds'],scale_assertion_diagnostic_tool_seconds=5.143178583,verification_engineering_failures=1,verification_first_failure=dict(error=vf['error'],observed_scale_max_error=vf['observed_scale_max_error'],old_scale_acceptance_tolerance=vf['old_scale_acceptance_tolerance'],wall_seconds=None),new_remote_bytes_at_inventory=115233814,local_artifacts_and_neutral_code_reserved_bytes=100000000,new_space_total_upper_bound_at_inventory=215233814,space_hard_cap_bytes=20000000000,complete_end_to_end_cost_available=False,unmetered_costs=['first verification failure wall','some historical import/connection/preparation overhead'],old_scientific_failure_preserved=True,old_scientific_failure_iteration_count=500,no_additional_fits_after_outer_evaluation=True)
ledger['time_aggregation_note']='sum of recorded process/diagnostic windows; some checks overlap and some overhead is unmetered; not a wall-clock span'
ledger['cumulative_recorded_seconds']=cost['cumulative_measured_seconds']-cost['wall_seconds']+terminal['wall_seconds']+vc['process_wall_seconds']+wg['process_wall_seconds']+ledger['scale_assertion_diagnostic_tool_seconds'];ledger['cumulative_measured_seconds_in_training_receipt']=ledger.pop('cumulative_measured_seconds');(dst/'DELIVERY_COSTS.json').write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(dict(cumulative_recorded_seconds=ledger['cumulative_recorded_seconds'],RSS_GiB=cost['RSS_peak_bytes']/2**30,GPU_GiB=cost['GPU_peak_reserved_bytes']/2**30)))
