"""Four-GPU admission amendment; preserves T0 and aggregate GPU-hour limit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

root=Path(os.environ['N78_ROOT']);pub=root/'public';private=root/'private'
def read(name):return json.loads((pub/name).read_text())
def write(name,value):(pub/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
old=read('P0_ENGINEERING_REPORT.json');assert old['status']=='BLOCKED_BUDGET'
probe=read('PARALLEL_PROBE_SUMMARY.json');restore=read('PARALLEL_NATIVE_RECOVERY_CHECK.json')
assert probe['status']==restore['status']=='PASS'
ledger=read('RESOURCE_LEDGER.json');m=read('P0_MEASUREMENTS.json');b=read('P0_END_TO_END.json')
assert ledger['counts']['engineering_steps']==110 and ledger['counts'].get('formal_steps',0)==0
archive=private/'single_gpu_admission';archive.mkdir(exist_ok=False)
for path in pub.iterdir():shutil.copyfile(path,archive/path.name)
now=time.time();elapsed=now-ledger['wall_T0_unix'];GPU_prior=ledger['GPU_process_residence_seconds']
base_R=next(x for x in m['timings'] if x['method']=='R')
# Never credit a measured speedup when admitting the full matrix.
compute_factor=max(1.,max(w['core_seconds'] for w in probe['workers'])/base_R['step_seconds'])
overhead_factor=max(1.,max(w['extra_seconds'] for w in probe['workers'])/b['batch_overhead_seconds'])
diag_factor=max(1.,max(w['diagnostic_seconds'] for w in probe['workers'])/base_R['diagnostic_step_seconds'])
forecasts=[]
for previous in old['forecasts']:
 c=dict(previous['components_seconds']);epochs=previous['epochs']
 c['updates']*=compute_factor;c['gradient_diagnostics']*=diag_factor;c['batch_loading_hash_transfer']*=overhead_factor
 c['rolling_save']*=b['rolling']['native']['save_seconds']/b['rolling']['gzip']['save_seconds']
 # Launch-to-exit accounting includes interpreter imports for all 29 worker jobs.
 c['worker_import_and_exit_reserve']=29*15.
 c['report_and_final_backup_reserve']=600.
 train=sum(c[k] for k in ('updates','gradient_diagnostics','batch_loading_hash_transfer','loader_startup','task_fit_before_after','rolling_save','snapshot','train_initialization','statistics','trained_stage_save'))
 remaining=sum(c.values())
 # Eighteen similar-cost trajectories on four devices: at most five per device.
 train_wall=train*5/18+18*15
 wall_remaining=train_wall+c['frozen']/3+c['evaluation']/4+c['other_IO']+11*15+c['report_and_final_backup_reserve']
 total_GPU_hours=(GPU_prior+remaining)/3600
 wall_margin_hours=(elapsed+1.3*wall_remaining)/3600
 training_finish_hours=(elapsed+train_wall)/3600
 forecasts.append(dict(epochs=epochs,optimizer_steps=previous['optimizer_steps'],predicted_GPU_hours=total_GPU_hours,
   remaining_wall_hours=wall_remaining/3600,wall_hours_with_1_3_margin=wall_margin_hours,training_finish_hours=training_finish_hours,
   pass_GPU=total_GPU_hours<=16,pass_wall=wall_margin_hours<=16,pass_training=training_finish_hours<=12.5,
   components_GPU_seconds=c))
chosen=next((v['epochs'] for v in forecasts if v['pass_GPU'] and v['pass_wall'] and v['pass_training']),None)
assert chosen==3,'BLOCKED_PARALLEL_ADMISSION'
primary_stage_budget=90*m['compact_stage_bytes']*1.25
storage=2*primary_stage_budget+4*b['rolling']['native']['bytes']*1.2+ledger['persistent_including_backup_bytes']+512*1024**2
assert storage<16*1024**3,'BLOCKED_STORAGE'
authorization=dict(user_instruction='没事，你可以用 4 个 GPU 并行',unix=now,GPU_indices=[0,1,2,3],GPU_count=4,
 wall_T0_unix=ledger['wall_T0_unix'],cumulative_GPU_hours_max=16,wall_hours_max=16,training_stop_hour=12.5,forward_stop_hour=14.5,
 scientific_matrix_unchanged=True,publication_authorized=False,monitor_automation_authorized=False,
 rolling_encoding='native torch.save, atomic replace; full-state equality and next-update recovery passed')
write('PARALLEL_AUTHORIZATION.json',authorization)
report=dict(status='PASS',engineering_status='PASS',matrix_status='NOT_STARTED',utility_status='NOT_EVALUABLE',epochs=chosen,
 formal_training_started=False,engineering_steps=110,forecasts=forecasts,storage_forecast_bytes=storage,
 admission_wall_margin=1.3,GPU_margin_policy='Cumulative launch-to-exit residence has an independent 16h hard stop; 1.3 admission margin applies to wall forecast as specified',
 prior_GPU_seconds=GPU_prior,wall_T0_unix=ledger['wall_T0_unix'],elapsed_at_admission_seconds=elapsed,
 parallel_probe=probe,native_recovery=restore,original_single_GPU_gate='private/single_gpu_admission/P0_ENGINEERING_REPORT.json',
 schedule='18 independent train trajectories, global state seal, 6 frozen trajectories, global seal, 4 evaluation shards, paired report',
 runtime_safety='Supervisor sums all child residence, checks every 2 seconds, reserves 60 seconds before GPU cap and 30 seconds before wall cap',
 image_scope='No validation images read during admission; only seed1993 Task1 fit plus synthetic data')
write('P0_ENGINEERING_REPORT.json',report)
config=read('NB_RL_A1_CONFIG_DRAFT.json');config.update(status='LOCKED',assets_verified=True,training_started=False,epochs=chosen,
 wall_T0_unix=ledger['wall_T0_unix'],formal_lock=True,
 source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').glob('*.py')},
 parameter_name_lock=read('INITIALIZATION_AUDIT.json'),parallel_authorization=authorization,
 batch_equality='All per-batch fingerprints compared across six methods before validation unlock',
 checkpoint_format='compact stages gzip, rolling native full-state per worker')
config['budgets']['gpu_count']=4
write('PROTOCOL_LOCK.json',config)
write('RUN_STATUS.json',dict(status='ADMITTED_AWAITING_LAUNCH',unix=now,epochs=chosen,GPU_count=4))
write('NEXT_DECISION.json',dict(NEXT_DECISION='RUN_LOCKED_MATRIX_THEN_STOP',engineering_status='PASS',matrix_status='NOT_STARTED',utility_status='NOT_EVALUABLE',retention_signal='NOT_EVALUABLE',sampling_added_value='NOT_EVALUABLE',frozen_comparison_status='NOT_EVALUABLE'))
(pub/'FINAL_REPORT_ZH.md').write_text('# NB-RL-A1 四卡执行状态\n\n四卡准入通过，锁定 3 epoch；正式矩阵尚未完成，无方法效果结论。\n\n保留原 T0、16 小时墙钟和 16 GPU 小时总驻留上限。全部 18 条训练轨迹、两个冻结参照与统一封存评价保持不变。最新状态见 RUN_STATUS.json，预算见 P0_ENGINEERING_REPORT.json。\n')
print(json.dumps(dict(status='PASS',epochs=chosen,forecasts=forecasts,storage_GiB=storage/1024**3),indent=2))
