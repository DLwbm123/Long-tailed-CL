"""Admission for the fixed replication matrix using prior complete worker residence."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

root=Path(os.environ['N78_ROOT']);pub=root/'public';private=root/'private'
def read(path):return json.loads(Path(path).read_text())
def write(name,value):(pub/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
cfg=read(private/'INPUT.json');prior=Path(cfg['prior_run'])
probe=read(pub/'PARALLEL_PROBE_SUMMARY.json');checks=read(pub/'REPORT_SYNTHETIC_CHECK.json')
assert probe['status']=='PASS' and probe['engineering_steps']==32
assert all(v['status']=='PASS' for v in checks.values())
assert read(pub/'INDEPENDENT_RESTORE_CHECK.json')['status']=='PASS'
workers={(v['method'],v['seed']):v for v in probe['workers']}
for key in ('initial_delta_sha256','first_batch_sha256'):
    assert len({workers[m,1993][key] for m in ('A','G','R','E')})==1
write('SEED_PAIRING_CHECK.json',dict(status='PASS',independent_order_and_training_seeds=True,
    paired_initialization_and_batches_identical=True,training_seed=74001,validation_images=0))
ledger=read(pub/'RESOURCE_LEDGER.json');assert ledger['counts']['formal_steps']==0
processes=read(prior/'private/PROCESS_LEDGER_formal.json')['processes']
assert len(processes)==22 and all(p['returncode']==0 for p in processes)
old_probe=read(prior/'public/PARALLEL_PROBE_SUMMARY.json')['workers']
old_probe=[w for w in old_probe if int(w['GPU']) in cfg['gpu_indices']]
factor=max(1.,max(w['core_seconds'] for w in workers.values())/min(w['core_seconds'] for w in old_probe),
           max(w['diagnostic_seconds'] for w in workers.values())/min(w['diagnostic_seconds'] for w in old_probe))
train_max=max(p['ended']-p['started'] for p in processes if p['role']=='train')*factor
frozen_max=max(p['ended']-p['started'] for p in processes if p['role']=='frozen')*factor
eval_per_stage=max((p['ended']-p['started'])/len(p['job']['indices']) for p in processes if p['role']=='evaluate')*factor
n_gpu=len(cfg['gpu_indices']);reserve=900.
gpu_remaining=12*train_max+3*frozen_max+60*eval_per_stage+reserve
wall_remaining=math.ceil(12/n_gpu)*train_max+math.ceil(3/n_gpu)*frozen_max+math.ceil(60/n_gpu)*eval_per_stage+reserve
elapsed=time.time()-cfg['wall_T0_unix'];gpu_prior=ledger['GPU_process_residence_seconds']
gpu_margin=(gpu_prior+1.3*gpu_remaining)/3600;wall_margin=(elapsed+1.3*wall_remaining)/3600
training_margin=(elapsed+1.3*math.ceil(12/n_gpu)*train_max)/3600
stage_max=max(v['bytes'] for v in read(prior/'public/ALL_STATES_LOCK.json')['entries'])
rolling=read(prior.parent/'nb_rl_a1_20260930/public/P0_END_TO_END.json')['rolling']['native']['bytes']
storage=2*57*stage_max*1.25+n_gpu*rolling*1.25+ledger['persistent_including_backup_bytes']+1024**3
assert gpu_margin<16 and wall_margin<16 and training_margin<12.5,'BLOCKED_ADMISSION_BUDGET'
assert storage<16*1024**3,'BLOCKED_ADMISSION_STORAGE'
report=dict(status='PASS',epochs=3,formal_training_started=False,engineering_steps=32,
    selected_GPU_indices=cfg['gpu_indices'],old_full_trajectory_max_seconds=train_max,
    measured_slowdown_factor=factor,remaining_wall_hours=wall_remaining/3600,
    wall_hours_with_1_3_margin=wall_margin,predicted_GPU_hours=(gpu_prior+gpu_remaining)/3600,
    GPU_hours_with_1_3_margin=gpu_margin,training_finish_hours_with_margin=training_margin,
    storage_forecast_bytes=storage,wall_T0_unix=cfg['wall_T0_unix'],elapsed_at_admission_seconds=elapsed,
    prior_GPU_seconds=gpu_prior,reserve_seconds=reserve,source='completed A3 worker residence and current A5 reward probes',
    no_validation_images=True,no_historical_trained_checkpoint_loaded=True,unix=time.time())
write('P0_ENGINEERING_REPORT.json',report)
lock=read(pub/'EXECUTION_AUTHORIZATION.json');lock.update(state='LOCKED',formal_lock=True,
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').glob('*.py')},
    actual_source_commit=cfg['source_commit'],source_asset_lock='SOURCE_AND_ASSET_LOCK.json',
    locked_before_first_update_unix=time.time(),wall_deadline_unix=cfg['wall_T0_unix']+57600,
    environment_lock=read(pub/'ENVIRONMENT_LOCK.json'))
write('PROTOCOL_LOCK.json',lock)
write('RUN_STATUS.json',dict(status='ADMITTED_AWAITING_LAUNCH',epochs=3,GPU_indices=cfg['gpu_indices'],unix=time.time()))
write('NEXT_DECISION.json',dict(NEXT_DECISION='RUN_LOCKED_MATRIX_THEN_STOP',matrix_status='NOT_STARTED',utility_status='NOT_EVALUABLE',publication_status='AUTHORIZED'))
(pub/'FINAL_REPORT_ZH.md').write_text('# NB-RL-A5 执行状态\n\n工程和预算准入通过，正式结果尚未生成。12 条训练轨迹、3 条冻结轨迹，60 阶段/300逐类，完成后自动发布。\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
