"""Admission from completed A1 residence plus the bounded A2 GPU probes."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

root=Path(os.environ['N78_ROOT']);pub=root/'public';private=root/'private'
cfg=json.loads((private/'INPUT.json').read_text());prior=Path(cfg['prior_run'])
def read(path):return json.loads(Path(path).read_text())
def write(name,value):(pub/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
probe=read(pub/'PARALLEL_PROBE_SUMMARY.json');checks=read(pub/'REPORT_SYNTHETIC_CHECK.json')
assert probe['status']=='PASS' and probe['engineering_steps']==32
assert all(v['status']=='PASS' for k,v in checks.items() if isinstance(v,dict))
assert read(pub/'INDEPENDENT_RESTORE_CHECK.json')['status']=='PASS'
ledger=read(pub/'RESOURCE_LEDGER.json');assert ledger['counts'].get('formal_steps',0)==0
processes=read(prior/'private/PROCESS_LEDGER_formal.json')['processes']
assert len(processes)==29 and all(p['returncode']==0 for p in processes)
old_probe=read(prior/'public/PARALLEL_PROBE_SUMMARY.json')['workers']
selected=[w for w in probe['workers'] if int(w['GPU']) in cfg['gpu_indices']]
factor=max(1.,max(w['core_seconds'] for w in selected)/min(w['core_seconds'] for w in old_probe),
           max(w['diagnostic_seconds'] for w in selected)/min(w['diagnostic_seconds'] for w in old_probe))
train_max=max(p['ended']-p['started'] for p in processes if p['role']=='train')*factor
frozen_max=max(p['ended']-p['started'] for p in processes if p['role']=='frozen')*factor
eval_per_stage=max((p['ended']-p['started'])/len(p['job']['indices']) for p in processes if p['role']=='evaluate')*factor
n_gpu=len(cfg['gpu_indices']);reserve=900.
gpu_remaining=12*train_max+3*frozen_max+60*eval_per_stage+reserve
wall_remaining=math.ceil(12/n_gpu)*train_max+frozen_max+math.ceil(60/n_gpu)*eval_per_stage+reserve
elapsed=time.time()-cfg['wall_T0_unix'];gpu_prior=ledger['GPU_process_residence_seconds']
gpu_margin=(gpu_prior+1.3*gpu_remaining)/3600;wall_margin=(elapsed+1.3*wall_remaining)/3600
training_margin=(elapsed+1.3*math.ceil(12/n_gpu)*train_max)/3600
stage_max=max(v['bytes'] for v in read(prior/'public/ALL_STATES_LOCK.json')['entries'])
rolling=read(prior/'public/P0_END_TO_END.json')['rolling']['native']['bytes']
storage=2*57*stage_max*1.25+n_gpu*rolling*1.25+ledger['persistent_including_backup_bytes']+1024**3
assert gpu_margin<16 and wall_margin<16 and training_margin<12.5,'BLOCKED_ADMISSION_BUDGET'
assert storage<16*1024**3,'BLOCKED_ADMISSION_STORAGE'
report=dict(status='PASS',epochs=3,formal_training_started=False,engineering_steps=32,
    selected_GPU_indices=cfg['gpu_indices'],old_full_trajectory_max_seconds=train_max,
    measured_slowdown_factor=factor,remaining_wall_hours=wall_remaining/3600,
    wall_hours_with_1_3_margin=wall_margin,predicted_GPU_hours=(gpu_prior+gpu_remaining)/3600,
    GPU_hours_with_1_3_margin=gpu_margin,training_finish_hours_with_margin=training_margin,
    storage_forecast_bytes=storage,wall_T0_unix=cfg['wall_T0_unix'],elapsed_at_admission_seconds=elapsed,
    prior_GPU_seconds=gpu_prior,reserve_seconds=reserve,source='completed A1 full worker residence, bounded current A2 probes',
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
(pub/'FINAL_REPORT_ZH.md').write_text('# NB-RL-A2 执行状态\n\n工程与预算准入通过；正式结果尚未生成。\n\n四训练条件、三个顺序、3 epoch；12 条轨迹、60 个逻辑阶段、300 条逐类结果。三卡后台运行，结束后按授权发布聚合结果。\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
