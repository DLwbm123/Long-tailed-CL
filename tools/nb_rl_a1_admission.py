"""Measured P0 admission report. Reads receipts only; never launches training."""
import hashlib
import json
import os
from pathlib import Path
import time

root=Path(os.environ['N78_ROOT']);pub=root/'public'
def read(name):return json.loads((pub/name).read_text())
def write(name,value):(pub/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
m=read('P0_MEASUREMENTS.json');b=read('P0_END_TO_END.json');matrix=read('METHOD_MATRIX.json');ledger=read('RESOURCE_LEDGER.json')
tasks=[t for ts in matrix['tasks'].values() for t in ts];batches=sum(t['batches'] for t in tasks)
assert batches==1175 and matrix['tiers']=={'5':35250,'3':21150}
elapsed=time.time()-ledger['wall_T0_unix'];transfer_rate=103361819/14.2245
fixed_bytes=b['rolling']['native']['bytes']-b['synthetic_shape'][0]*1536*8
forecasts=[]
for epochs in (5,3):
 train=batches*epochs*sum(x['step_seconds'] for x in m['timings'])
 diagnostic=12*epochs*sum(x['diagnostic_step_seconds']-x['step_seconds'] for x in m['timings'])
 overhead=b['batch_overhead_seconds']*batches*6*epochs
 loaders=max(0,b['first_loader_batch_seconds']-b['end_to_end_per_batch'])*72*epochs
 extract=batches*12*m['extraction_seconds_per_batch']
 rolling=sum((fixed_bytes+t['n_train']*1536*8)/b['rolling']['native']['bytes'] for t in tasks)*6*epochs*b['rolling']['gzip']['save_seconds']
 snapshot=72*epochs*b['snapshot_seconds']
 init_train=18*b['init_seconds']+72*b['anchor_copy_seconds']
 statistics=72*(b['start_statistics_seconds']+b['end_statistics_seconds'])
 stages=90*m['stage_save_seconds']
 training_phase=train+diagnostic+overhead+loaders+extract+rolling+snapshot+init_train+statistics+72*m['stage_save_seconds']
 frozen=sum(t['batches'] for t in tasks if t['task']>1)*2*m['extraction_seconds_per_batch']+6*b['init_seconds']+18*b['start_statistics_seconds']
 evaluate=sum(t['val_batches'] for t in tasks)*8*m['synthetic_evaluation_seconds_per_batch']+96*b['init_seconds']
 # Conservative serial IO accounting includes asynchronous transfer time in full.
 transfer=90*m['compact_stage_bytes']/transfer_rate
 restore=96*b['rolling']['native']['load_and_equal_seconds']*m['compact_stage_bytes']/b['rolling']['native']['bytes']
 extra_io=18*m['stage_save_seconds']+transfer+restore
 predicted=train+diagnostic+overhead+loaders+extract+rolling+snapshot+init_train+statistics+72*m['stage_save_seconds']+frozen+evaluate+extra_io
 native_saving=rolling*(1-b['rolling']['native']['save_seconds']/b['rolling']['gzip']['save_seconds'])
 forecasts.append(dict(epochs=epochs,optimizer_steps=batches*6*epochs,training_phase_seconds=training_phase,
  predicted_training_finish_hours=(elapsed+training_phase)/3600,remaining_seconds=predicted,
  wall_hours_with_1_3_margin=(elapsed+1.3*predicted)/3600,
  pass_training_deadline=elapsed+training_phase<=45000,pass_wall_deadline=elapsed+1.3*predicted<=57600,
  native_rolling_only_counterfactual_wall_hours=(elapsed+1.3*(predicted-native_saving))/3600,
  components_seconds=dict(updates=train,gradient_diagnostics=diagnostic,batch_loading_hash_transfer=overhead,
   loader_startup=loaders,task_fit_before_after=extract,rolling_save=rolling,snapshot=snapshot,
   train_initialization=init_train,statistics=statistics,trained_stage_save=72*m['stage_save_seconds'],
   frozen=frozen,evaluation=evaluate,other_IO=extra_io)))
assert not any(v['pass_training_deadline'] and v['pass_wall_deadline'] for v in forecasts),'A tier passed: review and launch separately'
report=dict(status='BLOCKED_BUDGET',engineering_checks='PASS',formal_training_started=False,formal_optimizer_steps=0,
 engineering_steps=b['engineering_steps'],wall_T0_unix=ledger['wall_T0_unix'],elapsed_at_admission_seconds=elapsed,
 forecasts=forecasts,forecast_rule='elapsed + 1.3 * measured training/statistics/evaluation/IO <=16h AND unpadded training finish <=12.5h',
 rolling_extrapolation='measured full-state save scaled by native payload size using actual task rows; synthetic dense features used for IO only',
 transfer_accounting='measured target reread transfer throughput; conservatively counted in full despite asynchronous operation',
 native_rolling_counterfactual='measurement only, not applied; changes checkpoint encoding only',
 admitted_epochs=None,validation_images_read=0,test_images_read=0,utility_status='NOT_EVALUABLE',NEXT_DECISION='STOP')
write('P0_ENGINEERING_REPORT.json',report)
config=read('NB_RL_A1_CONFIG_DRAFT.json');config.update(status='ADMISSION_REJECTED_BUDGET',assets_verified=True,training_started=False,epochs=None,
 wall_T0_unix=ledger['wall_T0_unix'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').glob('*.py')},
 parameter_name_lock=read('INITIALIZATION_AUDIT.json'),formal_lock=False)
write('PROTOCOL_LOCK.json',config)
write('RUN_STATUS.json',dict(status='BLOCKED_BUDGET',phase='P0_COMPLETE_NO_FORMAL_RUN',unix=time.time(),formal_steps=0))
write('NEXT_DECISION.json',dict(NEXT_DECISION='STOP',execution_status='BLOCKED_BUDGET',matrix_status='NOT_STARTED',utility_status='NOT_EVALUABLE',publication_status='NOT_AUTHORIZED'))
write('ACCESS_LEDGER.json',dict(formal_fit_images=0,validation_images=0,test_reserved_images=0,old_fit_images=0,future_fit_images=0,
 engineering_steps=b['engineering_steps'],P0_scope='ISIC seed1993 current Task1 fit only; synthetic heads/statistics; no validation image read',
 measured_consumed_train_rows=ledger['counts']['engineering_train_rows'],prefetch_note='Engineering DataLoader may prefetch additional legal Task1 images; consumed-row count is not total opens'))
lines=['# NB-RL-A1 P0 执行报告','', '**状态：BLOCKED_BUDGET。正式训练未启动；完整方法效用 NOT_EVALUABLE。**','',
 '按所提供协议完成资产与初始化审计、六条件吞吐、数学与梯度检查、下一真实批次恢复检查、独立备份恢复检查以及报告合成检查。P0 共 67 次工程更新；仅使用 seed1993 当前 Task1 fit，未读取 validation、test 或未来任务图像。','',
 '|epoch 档|正式更新数|预计训练结束（自 T0）|含 1.3 倍余量总墙钟|准入|','|---|---:|---:|---:|---|']
for v in forecasts:lines.append(f"|{v['epochs']}|{v['optimizer_steps']}|{v['predicted_training_finish_hours']:.2f} h|{v['wall_hours_with_1_3_margin']:.2f} h|不通过|")
lines+=['',f"T0={ledger['wall_T0_unix']}（2026-09-30 18:06:30 北京时间），准入时已用 {elapsed/60:.2f} 分钟；没有重置账本。",
 f"GPU 更新核约 1.41–1.42 秒/批；真实端到端 {b['end_to_end_per_batch']:.3f} 秒/批，批次加载、输入一致性检查与传输等额外开销约 {b['batch_overhead_seconds']:.3f} 秒/批。完整恢复状态压缩写入 {b['rolling']['gzip']['save_seconds']:.2f} 秒，原生写入 {b['rolling']['native']['save_seconds']:.2f} 秒。",
 '预测采用实际各任务 ceil(N/48)，包括分项梯度诊断、首批加载、任务前后特征提取、解析统计、模型初始化、完整 rolling 保存、阶段保存、冻结参照、全矩阵评价、独立传输与恢复读取；传输虽可异步，预测保守全额计入。rolling 按任务实际样本量缩放，IO 使用合成稠密特征，不据此推断模型效果。',
 f"即使仅将 rolling 改为原生非压缩格式，3 epoch 的反事实预测仍为 {forecasts[1]['native_rolling_only_counterfactual_wall_hours']:.2f} h（含余量）；该格式更改未应用到正式执行器。",'',
 '协议要求 3 epoch 也未通过时输出 BLOCKED_BUDGET，因此停止准入。未缩减方法、顺序、任务或 epoch，未延长预算。没有真实 stage/class/paired/bootstrap 结果，不填零，不用工程副本冒充正式结果。','',
 '运行资产保留于两台服务器；源代码和公开 P0 证据另保存于本地工作树。正式六条件四任务执行路径尚未实际运行，因此不能声称完整流水线通过运行验收。NEXT_DECISION=STOP；不创建小时监测，不自动推送 GitHub。独立备份最终状态见 BACKUP_REPORT.json。']
(pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,indent=2))
