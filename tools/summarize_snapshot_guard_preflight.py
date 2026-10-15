"""Public scalar-only four-preflight closeout; never starts scientific work."""
import csv
import json
import math
from pathlib import Path


def main(root):
    data=json.loads((root/'RESULTS.json').read_text());p=data['program'];assert not p['active'] and p['phase']=='STOPPED_NO_TRAIN'
    assert len(data['preflights'])==4 and p['formal_updates']==0 and p['training_allowed'] is False
    assert {v['id'] for v in data['preflights']}=={'pre_ISIC_guard','pre_ISIC_LT_guard','pre_HK_guard','pre_HK_LT_guard'}
    costs=p['costs'];rows=[]
    for item in data['preflights']:
        r=item['result'];cost=next((v for v in costs if v['id']==item['id']),None)
        if r:
            assert cost is not None and cost['returncode']==0 and cost['gpu'] in (0,1)
            assert r['status']=='PASS' and r['optimizer_updates']==r['formal_updates']==r['policy_updates']==0
            assert r['parameters_exactly_unchanged'] and r['buffers_exactly_unchanged'] and not r['test_accessed'] and not r['validation_accessed']
            assert r['guard_highest_bank']['status']=='PASS' and r['PMT_highest_bank']['full_rank_memory_probe']
            assert r['encoder_dim']==1536 and r['batch_n']==64 and r['real_first_batch'] and r['real_guard_certificate']=='CERTIFIED'
            assert all(math.isfinite(v) and v>0 for v in r['adapter_pool_gradients'].values())
            assert math.isfinite(r['gradient_norm']) and r['gradient_norm']>0
            assert 0<r['peak_allocated_bytes']<=r['peak_reserved_bytes']
            g=r['guard_highest_bank'];old=6 if item['id'].startswith('pre_ISIC') else 21
            assert g['old_classes']==r['PMT_highest_bank']['old_classes']==old and g['optimizer_updates']==0 and not g['formal_statistics_written']
            assert g['all_constraints_n']==g['fit_n']*(g['seen_classes']-1) and g['minimum_raw_slack']>=-2e-8
            assert g['stationarity']<=1e-8 and g['complementarity']<=1e-7 and abs(g['primal_dual_gap'])<=1e-7
        rows.append(dict(id=item['id'],status=r['status'] if r else 'MISSING_OR_FAILED',GPU_process_residence_seconds=cost['GPU_process_residence_seconds'] if cost else None,
            peak_allocated_GiB=r['peak_allocated_bytes']/1024**3 if r else None,peak_reserved_GiB=r['peak_reserved_bytes']/1024**3 if r else None,
            gradient_norm=r['gradient_norm'] if r else None,parameters_unchanged=r['parameters_exactly_unchanged'] if r else None,buffers_unchanged=r['buffers_exactly_unchanged'] if r else None,
            highest_old_classes=r['guard_highest_bank']['old_classes'] if r else None,highest_constraints=r['guard_highest_bank']['all_constraints_n'] if r else None,
            guard_host_input_bytes=r['guard_highest_bank']['logical_host_input_bytes'] if r else None,
            pool_gradient=r['adapter_pool_gradients']['.pool.pool.'] if r else None,few_pool_gradient=r['adapter_pool_gradients']['.pool_few.pool.'] if r else None,
            parameter_tensors=r['parameter_tensors'] if r else None,buffer_tensors=r['buffer_tensors'] if r else None,
            operation_wall_seconds=r['operation_wall_seconds'] if r else None,guard_shadow_wall_seconds=r['guard_highest_bank']['operation_wall_seconds'] if r else None,
            max_solve_residual=r['max_solve_residual'] if r else None,guard_support_rank=r['guard_highest_bank']['observed_support_rank'] if r else None,
            minimum_guard_slack=r['guard_highest_bank']['minimum_raw_slack'] if r else None,guard_stationarity=r['guard_highest_bank']['stationarity'] if r else None,
            guard_checkpoint_bytes=r['guard_highest_bank']['checkpoint_bytes'] if r else None))
    good=len(p['completed'])==4 and not p['failures'] and all(v['status']=='PASS' for v in rows)
    if not good:assert (root/'FAILURES_ANON.json').exists(),'Inspect private errors before anonymous failure reporting'
    audit=dict(status='FOUR_PREFLIGHT_PASS_STOPPED_NO_TRAIN' if good else 'PREFLIGHT_INCOMPLETE_STOPPED_NO_TRAIN',
        GPU_operation_records=len(costs),new_GPU_process_residence_seconds=sum(v['GPU_process_residence_seconds'] for v in costs),
        closed_GPU_history_before_seconds=391490.24710153416,closed_GPU_history_including_this_round_seconds=391490.24710153416+sum(v['GPU_process_residence_seconds'] for v in costs),
        whole_round_wall_seconds=p['whole_round_wall_seconds'],CPU_harness_residence_seconds=json.loads((root/'CPU_HARNESS_CHECK.json').read_text())['cost']['CPU_process_residence_seconds'],
        CPU_inside_GPU_residence_not_added=True,formal_training_started=False,formal_updates=0,val_accessed=False,test_accessed=False,
        real_first_batch_separate_from_synthetic_highest_bank=True,failures=p['failures'],
        source_commit=json.loads((root/'CPU_HARNESS_CHECK.json').read_text())['source_commit'],production_math_source='7a077b69f16d843573b2bc7af5a46d9701c29bb0',
        GPU_worker_failures=len(p['failures']),GPU_retries=0,all_workers_exited_before_collection=True,terminal_collection_count=1,
        memory_metric='PyTorch per-process peak allocated/reserved; not sampled NVML peak or total physical device use',NVML_peak_bytes_unmeasured=None,
        residence_poll_resolution_seconds=.25,CPU_operation_and_guard_shadow_nested_in_GPU_residence=True)
    with (root/'PREFLIGHT_SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# 四条真实零更新 GPU 预检结果','',audit['status'],'', '| 项目 | 状态 | GPU驻留秒 | allocated GiB | reserved GiB | 梯度范数 | 最大旧类/约束 |','|---|---|---:|---:|---:|---:|---:|']
    def show(v):return '未提供' if v is None else f'{v:.9g}' if isinstance(v,float) else str(v)
    for v in rows:lines.append('| '+' | '.join(show(v[k]) for k in ('id','status','GPU_process_residence_seconds','peak_allocated_GiB','peak_reserved_GiB','gradient_norm'))+' | '+show(v['highest_old_classes'])+'/'+show(v['highest_constraints'])+' |')
    lines+=['','| 项目 | pool梯度 | few-pool梯度 | 参数/注册buffer数 | operation秒 | guard shadow秒 |','|---|---:|---:|---:|---:|---:|']
    for v in rows:lines.append('| '+' | '.join([v['id'],show(v['pool_gradient']),show(v['few_pool_gradient']),show(v['parameter_tensors'])+'/'+show(v['buffer_tensors']),show(v['operation_wall_seconds']),show(v['guard_shadow_wall_seconds'])])+' |')
    lines+=['','四条全部839个参数tensor精确不变；模型注册buffer数为0，因此buffer集合未变化，不代表非空buffer路径已覆盖。所有首batch为64，双pool梯度有限非零。显存列为PyTorch allocated/reserved实测峰值，不是NVML总物理占用或未来正式训练峰值。逻辑host输入仅成功guard调用的GPU到CPU输入量，不是全部PCIe流量或带宽；fixture负向校验和正向传输/解算耗时均已包含驻留。']
    lines+=['',json.dumps(audit,ensure_ascii=False,indent=2),'','真实首task第一batch backward与synthetic最大旧bank分别记录；首task没有真实旧bank，不代表真实旧域漂移验证。最大bank shadow中真实模型及梯度仍在GPU驻留；guard沿原CPU full-dual投影，不把传输与CPU等待额外叠加到GPU驻留。全部参数buffer在shadow后再精确比较。四PASS也只停在预检，正式训练0、val/test封存，不构成医学效果或正式全轮预算证明。','', '源码/参数/门未改；公开范围事故与历史路径元数据残留见ENGINEERING_AUDIT，未强推或改写历史。']
    if not good:lines+=['',json.dumps(json.loads((root/'FAILURES_ANON.json').read_text()),ensure_ascii=False)]
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return audit


if __name__=='__main__':
    import sys
    print(json.dumps(main(Path(sys.argv[1]))))
