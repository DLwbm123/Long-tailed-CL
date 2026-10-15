"""Public scalar-only four-preflight closeout; never starts scientific work."""
import csv
import json
from pathlib import Path


def main(root):
    data=json.loads((root/'RESULTS.json').read_text());p=data['program'];assert not p['active'] and p['phase']=='STOPPED_NO_TRAIN'
    assert len(data['preflights'])==4 and p['formal_updates']==0 and p['training_allowed'] is False
    costs=p['costs'];rows=[]
    for item in data['preflights']:
        r=item['result'];cost=next((v for v in costs if v['id']==item['id']),None)
        if r:
            assert r['status']=='PASS' and r['optimizer_updates']==r['formal_updates']==r['policy_updates']==0
            assert r['parameters_exactly_unchanged'] and r['buffers_exactly_unchanged'] and not r['test_accessed'] and not r['validation_accessed']
            assert r['guard_highest_bank']['status']=='PASS' and r['PMT_highest_bank']['full_rank_memory_probe']
        rows.append(dict(id=item['id'],status=r['status'] if r else 'MISSING_OR_FAILED',GPU_process_residence_seconds=cost['GPU_process_residence_seconds'] if cost else None,
            peak_allocated_GiB=r['peak_allocated_bytes']/1024**3 if r else None,peak_reserved_GiB=r['peak_reserved_bytes']/1024**3 if r else None,
            gradient_norm=r['gradient_norm'] if r else None,parameters_unchanged=r['parameters_exactly_unchanged'] if r else None,buffers_unchanged=r['buffers_exactly_unchanged'] if r else None,
            highest_old_classes=r['guard_highest_bank']['old_classes'] if r else None,highest_constraints=r['guard_highest_bank']['all_constraints_n'] if r else None,
            guard_host_input_bytes=r['guard_highest_bank']['logical_host_input_bytes'] if r else None))
    good=len(p['completed'])==4 and not p['failures'] and all(v['status']=='PASS' for v in rows)
    if not good:assert (root/'FAILURES_ANON.json').exists(),'Inspect private errors before anonymous failure reporting'
    audit=dict(status='FOUR_PREFLIGHT_PASS_STOPPED_NO_TRAIN' if good else 'PREFLIGHT_INCOMPLETE_STOPPED_NO_TRAIN',
        GPU_operation_records=len(costs),new_GPU_process_residence_seconds=sum(v['GPU_process_residence_seconds'] for v in costs),
        closed_GPU_history_before_seconds=391490.24710153416,closed_GPU_history_including_this_round_seconds=391490.24710153416+sum(v['GPU_process_residence_seconds'] for v in costs),
        whole_round_wall_seconds=p['whole_round_wall_seconds'],CPU_harness_residence_seconds=json.loads((root/'CPU_HARNESS_CHECK.json').read_text())['cost']['CPU_process_residence_seconds'],
        CPU_inside_GPU_residence_not_added=True,formal_training_started=False,formal_updates=0,val_accessed=False,test_accessed=False,
        real_first_batch_separate_from_synthetic_highest_bank=True,failures=p['failures'])
    with (root/'PREFLIGHT_SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# 四条真实零更新 GPU 预检结果','',audit['status'],'', '| 项目 | 状态 | GPU驻留秒 | allocated GiB | reserved GiB | 梯度范数 | 最大旧类/约束 |','|---|---|---:|---:|---:|---:|---:|']
    def show(v):return '未提供' if v is None else f'{v:.9g}' if isinstance(v,float) else str(v)
    for v in rows:lines.append('| '+' | '.join(show(v[k]) for k in ('id','status','GPU_process_residence_seconds','peak_allocated_GiB','peak_reserved_GiB','gradient_norm'))+' | '+show(v['highest_old_classes'])+'/'+show(v['highest_constraints'])+' |')
    lines+=['',json.dumps(audit,ensure_ascii=False,indent=2),'','真实首task第一batch backward与synthetic最大旧bank分别记录；首task没有真实旧bank，不代表真实旧域漂移验证。最大bank shadow中真实模型及梯度仍在GPU驻留；guard沿原CPU full-dual投影，不把传输与CPU等待额外叠加到GPU驻留。全部参数buffer在shadow后再精确比较。四PASS也只停在预检，正式训练0、val/test封存，不构成医学效果或正式全轮预算证明。','', '源码/参数/门未改；公开范围事故与历史路径元数据残留见ENGINEERING_AUDIT，未强推或改写历史。']
    if not good:lines+=['',json.dumps(json.loads((root/'FAILURES_ANON.json').read_text()),ensure_ascii=False)]
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return audit


if __name__=='__main__':
    import sys
    print(json.dumps(main(Path(sys.argv[1]))))
