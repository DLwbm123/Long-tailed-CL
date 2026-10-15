"""Anonymous stdlib report; consumes completed CPU scalars only."""
import csv
import json
import subprocess
from pathlib import Path


def summarize(root):
    lock=json.loads((root/'DESIGN_LOCK.json').read_text());data=json.loads((root/'CPU_RESULTS.json').read_text())
    program=data['program'];assert program['status']=='COMPLETE_CPU_CHECKS' and program['active'] is None
    assert len(program['cases'])==len(lock['cases'])==len(data['results'])
    assert [c['id'] for c in program['cases']]==[c['id'] for c in lock['cases']]
    results={c['id']:v for c,v in zip(lock['cases'],data['results'])}
    costs={v['id']:v for v in program['costs']};rows=[];growth=[];small=[]
    fields=['id','kind','event','status','fit_n','dimension','classes','support_rank','constraints','positive_duals','iterations','inverse_calls','minimum_raw_slack','stationarity','complementarity','primal_dual_gap','available_gain','certified_gain','retained_gain','discriminative_delta','identity','synthetic_rescued','synthetic_damaged','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB']
    for cfg in lock['cases']:
        ident=cfg['id'];v=results[ident];a=v.get('result') or {};meta=v.get('meta',{});c=costs.get(ident,{})
        status=a.get('status',v.get('status','missing'))
        row=dict(id=ident,kind=cfg['kind'],event=cfg.get('event'),status=status,fit_n=cfg.get('fit_n'),dimension=cfg.get('dimension'),classes=cfg.get('classes'),support_rank=meta.get('observed_support_rank'),constraints=a.get('constraint_count'),positive_duals=a.get('positive_duals'),iterations=a.get('iterations'),inverse_calls=a.get('H_inverse_calls'),minimum_raw_slack=a.get('minimum_raw_slack'),stationarity=a.get('stationarity'),complementarity=a.get('complementarity'),primal_dual_gap=a.get('primal_dual_gap'),available_gain=a.get('available_gain'),certified_gain=a.get('certified_gain'),retained_gain=a.get('retained_gain'),discriminative_delta=a.get('discriminative_delta'),identity=a.get('identity'),synthetic_rescued=a.get('synthetic_rescued'),synthetic_damaged=a.get('synthetic_damaged'))
        row.update({k:c.get(k) for k in fields if k.startswith('CPU_') or k=='peak_RSS_MiB'});rows.append(row)
        if cfg['kind']=='small':small.append(v)
        for t in a.get('trace',[]):growth.append(dict(id=ident,**t))
        if a.get('status')=='CERTIFIED':
            assert a['minimum_raw_slack']>=-lock['slack_tolerance'] and a['stationarity']<=lock['stationarity_tolerance']
            assert a['complementarity']<=lock['complementarity_tolerance']
            assert abs(a['primal_dual_gap'])<=1e-7*(1+abs(a['objective_candidate']))
        elif cfg['kind']!='small':assert row['identity'] is None and row['certified_gain'] is None
    def write(name,records,columns):
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(records)
    write('CPU_SCALE_SUMMARY.csv',rows,fields)
    tracefields=['id','iteration','evaluations','dual_objective','minimum_raw_slack','positive_duals','solver_seconds']
    write('ACTIVE_DUAL_GROWTH.csv',growth,tracefields)
    costs_list=program['costs'];audit=dict(design_source_commit=program['source_commit'],reporting_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),GPU_seconds=0,closed_historical_GPU_seconds=391490.24710153416,CPU_independent_process_residence_seconds=sum(v['CPU_process_residence_seconds'] for v in costs_list),CPU_user_seconds=sum(v['CPU_user_seconds'] for v in costs_list),CPU_system_seconds=sum(v['CPU_system_seconds'] for v in costs_list),peak_RSS_MiB=max((v['peak_RSS_MiB'] for v in costs_list),default=None),supervisor_wall_seconds=program['supervisor_wall_seconds'],nested_pipeline_operation_not_added=True,failures=program['failures'],GPU_ready=False,production_integrated=False,medical_images_read=0,medical_checkpoints_read=0,val_reads=0,test_accessed=False,optimizer_updates=0)
    (root/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (root/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(status='PASS',case_rows=len(rows),growth_rows=len(growth),all_certificates_checked=True,missing_not_zero=True,scientific_effect_claim=False),indent=2)+'\n')
    text=['# 固定快照全 fit 守卫：完整 CPU 结果','',f"科学源 `{program['source_commit']}`。全部 {len(rows)} 个 CPU case 已终态。无 GPU、医学图像/checkpoint、val/test、optimizer；不是正式医学效果。",'', '| case | 状态 | 约束/正 dual | 全点最小 slack | 已认证收益 | 判别变化 | 驻留秒 | peak MiB |','|---|---|---|---|---|---|---|---|']
    def f(v):return 'missing' if v is None else f'{v:.9g}' if isinstance(v,float) else str(v)
    for v in rows:text.append('| '+' | '.join([v['id'],v['status'],f(v['constraints'])+'/'+f(v['positive_duals']),f(v['minimum_raw_slack']),f(v['certified_gain']),f(v['discriminative_delta']),f(v['CPU_process_residence_seconds']),f(v['peak_RSS_MiB'])])+' |')
    text+=['','全部详细目标/保留比例/identity/合成救回破坏、KKT/gap、支持秩与逐次资源见 CPU_SCALE_SUMMARY.csv；活跃增长见 ACTIVE_DUAL_GROWTH.csv。未解的最后迭代不是可用头，不填 identity 或0。小枚举、排列/公共logit、关闭精确与证书结果完整保留于 CPU_RESULTS.json。','',f"独立 CPU 驻留合计 {audit['CPU_independent_process_residence_seconds']:.9f} 秒；user {audit['CPU_user_seconds']:.9f} / system {audit['CPU_system_seconds']:.9f} 秒；peak {f(audit['peak_RSS_MiB'])} MiB；父 wall {audit['supervisor_wall_seconds']:.9f} 秒。pipeline/operation 为包含区间，不相加。历史 GPU 不变。",'', '## 限制','', '保护仅固定 encoder/frame 的 epoch 初始化与最终 boundary fit 支持，相对同状态 OFF；encoder 更新后失效。合成判别变化和二次收益不证明医学分类收益。已知旧 affine 不验证真实运输外推。checkpoint force 可检查 stationarity，无全支持不能独立复验所有 margin。未接生产，不是 GPU-ready。所有失败与未运行范围如实保留，不能用单 case 通过代替整体可行。']
    if program['failures']:
        assert (root/'CPU_FAILURES_ANON.json').exists(),'Privately inspect and sanitize failures before report'
        text+=['','工程失败摘要：',json.dumps(json.loads((root/'CPU_FAILURES_ANON.json').read_text()),ensure_ascii=False)]
    (root/'REPORT_ZH.md').write_text('\n'.join(text)+'\n')
    return dict(cases=len(rows),certified=sum(v['status']=='CERTIFIED' for v in rows),unresolved=sum(v['status']=='UNRESOLVED' for v in rows),audit=audit)

if __name__=='__main__':
    import sys
    print(json.dumps(summarize(Path(sys.argv[1])),ensure_ascii=False))
