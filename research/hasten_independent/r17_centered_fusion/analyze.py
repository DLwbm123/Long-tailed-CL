"""Count-level verification and frozen primary gate for the R17 intervention."""
import json,os,statistics as st
from pathlib import Path
root=Path(os.environ['RUN_ROOT']);out=root/'results';out.mkdir(exist_ok=True)
exit=json.loads((root/'CONTROL_EXIT.json').read_text())
assert exit['exit_code']==0 and exit['reason'] is None
assert exit['total_elapsed_seconds']<600 and exit['allocated_bytes']<100000000
x=json.loads((root/'RESULTS.json').read_text());report={};gate={};checks=0
for ds,data in x.items():
    split=data['split'];tail=[c for c,n in enumerate(split['train_counts']) if n<250];n_stage=len(split['stages'])
    assert len(data['records'])==3*n_stage
    rows=[]
    for seed in [74002,74003,74004]:
        rr=sorted([r for r in data['records'] if r['seed']==seed],key=lambda r:r['stage'])
        assert [r['stage'] for r in rr]==list(range(n_stage))
        for r in rr:
            for key in ['metrics']:
                m=r[key];n=r['seen'];assert m['counts']==split['eval_counts'][:n]
                recall=[100*c/t for c,t in zip(m['correct'],m['counts'])]
                assert abs(st.mean(recall)-m['BA'])<1e-8
                assert abs(st.mean(recall[c] for c in tail if c<n)-m['tail_BA'])<1e-8
                checks+=1
            assert r['H0_fusion_replay_max_error']<1e-7
            assert [c+a-b for c,a,b in zip(r['H0']['correct'],r['rescue'],r['harm'])]==r['metrics']['correct']
        def summarize(k,scale):return dict(mean_BA=st.mean(r[k]['BA']*scale for r in rr),final_BA=rr[-1][k]['BA']*scale,tail_BA=rr[-1][k]['tail_BA']*scale)
        new=summarize('metrics',1);old=summarize('H0',100)
        rows.append(dict(seed=seed,new=new,H0=old,delta_pp={k:new[k]-old[k] for k in new}))
    summary={key:{metric:dict(mean=st.mean(r[key][metric] for r in rows),sd=st.stdev(r[key][metric] for r in rows)) for metric in ['mean_BA','final_BA','tail_BA']} for key in ['new','H0']}
    delta={m:summary['new'][m]['mean']-summary['H0'][m]['mean'] for m in summary['new']}
    paired=sum(all(r['delta_pp'][m]>0 for m in ['final_BA','tail_BA']) for r in rows)
    report[ds]=dict(seeds=rows,summary=summary,delta_pp=delta)
    gate[ds]=dict(delta_pp=delta,paired_both_improved=paired)
g=gate['NIH'];gate['passed']=g['delta_pp']['mean_BA']>=1 and g['delta_pp']['final_BA']>=1 and g['delta_pp']['tail_BA']>=2.857142857 and g['paired_both_improved']>=2 and all(gate[d]['delta_pp'][m]>=-1 for d in ['ISIC','HK'] for m in ['final_BA','tail_BA'])
for f,v in [('ANALYSIS.json',report),('GATE.json',gate),('VERIFICATION.json',dict(count_metric_checks=checks,changed_stage_seed_records=45,reused_H0_training=True,fixed_centering_no_fit=True,no_new_training=True,independent_confirmation=False))]:
    (out/f).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
lines=['# R17：HASTEN原分支平移不变融合','', '保留H0双曲与NCM原始分数；唯一改变为每分支类维中心化后L2→softmax→半半融合。三seed复用原训练状态，无新训练或参数拟合。','', '|数据集|阶段平均BA|最终BA|最终尾类BA|相对H0 Δ平均/最终/尾类pp|','|---|---:|---:|---:|---|']
for ds,r in report.items():
    m=r['summary']['new'];d=r['delta_pp'];lines.append('|%s|%.4f ± %.4f|%.4f ± %.4f|%.4f ± %.4f|%+.4f / %+.4f / %+.4f|'%(ds,m['mean_BA']['mean'],m['mean_BA']['sd'],m['final_BA']['mean'],m['final_BA']['sd'],m['tail_BA']['mean'],m['tail_BA']['sd'],d['mean_BA'],d['final_BA'],d['tail_BA']))
lines+=['', '冻结主门：'+str(gate['passed'])+'。'+('通过本轮开发集门，需继续验证机制稳定性。' if gate['passed'] else '拒绝此固定候选，保留H0；不扫描融合权重、温度或中心化插值来救援此终点。'), '', '逐seed、逐阶段与逐类正确数、救回/损害见RESULTS.json与ANALYSIS.json。结果仅为反复使用的历史开发集；没有TEST、独立泛化或临床证据。', '', 'R17用相同已训练状态隔离推理融合中心化效果，不是增量训练机制的因果消融。若主门失败，应据分支救回/损害分析独立训练机制假设，不能把辅助分支事后晋升为主方法。']
(root/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(gate,indent=2))
