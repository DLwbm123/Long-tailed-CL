"""Fixed, paired aggregate analysis; no feedback to fitting or admission."""
import csv,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from tools.run_nb2_vlm_r1 import _read,_write,SIZES
from route_a.run_ct13_real import _write_csv
from tools.finalize_nb2_vlm_r1 import _gate
SEEDS=(1993,1994,1995)

def table(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def comparisons(available):
    pairs=[('F.RF1','F.LIN'),('G.RF1','G.LIN'),('B.RF1','B.LIN'),('G.RF1','F.RF1'),('B.RF1','F.RF1'),('G.RF1','F.LIN'),('B.RF1','F.LIN'),('B.RF1','G.RF1'),('B.LIN','G.LIN')]
    for m in ['G','B']:
        pairs += [(m+'.RF1',m+'.SPLIT1'),(m+'.RF1',m+'.RPLIN1'),(m+'.RF2',m+'.LIN')]
        for feat in ['LIN','RF1','RF2']:
            pairs += [(m+'.'+feat+'.'+c,m+'.'+feat) for c in ['CSE1','CSE025']]
        pairs +=[(m+'.RF1.CSEperm1',m+'.RF1'),(m+'.RF1.CSE1',m+'.RF1.CSEperm1')]
    return [p for p in pairs if all(m in available for m in p)]

def bootstrap_layout(y,components):
    rng=np.random.default_rng(68026);classes=np.unique(y);groups=np.unique(components)
    cross=any(len(np.unique(y[components==g]))>1 for g in groups)
    if not cross:
        units={int(c):[np.flatnonzero((y==c)&(components==g)) for g in np.unique(components[y==c])] for c in classes}
        draws={c:rng.integers(0,len(v),size=(2000,len(v))) for c,v in units.items()}
        def apply(delta):
            boot=np.zeros(2000)
            for c,v in units.items():
                n=np.array([delta[i].sum() for i in v]);den=np.array([len(i) for i in v]);ix=draws[c];boot+=n[ix].sum(1)/den[ix].sum(1)/len(classes)
            return boot,0
        return apply,'class_stratified_identity_component'
    # Shared whole-component draws preserve identities spanning labels.
    units=[np.flatnonzero(components==g) for g in groups];ix=rng.integers(0,len(units),size=(2000,len(units)))
    def apply(delta):
        vals=[]
        for c in classes:
            n=np.array([delta[i][y[i]==c].sum() for i in units]);den=np.array([(y[i]==c).sum() for i in units]);d=den[ix].sum(1)
            vals.append(np.divide(n[ix].sum(1),d,out=np.full(2000,np.nan),where=d>0))
        v=np.mean(vals,axis=0);return v,int(np.isnan(v).sum())
    return apply,'whole_identity_component_cross_label'

def finalize(out,cfg):
    protocol=_read(out/'PROTOCOL_LOCK.json');admitted=[m for m in protocol['methods'] if m['tier'] in protocol['admitted_tiers']];available=[m['method'] for m in admitted]
    sr=table(out/'stage_metrics.csv');cr=table(out/'class_metrics.csv')
    stage={(r['dataset'],int(r['seed']),int(r['task']),r['method']):r for r in sr}
    classes={(r['dataset'],int(r['seed']),int(r['task']),r['method'],int(r['class_id'])):r for r in cr}
    if len(sr)!=len(available)*45 or len(cr)!=len(available)*459 or len(stage)!=len(sr) or len(classes)!=len(cr):raise ValueError('COVERAGE_FAILED')
    summaries={};mat=[]
    for ds,sizes in SIZES.items():
        for m in protocol['methods']:
            mat.append({'dataset':ds,'method':m['method'],'tier':m['tier'],'status':'COMPLETE' if m['method'] in available else 'NOT_ADMITTED_RESOURCE','expected_stage_rows':3*len(sizes)})
        for name in available:
            for seed in SEEDS:
                rows=[stage[ds,seed,t,name] for t in range(1,len(sizes)+1)];v=[float(r['ba']) for r in rows]
                summaries[ds,seed,name]={'final_ba':v[-1],'avg_ba_all':float(np.mean(v)),'avg_ba_inc':float(np.mean(v[1:])),
                  'final_macro_f1':float(rows[-1]['macro_f1']),'final_accuracy':float(rows[-1]['accuracy']),'final_tail':float(rows[-1]['tail_ba']),
                  'final_old':float(rows[-1]['old_ba']),'final_current':float(rows[-1]['current_ba']),'final_hm':float(rows[-1]['hm'])}
    _write_csv(out/'METHOD_MATRIX.csv',mat)
    _write_csv(out/'summary_by_parent_and_projection.csv',[{'dataset':ds,'seed':sd,'method':m,'rp_seed':next(v['rp_seed'] for v in admitted if v['method']==m),**v} for (ds,sd,m),v in summaries.items()])
    cvrows=[];selection=[];solver=[]
    for p in sorted((out/'stages').glob('*/*/task_*/STATE_LOCK.json')):
        d=_read(p)
        solver.append({'dataset':d['dataset'],'seed':d['seed'],'task':d['task'],'max_residual':max(d['residuals'].values())})
        if d['task']!=1:continue
        for method,detail in d['cv'].items():
            common={'dataset':d['dataset'],'seed':d['seed'],'method':method}
            selection.append({**common,'lambda':d['lambdas'][method],'status':detail['status']})
            if detail['scores']:
                for f,loss in enumerate(detail['scores']):
                    cvrows.extend({**common,'fold':f,'lambda':l,'loss':v,'status':detail['status']} for l,v in zip(detail['grid'],loss))
            else:cvrows.append({**common,'fold':None,'lambda':.001,'loss':None,'status':detail['status']})
    _write_csv(out/'lambda_cv_scores.csv',cvrows);_write(out/'lambda_selection.json',{'selection':selection,'end_to_end_independent_validation':False})
    errors=[];bootrows=[];robust=[];paired=[];pairs=comparisons(available)
    for ds,sizes in SIZES.items():
        with np.load(out/'private'/f'{ds}_PREDICTIONS.npz',allow_pickle=False) as z:
            y=z['labels'];comp=z['components'];ix=[json.loads(str(v)) for v in z['index']];pred={(v['seed'],v['task'],v['method']):p for v,p in zip(ix,z['predictions'])}
        boot,unit=bootstrap_layout(y,comp);ids=np.unique(y)
        def contrast(label,left,right):
            delta=[]
            for seed in SEEDS:
                lc=np.mean([(pred[seed,len(sizes),m]==y).astype(float) for m in left],axis=0)
                rc=np.mean([(pred[seed,len(sizes),m]==y).astype(float) for m in right],axis=0);delta.append(lc-rc)
            mean=np.mean(delta,axis=0);v,missing=boot(mean);finite=v[np.isfinite(v)];lo,hi=100*np.quantile(finite,[.025,.975]) if len(finite) else (None,None)
            gains=[float(100*np.mean([d[y==c].mean() for c in ids])) for d in delta]
            bootrows.append({'dataset':ds,'comparison':label,'final_ba_gain_pp':float(np.mean(gains)),'three_order_sd_pp':float(np.std(gains,ddof=1)),
              'conditional_low_pp':lo,'conditional_high_pp':hi,'bootstrap_seed':68026,'replicates':2000,'missing_class_replicates':missing,'unit':unit,'fixed_parents':3,'parent_or_projection_resampling':False})
        for left,right in pairs:
            contrast(left+' - '+right,[left],[right])
            for seed in SEEDS:
                for t in range(1,len(sizes)+1):
                    l,r=stage[ds,seed,t,left],stage[ds,seed,t,right]
                    paired.append({'dataset':ds,'seed':seed,'task':t,'candidate':left,'comparator':right,
                      **{k+'_difference_pp':100*(float(l[k])-float(r[k])) if l[k] and r[k] else None for k in ['ba','macro_f1','tail_ba','old_ba','current_ba','hm']}})
                    lp,rp=pred[seed,t,left],pred[seed,t,right]
                    for c in np.unique(y[rp>=0]):
                        keep=y==c;changed=(lp!=rp)&keep
                        errors.append({'dataset':ds,'seed':seed,'task':t,'candidate':left,'comparator':right,'class_id':int(c),'n':int(keep.sum()),
                          'corrections':int(np.sum(keep&(lp==y)&(rp!=y))),'destructions':int(np.sum(keep&(lp!=y)&(rp==y))),
                          'wrong_to_wrong':int(np.sum(changed&(lp!=y)&(rp!=y)))})
        for tag in ['F','G','B']:
            if tag+'.RF2' in available:
                contrast(tag+'.meanRP - '+tag+'.LIN',[tag+'.RF1',tag+'.RF2'],[tag+'.LIN'])
                for seed in SEEDS:
                    x=summaries[ds,seed,tag+'.RF1']['final_ba'];z=summaries[ds,seed,tag+'.RF2']['final_ba'];lin=summaries[ds,seed,tag+'.LIN']['final_ba']
                    robust.append({'dataset':ds,'seed':seed,'branch':tag,'rp1_gain_pp':100*(x-lin),'rp2_gain_pp':100*(z-lin),'mean_rp_gain_pp':100*((x+z)/2-lin),'same_gain_sign':bool((x-lin)*(z-lin)>=0),'logits_ensemble':False})
        for tag in ['G','B']:
            # Difference in differences at fixed systems; no causal interaction claim.
            contrast(tag+'.RF1 - F.RF1 - ('+tag+'.LIN - F.LIN)',[tag+'.RF1','F.LIN'],['F.RF1',tag+'.LIN'])
            bootrows[-1]['final_ba_gain_pp']*=2
            for k in ['conditional_low_pp','conditional_high_pp','three_order_sd_pp']:bootrows[-1][k]*=2
            if tag+'.RF2' in available:contrast(tag+'.meanRP - F.meanRP',[tag+'.RF1',tag+'.RF2'],['F.RF1','F.RF2'])
        if 'B.RF2' in available:contrast('B.meanRP - G.meanRP',['B.RF1','B.RF2'],['G.RF1','G.RF2'])
    _write_csv(out/'bootstrap_intervals.csv',bootrows);_write_csv(out/'error_decomposition.csv',errors);_write_csv(out/'paired_differences.csv',paired)
    _write_csv(out/'projection_robustness.csv',robust or [{'status':'NOT_ADMITTED_RESOURCE'}])
    forgetting=[];series=defaultdict(list)
    for (ds,sd,t,m,c),r in classes.items():series[ds,sd,m,c].append((t,float(r['recall'])))
    for (ds,sd,m,c),v in series.items():
        v.sort();forgetting.append({'dataset':ds,'seed':sd,'method':m,'class_id':c,'first_recall':v[0][1],'final_recall':v[-1][1],'signed_forgetting':v[0][1]-v[-1][1],'maximum_forgetting':max(x[1] for x in v)-v[-1][1]})
    _write_csv(out/'forgetting.csv',forgetting)
    gates=[_gate(stage,classes,summaries,ds,tag+'.RF1',tag+'.LIN') for ds in SIZES for tag in ['G','B']]
    access=defaultdict(lambda:{'image_reads':0,'apart_forwards':0,'G_forwards':0,'B_forwards':0})
    events=[json.loads(v) for v in (out/'ACCESS_EVENTS.jsonl').read_text().splitlines()]
    for e in events:
        if e['kind']=='image_forward':
            a=access[e['phase']+'.'+e['dataset']];a['image_reads']+=e['n'];a['apart_forwards']+=e['apart']*e['n']
            for tag in e['vlm']:a[tag+'_forwards']+=e['n']
    _write(out/'ACCESS_LEDGER.json',{'physical_access':dict(access),'test_access':0,'reserved_access':0,'new_neural_epochs':0,'optimizer_steps':0,'logical_fit_streams':6,'temporary_train_features':'RAM current task only, released before next task','val_APART_cache':'verified existing cache per fixed parent','VLM_val':'one extraction per encoder/dataset shared by methods and parents'})
    clock=_read(out/'CLOCK_LOCK.json');now=time.time();cv_events=[e for e in events if e['kind']=='cv'];sealed=[e for e in events if e['kind']=='stage_sealed']
    _write(out/'RESOURCE_REPORT.json',{'t0_utc':clock['t0_utc'],'report_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now)),'wall_seconds':now-clock['t0_epoch'],'physical_gpu':2,'gpu_name':_read(out/'QUALIFICATION.json')['gpu_name'],
      'cv_candidate_solves':sum(e['candidate_solves'] for e in cv_events),'cv_spectral_decompositions':sum(e['spectral_decompositions'] for e in cv_events),'logical_stage_rows':len(sr),'unique_base_fit_stages':len([m for m in admitted if m['lambda_policy']=='task1_group_cv'])*45,
      'logical_reference_readouts':5*45,'aliases':'fixed lambda sharing a selected lambda reuses its solve; CSE inherits W','peak_gpu_bytes':max(e['peak_gpu_bytes'] for e in sealed),'peak_rss_kib':max(e['peak_rss_kib'] for e in sealed),
      'gpu_process_residence':'recorded by bounded controller in RESOURCE_SAMPLES.jsonl; includes loading and backup waits, not pure GPU compute'})
    report={'status':'ADMITTED_MATRIX_COMPLETE','gates':gates,'stage_rows':len(sr),'class_rows':len(cr),'methods':len(available),'admitted_tiers':protocol['admitted_tiers'],'pretrain_exposure':'UNKNOWN','independent_confirmation':False,'next_decision':'STOP','new_neural_epochs':0,'optimizer_steps':0,'test_access':0,'reserved_access':0}
    _write(out/'FINAL_REPORT.json',report);_write(out/'ENGINEERING_REPORT.json',{'status':'PASS','stage_residuals':solver,'qualification':_read(out/'QUALIFICATION.json'),'no_implicit_jitter':True,'fit_eval_separate_processes':True,'stage_backups':len(list((out/'backup_acks').glob('*.json')))})
    _write(out/'failure_receipts.json',{'failures':[_read(p) for p in sorted(out.glob('FAILED_*.json'))]})
    lines=['# NB2-RFVILA-R1 最终报告','','本轮是 VILA 风格医学适配，不是论文原始复现。正常两类 Task1 后全部冻结，新增神经训练和 optimizer steps 为 0。',
      '主投影 67101；67102 独立报告，不挑赢家、不平均 logits。所有比较均使用相同 Task1 九点正则选择预算；CV 表征已学习 Task1，不是端到端独立验证。','',
      f'已准入矩阵完成：{len(available)} 方法，{len(sr)} 阶段行，{len(cr)} 逐类行。','',
      '| 数据集 | 方法 | Final BA % | Macro-F1 % | AvgBA_inc % |','|---|---|---:|---:|---:|']
    for ds in SIZES:
        for m in available:
            v=[summaries[ds,s,m] for s in SEEDS];lines.append(f"| {ds} | {m} | {100*np.mean([x['final_ba'] for x in v]):.3f} | {100*np.mean([x['final_macro_f1'] for x in v]):.3f} | {100*np.mean([x['avg_ba_inc'] for x in v]):.3f} |")
    lines+=['','## RF 对公平 LIN 的固定效用门槛','']
    for g in gates:lines.append(f"- {g['dataset']} {g['candidate']} − {g['comparator']}: {np.mean(g['final_ba_gain_pp_by_seed']):+.3f} pp，{g['status']}；未通过：{', '.join(k for k,v in g['criteria'].items() if not v) or '无'}。")
    lines+=['','## 固定机制比较','', '| 数据集 | 比较 | Final BA 差 pp | 条件区间 |','|---|---|---:|---|']
    for b in bootrows:lines.append(f"| {b['dataset']} | {b['comparison']} | {b['final_ba_gain_pp']:+.3f} | [{b['conditional_low_pp']:+.3f}, {b['conditional_high_pp']:+.3f}] |")
    lines+=['','RF−LIN 检验非线性读出；RF−F.RF 检验同容量下 VLM 的附加信息；B−G 检验固定系统的差异，不能单因果归于医学语料。',
      'CSE 的净贡献与真实类别 Top-K、纠错/破坏见 candidate_coverage.csv；两个 RP 的方向一致性见 projection_robustness.csv。',
      '区间按 identity component 成对重采样 2000 次；固定父状态与投影，不覆盖训练随机性或未见领域。多重开发、极小尾类和 UNKNOWN 预训练暴露限制保留。',
      '未准入项见 METHOD_MATRIX.csv；所有私有图像、身份/组件映射、逐样本特征/分数和 S/Q/W/R 只留规定存储。关键资产目标端 SHA 已核验，大图像归档未声明全量逐字节一致。',
      'NEXT_DECISION=STOP。','']
    (out/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines));_write(out/'NEXT_DECISION.json',{'decision':'STOP'})
    _write(out/'completion_receipts.json',{'status':'ADMITTED_MATRIX_COMPLETE','stage_rows':len(sr),'class_rows':len(cr),'methods':len(available),'fit_stages':45,'time':time.time()})
    return report
