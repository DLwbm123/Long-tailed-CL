"""Aggregate the complete locked matrix; never select a method from validation."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from tools.run_nb2_vlm_r1 import _read,_write,_sha,SIZES
from tools.finalize_nb2_vlm_r1 import _gate
from route_a.run_ct13_real import _write_csv
from tools.medvlm_math import CORE,EXTRA

def contrasts(ds,models):
    pairs=[('B.J','G.J'),('B.J','F'),('B.V','G.V'),('B.Z','G.Z')]
    for tag in models:
        if tag=='D' and ds!='ISIC':continue
        pairs += [(tag+'.L',tag+'.J'),(tag+'.R',tag+'.J'),(tag+'.J','F'),(tag+'.J1','F2')]
        pairs += [(tag+'.'+m,tag+'.R') for m in EXTRA[:3]]
        pairs += [(tag+'.L',tag+'.L-permute'),(tag+'.template-0',tag+'.template-1')]
    if ds=='ISIC' and 'D' in models:pairs += [('D.J','G.J'),('D.J','B.J')]
    return list(dict.fromkeys(pairs))

def read_csv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def finalize(out,cfg):
    stage_rows=read_csv(out/'stage_metrics.csv');class_rows=read_csv(out/'class_metrics.csv')
    stage={(r['dataset'],int(r['seed']),int(r['task']),r['method']):r for r in stage_rows}
    classes={(r['dataset'],int(r['seed']),int(r['task']),r['method'],int(r['class_id'])):r for r in class_rows}
    summaries={};expected_stage=0;expected_class=0;matrix=[]
    for ds,sizes in SIZES.items():
        methods=['P','F','F2']+[tag+'.'+m for tag in cfg['models'] if tag!='D' or ds=='ISIC' for m in CORE+EXTRA]
        expected_stage+=len(sizes)*3*len(methods);expected_class+=sum(np.cumsum(sizes))*3*len(methods)
        for method in methods:
            matrix.append({'dataset':ds,'method':method,'status':'COMPLETE','seeds':3,'tasks_per_seed':len(sizes)})
            for seed in (1993,1994,1995):
                rows=[stage[ds,seed,t,method] for t in range(1,len(sizes)+1)]
                vals=[float(r['ba']) for r in rows]
                summaries[ds,seed,method]={'avg_ba_all':float(np.mean(vals)),'avg_ba_inc':float(np.mean(vals[1:])),
                    'final_ba':vals[-1],'final_macro_f1':float(rows[-1]['macro_f1']),'final_tail':float(rows[-1]['tail_ba'])}
    if len(stage)!=expected_stage or len(stage_rows)!=expected_stage or len(classes)!=expected_class or len(class_rows)!=expected_class:raise ValueError('FINAL_COVERAGE')
    if 'D' not in cfg['models']:
        matrix.extend({'dataset':'ISIC','method':'D.'+m,'status':'BLOCKED_ACCESS_LICENSE','seeds':0,'tasks_per_seed':0} for m in CORE+EXTRA)
    _write_csv(out/'METHOD_MATRIX.csv',matrix)
    _write_csv(out/'summary_by_seed.csv',[{'dataset':d,'seed':s,'method':m,**v} for (d,s,m),v in summaries.items()])
    forgetting=[]
    series=defaultdict(list)
    for (ds,sd,t,m,c),r in classes.items():series[ds,sd,m,c].append((t,float(r['recall'])))
    for (ds,sd,m,c),vals in series.items():
        vals.sort();a,b=vals[0][1],vals[-1][1]
        forgetting.append({'dataset':ds,'seed':sd,'method':m,'class_id':c,'first_task':vals[0][0],
            'first_recall':a,'final_recall':b,'signed_forgetting':a-b,'maximum_forgetting':max(v for _,v in vals)-b})
    _write_csv(out/'forgetting.csv',forgetting)
    paired=[];errors=[];bootstrap=[];parity=[]
    for ds,sizes in SIZES.items():
        path=out/'private'/f'{ds}_PREDICTIONS.npz'
        if _sha(path)!=_read(out/'PREDICTIONS_LOCK.json')['files'][ds]:raise ValueError('PREDICTIONS_CHANGED')
        with np.load(path,allow_pickle=False) as z:
            labels=z['labels'];components=z['components'];idx=[json.loads(str(x)) for x in z['index']]
            pred={(int(k['seed']),int(k['task']),k['method']):p for k,p in zip(idx,z['predictions'])}
        with np.load(Path(cfg['generic_run'])/'private'/f'{ds}_VAL_PREDICTIONS.npz',allow_pickle=False) as z:
            oldidx=[json.loads(str(x)) for x in z['index']]
            old={(int(k['seed']),int(k['task']),k['method']):p for k,p in zip(oldidx,z['predictions'])}
            if not np.array_equal(labels,z['labels']):raise ValueError('GENERIC_VAL_LAYOUT')
        for seed in (1993,1994,1995):
            for t in range(1,len(sizes)+1):
                for new,oldname in [('G.J1','A2'),('G.V','A1'),('F','A0'),('P','P')]:
                    parity.append({'dataset':ds,'seed':seed,'task':t,'method':new,
                        'prediction_differences':int(np.count_nonzero(pred[seed,t,new]!=old[seed,t,oldname]))})
        rng=np.random.default_rng(57026)
        groups={int(c):[np.flatnonzero((labels==c)&(components==x)) for x in np.unique(components[labels==c])] for c in np.unique(labels)}
        draws={c:rng.integers(0,len(g),size=(2000,len(g))) for c,g in groups.items()}
        for left,right in contrasts(ds,cfg['models']):
            final_diff=[]
            for seed in (1993,1994,1995):
                for t in range(1,len(sizes)+1):
                    l,r=stage[ds,seed,t,left],stage[ds,seed,t,right]
                    paired.append({'dataset':ds,'seed':seed,'task':t,'candidate':left,'comparator':right,
                        **{k+'_difference_pp':100*(float(l[k])-float(r[k])) for k in ('ba','macro_f1','tail_ba','old_ba','current_ba','hm')}})
                    lp,rp=pred[seed,t,left],pred[seed,t,right]
                    for cid in np.unique(labels[rp>=0]):
                        keep=labels==cid;changed=(lp!=rp)&keep
                        errors.append({'dataset':ds,'seed':seed,'task':t,'candidate':left,'comparator':right,'class_id':int(cid),
                            'n':int(keep.sum()),'changed':int(changed.sum()),
                            'corrections':int(np.sum(keep&(lp==labels)&(rp!=labels))),
                            'destructions':int(np.sum(keep&(lp!=labels)&(rp==labels))),
                            'wrong_to_different_wrong':int(np.sum(changed&(lp!=labels)&(rp!=labels)))})
                final_diff.append((pred[seed,len(sizes),left]==labels).astype(float)-(pred[seed,len(sizes),right]==labels).astype(float))
            difference=np.mean(final_diff,axis=0);boot=np.zeros(2000)
            for cid,items in groups.items():
                nums=np.asarray([difference[x].sum() for x in items]);dens=np.asarray([len(x) for x in items]);ix=draws[cid]
                boot+=nums[ix].sum(1)/dens[ix].sum(1)/len(groups)
            gains=[100*np.mean([d[labels==c].mean() for c in groups]) for d in final_diff]
            lo,hi=100*np.quantile(boot,[.025,.975])
            bootstrap.append({'dataset':ds,'candidate':left,'comparator':right,'final_ba_gain_pp':float(np.mean(gains)),
                'three_order_sd_pp':float(np.std(gains,ddof=1)),'conditional_low_pp':float(lo),'conditional_high_pp':float(hi),
                'crosses_zero':bool(lo<=0<=hi),'bootstrap_replicates':2000,'bootstrap_seed':57026,
                'unit':'class_stratified_identity_component','fixed_model_seeds':3,'minimum_class_val_n':min(int((labels==c).sum()) for c in groups)})
    _write_csv(out/'paired_differences.csv',paired);_write_csv(out/'error_decomposition.csv',errors)
    _write_csv(out/'bootstrap_intervals.csv',bootstrap)
    _write(out/'GENERIC_REPRODUCTION.json',{'rows':parity,'total_prediction_differences':sum(x['prediction_differences'] for x in parity)})
    gates=[_gate(stage,classes,summaries,ds,'B.J',ref) for ds in SIZES for ref in ('G.J','F')]
    primary=[g for g in gates if g['comparator']=='G.J']
    access=defaultdict(lambda:{'image_reads':0,'apart_batches':0,'vlm_batches':0})
    for p in out.glob('access_*.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line);key=r['phase']+'.'+r['model']+'.'+r['dataset']
            for field in access[key]:access[key][field]+=r[field]
    _write(out/'access_ledger.json',{'physical_access':dict(access),'test_access':0,'reserved_access':0,
        'new_neural_epochs':0,'optimizer_steps':0,'generic_training_statistics_reused':True,
        'medical_val_vlm_cache':'one extraction per dataset, shared across fixed parents after fit lock',
        'medical_val_apart_cache':'G evaluation features reused per parent after fit lock'})
    report={'status':'PRIMARY_MATRIX_COMPLETE','primary_utility_gate':'PASS' if all(g['status']=='PASS' for g in primary) else 'FAIL',
        'gates':gates,'stage_rows':len(stage),'class_rows':len(classes),'dermlip_status':'COMPLETE' if 'D' in cfg['models'] else 'BLOCKED_ACCESS_LICENSE',
        'pretrain_exposure':'UNKNOWN','independent_confirmation':False,'new_neural_epochs':0,'optimizer_steps':0,
        'test_access':0,'reserved_access':0,'next_decision':'STOP'}
    _write(out/'FINAL_REPORT.json',report);_write(out/'NEXT_DECISION.json',{'decision':'STOP'})
    lines=['# NB2-MedVLM-R1 最终报告','',f"G/B 主矩阵完成；BiomedCLIP 对 Generic 主效用门槛：{report['primary_utility_gate']}。",'',
        '本轮是既有开发数据上的固定医学 VLM 对照；不是独立确认或临床验证。新增神经训练与 optimizer steps 均为 0，test/reserved 访问为 0。',
        '本轮 G.J 使用 λ=5e-4，原 R1 A2 对应 G.J1（λ=1e-3）；旧 R1 保持原样。',
        'BiomedCLIP 使用官方视觉编码器、PubMedBERT 文本编码器、context=256 tokenizer 与预处理。预训练直接样本暴露 UNKNOWN。','',
        '## 各方法三顺序均值','', '| 数据集 | 方法 | Final BA (%) | Macro-F1 (%) | AvgBA_inc (%) |','|---|---|---:|---:|---:|']
    for ds in SIZES:
        methods=list(dict.fromkeys(m for d,s,m in summaries if d==ds))
        for method in methods:
            v=[summaries[ds,s,method] for s in (1993,1994,1995)]
            lines.append(f"| {ds} | {method} | {100*np.mean([x['final_ba'] for x in v]):.3f} | {100*np.mean([x['final_macro_f1'] for x in v]):.3f} | {100*np.mean([x['avg_ba_inc'] for x in v]):.3f} |")
    lines+=['','## 固定门槛','']
    for g in gates:
        fail=[k for k,v in g['criteria'].items() if not v]
        lines.append(f"- {g['dataset']} B.J − {g['comparator']}：{np.mean(g['final_ba_gain_pp_by_seed']):+.3f} pp，{g['status']}；未通过项：{', '.join(fail) or '无'}。")
    lines+=['','## 机制比较与区间','', '| 数据集 | 比较 | Final BA 差值 (pp) | 固定模型条件区间 |','|---|---|---:|---|']
    for b in bootstrap:
        lines.append(f"| {b['dataset']} | {b['candidate']} − {b['comparator']} | {b['final_ba_gain_pp']:+.3f} | [{b['conditional_low_pp']:+.3f}, {b['conditional_high_pp']:+.3f}] |")
    lines+=['','条件区间仅重采样已有验证身份组件，不覆盖训练随机性或未见领域；跨零表示证据较弱。HK 极小尾类的实际分母见 class_metrics.csv。',
        '完整阶段、类别、遗忘与纠错/破坏分解均已公开；私有 logits、图像 ID、父状态、统计 bank/W 留在规定服务器存储。',
        f"DermLIP：{report['dermlip_status']}。当前官方访问需帐号授权，模型卡许可证字段与正文不一致，未满足探索项执行条件。" if 'D' not in cfg['models'] else 'DermLIP 仅是 ISIC 探索项，不参与主效用门槛。',
        '独立备份状态见 backup_report.json；NEXT_DECISION=STOP。','']
    (out/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines))
    return report
