"""HK1-M1 fixed class-balanced fits and validation report. CPU only, no images."""
import csv
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from prepare_hyperkvasir_hk1_m1 import ORDERS, SEEN, json_hash
from isic_g1_controlled_mixup import Increment, moments, solve, close, sha, digest
from isic_a1_attribution import view, duplicate_embedding
from report_locked_holdout_r1 import read, write, csvwrite, predict_columns, bootstrap_weights, boot_unit

METHODS=('A-CB','S-J-CB','S-M-CB')
COMPARISONS=(('S-J-CB','A-CB'),('S-M-CB','A-CB'),('S-J-CB','S-M-CB'))


def mean(x):return float(np.mean(x)) if len(x) else None


def analyze(p,method,seed,stage,groups):
    y=p['y'];raw=p['raw'];order=p['order'];n=SEEN[stage];known=0 if stage==0 else SEEN[stage-1]
    assert raw.shape==(len(y),n) and np.isfinite(raw).all() and np.array_equal(order[y],p['original'])
    assert len(y)==len(set(p['ids'].tolist()))
    pred=predict_columns(raw,order);correct=pred==y
    prefix=dict(method=method,order_seed=seed,session=stage,split='val',seen_classes=n,old_classes=known,current_classes=n-known)
    pc=[]
    for c in range(n):
        mask=y==c;assert mask.any()
        pc.append(dict(**prefix,head_id=c,original_label=int(order[c]),n_images=int(mask.sum()),
            n_correct=int(correct[mask].sum()),n_components=len(np.unique(p['component'][mask])),
            recall=100*float(correct[mask].mean()),in_parent_base=c<13))
    recalls=np.array([r['recall'] for r in pc]);old=list(range(known));current=list(range(known,n))
    m=dict(**prefix,balanced_accuracy=float(recalls.mean()),accuracy=100*float(correct.mean()),
        macro_f1=100*float(f1_score(y,pred,labels=range(n),average='macro',zero_division=0)),
        old_macro_recall=mean(recalls[old]),current_macro_recall=mean(recalls[current]),
        worst_class_recall=float(recalls.min()),zero_recall_classes=[int(order[c]) for c in range(n) if recalls[c]==0],
        n_images=len(y),n_components=len(set(p['component'].tolist())),
        exact_argmax_ties=int(np.sum(np.sum(raw==raw.max(1,keepdims=True),axis=1)>1)))
    a=m['old_macro_recall'];b=m['current_macro_recall'];m['HM_old_current_macro']=None if a is None else (2*a*b/(a+b) if a+b else 0.)
    scopes=dict(groups,base=list(order[:13]),post_base=list(order[13:]),
        old_tail=[c for c in order[:known] if c in groups['tail_rank8']],
        current_tail=[c for c in order[known:n] if c in groups['tail_rank8']],
        tail_base=[c for c in order[:13] if c in groups['tail_rank8']],
        tail_post_base=[c for c in order[13:] if c in groups['tail_rank8']])
    for name,classes in scopes.items():
        cs=[c for c in range(n) if order[c] in classes];m[name+'_macro_recall']=mean(recalls[cs]);m[name+'_n_classes']=len(cs)
    errors=[]
    for scope,cs in (('current',current),('old',old)):
        mask=np.isin(y,cs);same=np.isin(pred,cs)
        events=dict(correct=correct,to_other=~same,wrong_within=same&~correct)
        er=dict(**prefix,scope=scope,n_images=int(mask.sum()),n_classes=len(cs))
        for name,event in events.items():
            er[name+'_sample_weighted']=100*float(event[mask].mean()) if mask.any() else None
            er[name+'_class_macro']=100*mean([event[y==c].mean() for c in cs]) if cs else None
        if cs:
            for suffix in ('sample_weighted','class_macro'):assert abs(sum(er[k+'_'+suffix] for k in events)-100)<1e-10
            rp=np.array(cs)[predict_columns(raw[mask][:,cs],order[cs])]
            m['restricted_'+scope+'_BA_diagnostic']=100*mean([(rp[y[mask]==c]==c).mean() for c in cs])
        else:m['restricted_'+scope+'_BA_diagnostic']=None
        errors.append(er)
    return m,pc,errors


def engineering(private):
    private=Path(private);rng=np.random.default_rng(71);order=[2,0,1]
    xs=[rng.normal(size=(n,7)) for n in (8,3,5)];inc=Increment(order,7,'B0');checks=[]
    for j,(c,x) in enumerate(zip(order,xs)):
        inc.arrive(c,moments(x));G,R=inc.system();W,res=solve(G,R)
        X=np.concatenate(xs[:j+1]);w=np.concatenate([np.full(len(a),1/len(a)/(j+1)) for a in xs[:j+1]])
        y=np.concatenate([np.full(len(a),k) for k,a in enumerate(xs[:j+1])]);Y=np.eye(j+1)[y]
        close(G,X.T@(w[:,None]*X));close(R,X.T@(w[:,None]*Y));close(W,solve(X.T@(w[:,None]*X),X.T@(w[:,None]*Y))[0])
        checks.append(dict(stage=j,residual=res,**duplicate_embedding(G,R,W,X)))
        if j==0:
            p=private/'toy_resume.npz';inc.save(p);inc=Increment.restore(p);p.unlink()
    reverse=Increment(order[::-1],7,'B0')
    for c,x in zip(order[::-1],xs[::-1]):reverse.arrive(c,moments(x))
    close(solve(*reverse.system())[0][:,::-1],W)
    try:Increment(order,7,'B0').arrive(0,moments(xs[0]))
    except AssertionError:pass
    else:raise AssertionError('FUTURE_GUARD_FAILED')
    assert np.array_equal(predict_columns(np.ones((2,3)),np.array([9,2,5])),[1,1])
    for stage,n in enumerate(SEEN):
        labels=np.array(ORDERS[1993][:n]);p=dict(y=np.arange(n),original=labels,order=np.array(ORDERS[1993]),
            ids=labels.astype(str),component=labels.astype(str),raw=np.eye(n))
        m,pc,er=analyze(p,'A-CB',1993,stage,dict(tail_rank8=list(range(8)),middle_rank7=list(range(8,15)),head_rank8=list(range(15,23))))
        assert m['balanced_accuracy']==100 and len(pc)==n and m['post_base_macro_recall']==(None if stage==0 else 100.)
    return dict(status='PASS',class_balanced_stage_batch_stream_equivalence=True,checks=checks,
        same_parent_order_invariance=True,duplicate_embedding=True,restore_then_next_arrival=True,
        future_class_guard=True,tie_minimum_canonical_ID=True,dynamic_23_class_six_stage_metrics=True,empty_groups_null=True)


def run(cfg):
    root=Path(cfg['root']);out=root/'output';pub=out/'public';private=out/'private';started=time.monotonic()
    protocol=read(pub/'PROTOCOL_HK1.json');assert read(pub/'GPU_PHASE_COMPLETE.json')['status']=='PASS'
    cpu_prior=read(pub/'ENGINEERING_HK1.json')['analytic']['wall_seconds']+60. # conservative reserve for the independent preflight check
    lock=read(pub/'FEATURE_CACHE_LOCK_HK1.json');parents=read(pub/'ALL_S0_LOCK_HK1.json');assert len(parents['parents'])==3
    resources=read(pub/'GPU_RESOURCE_LEDGER.json');peak=resources['artifact_peak_bytes'];minfree=resources['min_free_bytes']
    def budget():
        nonlocal peak,minfree
        size=cfg['external_artifact_bytes']+sum(p.stat().st_size for p in out.rglob('*') if p.is_file());peak=max(peak,size);minfree=min(minfree,shutil.disk_usage(root).free)
        assert cpu_prior+time.monotonic()-started<7200,'BLOCKED_CPU_BUDGET'
        assert size<=768*1024**2 and minfree>=1024**3,'BLOCKED_STORAGE'
    rows={s:list(csv.DictReader((root/'inputs'/(s+'.csv')).open())) for s in ('train','val')}
    ys={s:np.array([int(r['original_label']) for r in rs]) for s,rs in rows.items()}
    for s in rows:assert sha(root/'inputs'/(s+'.csv'))==protocol['manifest_sha256'][s]
    numeric=[];readouts=[];arrival=[];final_A={};final_states=[]
    for method in METHODS:
        for seed,order in ORDERS.items():
            source=private/(f'A_train.npy' if method=='A-CB' else f'raw_{seed}_train.npy')
            owner=1993 if method=='A-CB' else seed
            item=next(x for x in lock if x['seed']==owner and x['split']=='train')
            assert sha(source)==item['A_sha256' if method=='A-CB' else 'raw_sha256']
            raw=np.load(source,mmap_mode='r');z=raw.astype(np.float64) if method=='A-CB' else view(raw,'J' if method=='S-J-CB' else 'M')
            assert len(z)==len(rows['train']) and np.isfinite(z).all()
            inc=Increment(order,z.shape[1],'B0');stage=0
            for index,c in enumerate(order):
                ix=ys['train']==c;assert ix.sum()==protocol['fit_counts'][str(c)]
                inc.arrive(c,moments(z[ix]));arrival.append(dict(method=method,seed=seed,original_label=c,n_fit=int(ix.sum()),arrival_index=index))
                if index+1 not in SEEN:continue
                G,R=inc.system();close(G,G.T);W,res=solve(G,R)
                # Independent weighted batch target, with only the same seen fit classes.
                mask=np.isin(ys['train'],order[:index+1]);X=z[mask];lbl=ys['train'][mask]
                y=np.array([order.index(int(c0)) for c0 in lbl]);weights=np.array([1/protocol['fit_counts'][str(c0)]/(index+1) for c0 in lbl])
                Gb=X.T@(weights[:,None]*X);Rb=X.T@(weights[:,None]*np.eye(index+1)[y]);close(G,Gb);close(R,Rb)
                Wb,_=solve(Gb,Rb);werr=close(W,Wb)
                assert np.array_equal(predict_columns(X@W,np.array(order)),predict_columns(X@Wb,np.array(order)))
                checks=duplicate_embedding(G,R,W,X[:48]) if method=='A-CB' else {}
                if method=='S-J-CB':assert np.linalg.norm(G[:768,768:])>0
                path=private/f'W_{method}_{seed}_s{stage}.npy';np.save(path,W)
                readouts.append(dict(method=method,seed=seed,stage=stage,seen=index+1,path=path.name,W_sha256=sha(path),
                    feature_sha256=item['A_sha256' if method=='A-CB' else 'raw_sha256'],columns=order[:index+1],fit_only=True))
                numeric.append(dict(method=method,seed=seed,stage=stage,residual=res,stream_batch_W_max_abs=werr,
                    G_sha256=digest(G),R_sha256=digest(R),full_joint_cross_block_norm=float(np.linalg.norm(G[:768,768:])) if method=='S-J-CB' else None,**checks))
                stage+=1;budget()
            assert stage==6
            # Same-parent alternate accumulation order is an isolated algebra check.
            check=Increment(order[::-1],z.shape[1],'B0')
            for c in order[::-1]:check.arrive(c,moments(z[ys['train']==c]))
            Wrev,_=solve(*check.system());close(W,Wrev[:,::-1])
            state=private/f'statistics_{method}_{seed}.npz';inc.save(state)
            restored=Increment.restore(state);close(solve(*restored.system())[0],W)
            final_states.append(dict(method=method,seed=seed,path=state.name,sha256=sha(state),same_parent_order_invariance=True,restore=True))
            if method=='A-CB':final_A[seed]=W[:,np.argsort(order)]
            del raw,z,inc,check,X,G,R,W,Wrev;budget()
    assert len(readouts)==54
    close(final_A[1993],final_A[1994]);close(final_A[1993],final_A[1995])
    write(pub/'READOUT_LOCK_HK1.json',dict(status='PASS',readouts=readouts,states=final_states,lambda_value=.001,
        bias=False,statistics_dtype='float64',formal_arrivals=arrival,A_final_order_invariant=True,all_S0_locked=True))
    csvwrite(pub/'ANALYTIC_NUMERICS.csv',numeric)
    write(pub/'VAL_SCORE_RELEASE_HK1.json',dict(status='RELEASED',readout_lock_sha256=sha(pub/'READOUT_LOCK_HK1.json'),
        fixed_readouts=54,performance_feedback_before_all_W_lock=False,test_release=False))
    # No validation performance is computed until the complete W lock exists.
    data={};metrics=[];classes=[];errors=[]
    for method in METHODS:
        for seed,order in ORDERS.items():
            path=private/('A_val.npy' if method=='A-CB' else f'raw_{seed}_val.npy')
            item=next(x for x in lock if x['seed']==(1993 if method=='A-CB' else seed) and x['split']=='val')
            assert sha(path)==item['A_sha256' if method=='A-CB' else 'raw_sha256']
            raw=np.load(path,mmap_mode='r');z=raw.astype(float) if method=='A-CB' else view(raw,'J' if method=='S-J-CB' else 'M')
            for stage,n in enumerate(SEEN):
                W=np.load(private/f'W_{method}_{seed}_s{stage}.npy');ix=np.flatnonzero(np.isin(ys['val'],order[:n]));labels=ys['val'][ix]
                p=dict(raw=z[ix]@W,y=np.array([order.index(int(c)) for c in labels]),original=labels,order=np.array(order),
                    ids=np.array([rows['val'][i]['sample_id'] for i in ix]),component=np.array([rows['val'][i]['identity_component'] for i in ix]))
                m,pc,er=analyze(p,method,seed,stage,protocol['frequency_groups'])
                for row in pc:
                    source=next(r for r in rows['val'] if int(r['original_label'])==row['original_label'])
                    row['official_class_name']=source['official_class_name'];row['n_fit']=protocol['fit_counts'][str(row['original_label'])]
                    row['tail_rank8']=row['original_label'] in protocol['frequency_groups']['tail_rank8']
                np.savez_compressed(private/f'pred_{method}_{seed}_s{stage}.npz',**p)
                data[method,seed,stage]=p;metrics.append(m);classes+=pc;errors+=er;budget()
    assert len(metrics)==54 and len(classes)==972
    csvwrite(pub/'val_metrics.csv',metrics);csvwrite(pub/'val_per_class_metrics.csv',classes);csvwrite(pub/'error_decomposition.csv',errors)
    for seed in (1994,1995):
        a=data['A-CB',1993,5];b=data['A-CB',seed,5]
        assert np.array_equal(a['order'][predict_columns(a['raw'],a['order'])],b['order'][predict_columns(b['raw'],b['order'])])
    by={(r['method'],r['order_seed'],r['session']):r for r in metrics}
    pclass={(r['method'],r['order_seed'],r['session'],r['original_label']):r for r in classes}
    summary=[];forgetting=[];paired=[];keymetrics=['balanced_accuracy','tail_rank8_macro_recall','head_rank8_macro_recall','middle_rank7_macro_recall',
        'old_macro_recall','current_macro_recall','HM_old_current_macro','base_macro_recall','post_base_macro_recall',
        'old_tail_macro_recall','current_tail_macro_recall','tail_base_macro_recall','tail_post_base_macro_recall']
    for method in METHODS:
        for seed,order in ORDERS.items():
            final=by[method,seed,5]
            summary.append(dict(method=method,seed=seed,Final_BA=final['balanced_accuracy'],
                AvgBA_all=mean([by[method,seed,t]['balanced_accuracy'] for t in range(6)]),
                AvgBA_inc=mean([by[method,seed,t]['balanced_accuracy'] for t in range(1,6)]),
                **{k:final[k] for k in keymetrics if k!='balanced_accuracy'}))
            for c in order:
                first=next(t for t,n in enumerate(SEEN) if c in order[:n]);a=pclass[method,seed,first,c]['recall'];b=pclass[method,seed,5,c]['recall']
                forgetting.append(dict(method=method,seed=seed,original_label=c,first_stage=first,first_recall=a,final_recall=b,
                    recall_change_final_minus_first=b-a if first<5 else None,forgetting_first_minus_final=a-b if first<5 else None,
                    no_later_stage=first==5))
    for lhs,rhs in COMPARISONS:
        for seed in ORDERS:
            for stage in range(6):
                a=by[lhs,seed,stage];b=by[rhs,seed,stage]
                for key in keymetrics:
                    paired.append(dict(comparison=f'{lhs} minus {rhs}',seed=seed,stage=stage,metric=key,
                        delta_pp=a[key]-b[key] if a[key] is not None and b[key] is not None else None))
    csvwrite(pub/'summary_by_seed.csv',summary);csvwrite(pub/'forgetting.csv',forgetting);csvwrite(pub/'paired_deltas.csv',paired)
    # Reuse the existing paired component bootstrap, explicitly parameterized for 23 labels.
    canonical=data['A-CB',1993,5];weights=bootstrap_weights(canonical,2000,44001,labels=range(23))
    idmap={s:i for i,s in enumerate(canonical['ids'])};boot={}
    for key,p in data.items():
        ix=np.array([idmap[s] for s in p['ids']]);assert np.array_equal(p['component'],canonical['component'][ix])
        rec=boot_unit(p,weights[:,ix],True);order=p['order'];stage=key[2];known=0 if stage==0 else SEEN[stage-1]
        sets=dict(balanced_accuracy=order[:SEEN[stage]],old_macro_recall=order[:known],current_macro_recall=order[known:SEEN[stage]],
            base_macro_recall=order[:13],post_base_macro_recall=order[13:],
            **{name+'_macro_recall':cs for name,cs in protocol['frequency_groups'].items()})
        d={name:rec[:,[i for i,c in enumerate(order[:SEEN[stage]]) if c in cs]].mean(1) if any(c in cs for c in order[:SEEN[stage]]) else None for name,cs in sets.items()}
        if known:
            a=d['old_macro_recall'];b=d['current_macro_recall'];d['HM_old_current_macro']=np.divide(2*a*b,a+b,out=np.zeros_like(a),where=a+b>0)
        else:d['HM_old_current_macro']=None
        for i,c in enumerate(order[:SEEN[stage]]):d[f'label_{c}_recall']=rec[:,i]
        boot[key]=d;budget()
    intervals=[]
    for lhs,rhs in COMPARISONS:
        for stage in range(6):
            for metric in boot[lhs,1993,stage]:
                values=[boot[lhs,s,stage].get(metric) for s in ORDERS]+[boot[rhs,s,stage].get(metric) for s in ORDERS]
                if any(v is None for v in values):continue
                delta=np.mean([boot[lhs,s,stage][metric]-boot[rhs,s,stage][metric] for s in ORDERS],axis=0)
                lo,hi=np.quantile(delta,[.025,.975]);intervals.append(dict(comparison=f'{lhs} minus {rhs}',stage=stage,metric=metric,
                    lower_pp=float(lo),upper_pp=float(hi),resamples=2000,seed=44001,valid_draws=2000,parents_resampled=False))
        for name,stages in (('AvgBA_all',range(6)),('AvgBA_inc',range(1,6))):
            delta=np.mean([np.mean([boot[lhs,s,t]['balanced_accuracy']-boot[rhs,s,t]['balanced_accuracy'] for t in stages],axis=0) for s in ORDERS],axis=0)
            lo,hi=np.quantile(delta,[.025,.975]);intervals.append(dict(comparison=f'{lhs} minus {rhs}',stage='average',metric=name,
                lower_pp=float(lo),upper_pp=float(hi),resamples=2000,seed=44001,valid_draws=2000,parents_resampled=False))
    csvwrite(pub/'bootstrap_intervals.csv',intervals)
    groups=protocol['frequency_groups'];table=[]
    for method in METHODS:
        rs=[r for r in summary if r['method']==method]
        table.append(dict(method=method,Final_BA=mean([r['Final_BA'] for r in rs]),sample_std_BA=float(np.std([r['Final_BA'] for r in rs],ddof=1)),
            worst_seed_BA=min(r['Final_BA'] for r in rs),AvgBA_all=mean([r['AvgBA_all'] for r in rs]),
            AvgBA_inc=mean([r['AvgBA_inc'] for r in rs]),tail_rank8=mean([r['tail_rank8_macro_recall'] for r in rs]),
            **{k:mean([r[k] for r in rs if r[k] is not None]) for k in keymetrics if k!='balanced_accuracy'}))
    csvwrite(pub/'summary_mean.csv',table)
    main='S-J-CB minus A-CB';ba_ci=next(r for r in intervals if r['comparison']==main and r['stage']==5 and r['metric']=='balanced_accuracy')
    tail_ci=next(r for r in intervals if r['comparison']==main and r['stage']==5 and r['metric']=='tail_rank8_macro_recall')
    delta=[by['S-J-CB',s,5]['balanced_accuracy']-by['A-CB',s,5]['balanced_accuracy'] for s in ORDERS]
    tail=[by['S-J-CB',s,5]['tail_rank8_macro_recall']-by['A-CB',s,5]['tail_rank8_macro_recall'] for s in ORDERS]
    if ba_ci['lower_pp']>0:
        decision='BROADER_DESCRIPTIVE_TRANSFER_SIGNAL' if tail_ci['lower_pp']>0 and min(delta)>0 and min(tail)>=0 else ('OVERALL_GAIN_WITH_TAIL_COST' if mean(tail)<0 else 'OVERALL_TRANSFER_SIGNAL_TAIL_UNRESOLVED')
    else:decision='MIXED_TRANSFER' if min(delta)<0<max(delta) else 'NO_CLEAR_TRANSFER'
    access=read(pub/'ACCESS_AUDIT.json');access.update(incremental_neural_epochs_S1_to_S5=0,incremental_optimizer_steps_S1_to_S5=0,
        core_val_metric_rows=54,core_val_per_class_rows=972,new_test_predictions=0,new_test_feature_reads=0,new_test_model_forwards=0,
        historical_data_exposure_disclosed=True,independent_confirmation=False,full_GSR_or_ConCM_reproduction=False,
        analytic_classifier_fits=54,new_val_prediction_units=54)
    assert access['new_formal_S0_runs']==3 and access['new_formal_neural_training_epochs']==30
    csvwrite(pub/'S0_TRAINING_EPOCHS.csv',[json.loads(s) for s in (pub/'S0_TRAINING_EPOCHS.jsonl').read_text().splitlines()])
    write(pub/'ACCESS_AUDIT.json',access);budget()
    resources.update(analytic_CPU_seconds=time.monotonic()-started,CPU_budget_accounted_seconds=cpu_prior+time.monotonic()-started,
        CPU_preflight_unmeasured_reserve_seconds=60.,artifact_peak_bytes=peak,min_free_bytes=minfree)
    write(pub/'RESOURCE_REPORT.json',resources)
    write(pub/'COMPLETION_AUDIT.json',dict(status='COMPLETE_HK1_M1',classification=decision,**access))
    write(pub/'NEXT_DECISION.json',dict(action='STOP',classification=decision,run_more=False,test_released=False,monitoring_created=False))
    lines=['# HK1-M1 固定解析迁移结果','',f'状态：COMPLETE_HK1_M1。描述性结论：{decision}。',
        '','三个全新 HyperKvasir S0 均从锁定 AugReg 初始化，分别完成 10 epoch；后续五阶段神经更新为 0。三父状态与全部 54 个解析分类器先锁定，再统一评价 validation。',
        '',f'数据：fit={protocol["n_train"]}，val={protocol["n_val"]}，统一官方内层 fold={protocol["selected_official_fold"]}；真实 fit 不平衡比={protocol["actual_fit_imbalance_ratio"]:.6g}:1。历史 seed1/fold1 外层成员清理前逐样本差异为 0。',
        '','| 方法 | Final BA | 三顺序样本标准差 | 最差顺序 BA | AvgBA all | AvgBA inc | Tail8 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append('| '+r['method']+' | '+' | '.join(f'{r[k]:.3f}' for k in ('Final_BA','sample_std_BA','worst_seed_BA','AvgBA_all','AvgBA_inc','tail_rank8'))+' |')
    lines+=['',f'主比较 J−A：三配对 Final BA 差值为 {delta} 个百分点；尾类差值为 {tail}。固定父模型条件下 BA 95% 区间 [{ba_ci["lower_pp"]:.3f}, {ba_ci["upper_pp"]:.3f}]，Tail8 区间 [{tail_ci["lower_pp"]:.3f}, {tail_ci["upper_pp"]:.3f}]。',
        '','逐阶段、逐类、尾类正确数/分母/component 数、旧新及基础/后续类、零召回、错误分解和有符号遗忘分别见 CSV。最终两新增类无后续阶段，遗忘记 NA。M 是预设诊断，不替换 J 的主候选身份。',
        '','局限：本轮使用已在历史开发中接触的 HyperKvasir，不能称独立确认。患者/检查隔离未知；精确内容组件并非患者。极少数类的 validation 组数很少，召回与区间分辨率低。三个已训练父模型固定，bootstrap 不推断新训练种子总体；A 最终为确定性参照，其三顺序不是三次神经训练重复。1536 维联合表征比 768 维拥有不同容量，重复嵌入检查仅排除机械尺度错误。解析批量等价不意味着所有旧类召回不降，也不构成隐私或临床有效性证明。',
        '',f'访问：新 test 预测/特征读取/模型前向均为 0；reserved 身份审计文件读取 {access["reserved_identity_audit_file_reads"]} 次、解码 {access["reserved_identity_audit_decodes"]} 次，另行记账。',
        '',f'资源：累计 GPU 驻留 {resources["GPU_process_residence_seconds"]/3600:.3f} 小时；CPU 解析报告 {resources["analytic_CPU_seconds"]:.1f} 秒；观测实验产物峰值 {peak/1024**2:.1f} MiB。输入镜像单列，不占实验产物预算，但计入磁盘保留量。',
        '','NEXT_DECISION=STOP。不释放 test，不增加方法、调参或监测。']
    tail_lines=['','尾类逐项结果（召回为三个固定父模型的平均；A 为确定性参照）：','',
        '| 类别 | fit / val / val组件 | A | J | M | J−A |','|---|---:|---:|---:|---:|---:|']
    for c in groups['tail_rank8']:
        record=pclass['S-J-CB',1993,5,c]
        values={m:mean([pclass[m,s,5,c]['recall'] for s in ORDERS]) for m in METHODS}
        tail_lines.append(f'| {record["official_class_name"]} | {record["n_fit"]} / {record["n_images"]} / {record["n_components"]} | '
            f'{values["A-CB"]:.3f} | {values["S-J-CB"]:.3f} | {values["S-M-CB"]:.3f} | {values["S-J-CB"]-values["A-CB"]:+.3f} |')
    tail_lines+=['','上述差值是已观察事实。对“首阶段适配能否迁移”的解释仅限本固定数据、表示和读出；本实验不能单独区分监督熟悉度、父任务组成、特征维度和未知患者关联的贡献，也未验证临床有效性。']
    at=next(i for i,s in enumerate(lines) if s.startswith('局限：'));lines[at:at]=tail_lines+['']
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    (pub/'RESOURCE_AND_STORAGE_REPORT.md').write_text('# 实际资源与存储\n\n```json\n'+json.dumps(resources,indent=2)+'\n```\n\n'
        '输入镜像单列；实验产物记账包括源端身份审计及运行代码/清单的保守外部额度。GPU 为单 worker，神经阶段退出后才进入独立 CPU 解析报告进程。删除范围只限本轮已验证恢复的滚动临时文件，见 STORAGE_LEDGER.jsonl；未删除或迁移更多历史资产。\n')
    print('COMPLETE_HK1_M1',decision,flush=True)


def main():
    cfg=read(os.environ['P19_CONFIG'])
    with threadpool_limits(limits=4):
        try:run(cfg)
        except BaseException as e:
            write(Path(cfg['root'])/'output/public/FAILURE_analytic.json',dict(status='BLOCKED',reason=str(e),exception=type(e).__name__))
            raise


if __name__=='__main__':main()
