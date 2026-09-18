"""Saved-score prefix analysis only; no models, fitting, or image access."""
import csv,json,os,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from report_locked_holdout_r1 import read,write,csvwrite,bootstrap_weights,boot_unit
from ct2d_math import metrics
import hashlib

def load(path):
    with np.load(path) as f:return {k:f[k].copy() for k in f.files}
def lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def main():
    started=time.monotonic();cfg=read(os.environ['P22_CONFIG']);root=Path(cfg['root']);pub=root/'output/public';private=root/'output/private'
    lock=read(pub/'ONLINE_PREDICTIONS_LOCK.json');assert lock['status']=='LOCKED' and len(lock['units'])==90
    assert read(pub/'ORACLE_COMPLETE.json')['status']=='PASS'
    data={};ms=[];pcs=[];errors=[];mapping={}
    for e in lock['units']:
        path=private/'sealed'/e['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256']
        p=load(path);C=e['seen'];assert C==2*e['task'] and len(p['order'])==C
        meta={k:e[k] for k in ('dataset','seed','task','method')};groups=cfg['frequency_groups'][e['dataset']]
        m,pc,er=metrics(p['raw'],p['y'],p['order'],C-2,groups,meta,p['component'])
        m['prefix_terminal_BA']=m['balanced_accuracy'] if e['task']==3 else None
        m['seen_tail_classes']=int(np.isin(p['order'],groups['tail']).sum())
        ms.append(m);pcs+=pc;errors+=er;key=(e['dataset'],e['seed'],e['task'],e['method']);data[key]=p;mapping[key]=m
    assert len(ms)==90 and len(pcs)==360
    csvwrite(pub/'validation_prefix_metrics.csv',ms);csvwrite(pub/'validation_prefix_per_class.csv',pcs);csvwrite(pub/'error_decomposition.csv',errors)
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            q=[data[name,seed,1,m] for m in ('C0','C1','C2','C3')]
            for p in q[1:]:np.testing.assert_array_equal(p['raw'],q[0]['raw'])
    summaries=[];forget=[]
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            for method in ('P','C0','C1','C2','C3'):
                rr=[mapping[name,seed,t,method] for t in (1,2,3)]
                summaries.append(dict(dataset=name,seed=seed,method=method,prefix_terminal_BA=rr[-1]['balanced_accuracy'],prefix_average_BA=float(np.mean([r['balanced_accuracy'] for r in rr])),old=rr[-1]['old_macro_recall'],current=rr[-1]['current_macro_recall'],tail=rr[-1]['tail_recall']))
                for c in data[name,seed,3,method]['order']:
                    zz=[r for r in pcs if (r['dataset'],r['seed'],r['method'],r['original_label'])==(name,seed,method,int(c))];zz.sort(key=lambda r:r['task'])
                    forget.append(dict(dataset=name,seed=seed,method=method,original_label=int(c),arrival_task=zz[0]['task'],first_recall=zz[0]['recall'],terminal_recall=zz[-1]['recall'],first_to_terminal=None if len(zz)==1 else zz[0]['recall']-zz[-1]['recall'],max_to_terminal=None if len(zz)==1 else max(r['recall'] for r in zz)-zz[-1]['recall']))
    csvwrite(pub/'prefix_summary.csv',summaries);csvwrite(pub/'first_learning_and_forgetting.csv',forget)
    contrasts={'C3-C1':{'C3':1,'C1':-1},'C1-C0':{'C1':1,'C0':-1},'C2-C0':{'C2':1,'C0':-1},'C3-C0':{'C3':1,'C0':-1},'interaction':{'C3':1,'C2':-1,'C1':-1,'C0':1},'C2-P':{'C2':1,'P':-1},'C3-P':{'C3':1,'P':-1}}
    pairs=[];intervals=[];support=[]
    for name in ('HK','ISIC'):
        # Union is assembled from already locked seen-class predictions, never future features.
        identities={}
        for key,p in data.items():
            if key[0]!=name:continue
            for i,sid in enumerate(p['ids']):
                value=(int(p['original'][i]),str(p['component'][i]))
                assert sid not in identities or identities[sid]==value;identities[sid]=value
        ids=sorted(identities);index={sid:i for i,sid in enumerate(ids)}
        canon=dict(y=np.zeros(len(ids),int),original=np.array([identities[x][0] for x in ids]),component=np.array([identities[x][1] for x in ids]))
        weights=bootstrap_weights(canon,2000,47001,sorted(set(canon['original'])))
        for c in sorted(set(canon['original'])):
            n=len(np.unique(canon['component'][canon['original']==c]));support.append(dict(dataset=name,original_label=int(c),components=n,singleton=n==1))
        boot={}
        for key,p in data.items():
            if key[0]!=name:continue
            rec=boot_unit(p,weights[:,[index[x] for x in p['ids']]],True);C=len(p['order']);old=rec[:,:C-2].mean(1) if C>2 else None;cur=rec[:,C-2:].mean(1);tail=np.isin(p['order'],cfg['frequency_groups'][name]['tail'])
            boot[key]=dict(BA=rec.mean(1),old=old,current=cur,tail=rec[:,tail].mean(1) if tail.any() else None,HM=None if old is None else np.divide(2*old*cur,old+cur,out=np.zeros(2000),where=old+cur!=0))
        for task in (1,2,3):
            for contrast,terms in contrasts.items():
                for metric,col in [('BA','balanced_accuracy'),('old','old_macro_recall'),('current','current_macro_recall'),('tail','tail_recall'),('HM','HM')]:
                    means=[];samples=[]
                    for seed in (1993,1994,1995):
                        valid=all(mapping[name,seed,task,m][col] is not None for m in terms)
                        point=sum(w*mapping[name,seed,task,m][col] for m,w in terms.items()) if valid else None
                        sample=sum(w*boot[name,seed,task,m][metric] for m,w in terms.items()) if valid else None
                        row=dict(dataset=name,seed=seed,task=task,contrast=contrast,metric=metric,difference_pp=point)
                        pairs.append(row)
                        if valid:
                            lo,hi=np.quantile(sample,[.025,.975]);intervals.append(dict(row,low=float(lo),high=float(hi)));means.append(point);samples.append(sample)
                    if len(means)==3:
                        lo,hi=np.quantile(np.mean(samples,axis=0),[.025,.975]);intervals.append(dict(dataset=name,seed='fixed_three_mean',task=task,contrast=contrast,metric=metric,difference_pp=float(np.mean(means)),low=float(lo),high=float(hi)))
    csvwrite(pub/'paired_differences.csv',pairs);csvwrite(pub/'bootstrap_intervals.csv',intervals);csvwrite(pub/'component_support.csv',support)
    om=[];opc=[];gaps=[]
    for e in lines(pub/'ORACLE_INDEX.jsonl'):
        path=private/'sealed'/e['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256'];p=load(path)
        meta=dict(dataset=e['dataset'],seed=1993,source=e['source'],method='Q11',identity='OFFLINE_ORACLE_NOT_ONLINE_CIL')
        m,pc,_=metrics(p['raw'],p['y'],p['order'],4,cfg['frequency_groups'][e['dataset']],meta,p['component']);m['prefix_terminal_BA']=m.pop('balanced_accuracy');om.append(m);opc+=pc
        for method in (('C0','C1') if e['source']=='U' else ('C2','C3')):
            online=data[e['dataset'],1993,3,method];assert np.array_equal(online['ids'],p['ids'])
            gaps.append(dict(dataset=e['dataset'],source=e['source'],method=method,online_BA=mapping[e['dataset'],1993,3,method]['balanced_accuracy'],Q11_BA=m['prefix_terminal_BA'],Q11_minus_online=m['prefix_terminal_BA']-mapping[e['dataset'],1993,3,method]['balanced_accuracy']))
    assert len(om)==4 and len(opc)==24
    csvwrite(pub/'oracle_prefix_terminal_metrics.csv',om);csvwrite(pub/'oracle_prefix_terminal_per_class.csv',opc);csvwrite(pub/'oracle_gaps.csv',gaps)
    epochs=lines(pub/'TRAIN_EPOCH_METRICS.jsonl');assert len(epochs)==120 and sum(x['optimizer_steps'] for x in epochs)==6620
    csvwrite(pub/'TRAIN_EPOCH_METRICS.csv',epochs)
    ledger=read(pub/'RESOURCE_AND_ACCESS_LEDGER.json');assert ledger['formal_optimizer_steps']==6620 and ledger['formal_task_epochs']==120 and ledger['new_checkpoints']==12
    ledger.update(status='COMPLETE_CT3P',CPU_report_seconds=time.monotonic()-started,NEXT_DECISION='STOP');write(pub/'RESOURCE_AND_ACCESS_LEDGER.json',ledger)
    text=['# CT3-P 前缀实验结果','', '**COMPLETE_CT3P；NEXT_DECISION=STOP。**','',
          'Task1 为既有两类任务；新增六条 Task2/3 轨迹、120 task-epochs、6620 optimizer steps、12 checkpoint。在线表90/360行，离线表4/24行。',
          '仅比较每个既有 order 的2→4→6类；以下为 prefix_terminal_BA，不是完整8/23类 Final BA。P为同阶段预训练参照。','',
          '| 数据 | 方法 | prefix_terminal_BA mean ± sd | old | current |','|---|---|---:|---:|---:|']
    for name in ('HK','ISIC'):
        for method in ('P','C0','C1','C2','C3'):
            rr=[x for x in summaries if x['dataset']==name and x['method']==method];v=[x['prefix_terminal_BA'] for x in rr]
            text.append(f"| {name} | {method} | {np.mean(v):.3f} ± {np.std(v,ddof=1):.3f} | {np.mean([x['old'] for x in rr]):.3f} | {np.mean([x['current'] for x in rr]):.3f} |")
    text+=['','## 固定比较与局限','']
    for q in intervals:
        if q['seed']=='fixed_three_mean' and q['task']==3 and q['metric']=='BA':text.append(f"- {q['dataset']} {q['contrast']}: {q['difference_pp']:+.3f} pp，条件性95%区间 [{q['low']:+.3f}, {q['high']:+.3f}]。")
    text+=['','区间使用2000次按类component配对bootstrap，seed47001；各seed使用自身已见集合，共享样本使用共同抽样；固定模型，不重采样训练seed。Singleton及反复validation开发限制保留，不是独立确认。',
           'C3−C1是固定T下FD的主机制比较；C3−C0同时改变两个因素。旧类改善与当前类下降必须同时报告。高于弱C0而仍低于P不能证明持续适配净收益。',
           '另见四个Q11诊断、online–Q11差距和旧均值/几何误差。离线旧fit访问只用于锁定后的独立进程，不是合法无回放成绩，也不是必然上界。前缀结果不能外推全部HK11任务。',
           f"在线 old/future fit读取均0；离线旧fit读取 {ledger['offline_diagnostic_old_fit_reads']}；全部test访问0；工程更新 {ledger['engineering_optimizer_steps']}，与正式6620更新分账。", 
           f"GPU累计驻留 {ledger['GPU_process_residence_seconds']/3600:.3f}小时；无额外训练或监测。",'','NEXT_DECISION=STOP。']
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(text))
    write(pub/'NEXT_DECISION.json',dict(status='COMPLETE_CT3P',action='STOP',further_training=False,monitoring=False))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
