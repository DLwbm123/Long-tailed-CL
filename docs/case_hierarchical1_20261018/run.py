"""One frozen campaign; all fitting finishes before any development score is read."""
import csv
import json
import os
from pathlib import Path
import time
import traceback
import torch
from torch.utils.data import DataLoader
from data import read_rows, Images
from core import accumulator, add_patient, finish_accumulator, models, scores, sym


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def metrics(score, labels, seen, old):
    prediction = torch.tensor(seen)[score.argmax(-1)]
    counts = [int(labels.eq(c).sum()) for c in seen]
    correct = [int(((prediction == c) & (labels == c)).sum()) for c in seen]
    recall = [a / b for a, b in zip(correct, counts)]
    confusion = [[int(((labels == a) & (prediction == b)).sum()) for b in seen] for a in seen]
    return dict(macro=sum(recall)/len(recall), micro=float(prediction.eq(labels).double().mean()),
                old_macro=sum(recall[:old])/old if old else None,
                new_macro=sum(recall[old:])/(len(seen)-old) if old < len(seen) else None,
                counts=counts, correct=correct, recall=recall, confusion=confusion)


def main(cfg):
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(cfg['output'])
    out.mkdir(exist_ok=False)
    write_json(out/'config.private.json', cfg)
    start = time.time()
    images_done = 0
    def status(phase, **kw):
        if time.time()-start >= cfg['max_wall_seconds']:
            raise TimeoutError('Frozen campaign wall budget exhausted')
        assert images_done <= cfg['train_count'], 'Training extraction budget exceeded'
        if torch.cuda.is_initialized():
            assert torch.cuda.max_memory_reserved() <= cfg['gpu_reserved_budget_bytes'], 'GPU memory budget exceeded'
        write_json(out/'status.private.json', dict(state='RUNNING', phase=phase,
                   started=start, updated=time.time(), images_extracted=images_done, **kw))
    status('data_check')
    train, val = read_rows(cfg, 'train'), read_rows(cfg, 'val')
    assert len(train)==cfg['train_count'] and len(val)==cfg['val_count']
    assert not ({r['group'] for r in train} & {r['group'] for r in val})
    assert set(r['label'] for r in train)==set(cfg['order'])
    assert cfg['workers']==0 and cfg['dim']==128 and cfg['seed']==74002
    support={str(c):dict(images=sum(r['label']==c for r in train),
                        patients=len({r['group'] for r in train if r['label']==c})) for c in cfg['order']}
    write_json(out/'support.json', support)
    from safetensors.torch import load_file
    import timm
    model=timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    weights={k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')}
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False).cuda()
    prior=Path(cfg['prior_run'])
    prior_cfg=json.loads((prior/'config.private.json').read_text())
    for k in ['seed','dim','manifest','images','weight','author_labels','order','task_sizes']:
        assert prior_cfg[k]==cfg[k], 'Historical feature binding mismatch: '+k
    assert json.loads((prior/'status.private.json').read_text())['state']=='COMPLETE'
    projection=torch.load(prior/'projection.private.pt', weights_only=True).cuda()
    assert projection.shape==(768,128)
    assert torch.allclose(projection.T@projection, torch.eye(128,device='cuda'),atol=2e-6)
    stats=[]
    # Stage order is respected; no future class contributes to the base prior.
    for c in cfg['order']:
        rows=sorted([r for r in train if r['label']==c],key=lambda r:(r['group'],r['relative_path']))
        loader=DataLoader(Images(rows,cfg),batch_size=cfg['batch_size'],shuffle=False,
                          num_workers=0,pin_memory=True)
        acc=accumulator(cfg['dim'],'cuda')
        current=None; count=0
        total=torch.zeros(128,device='cuda',dtype=torch.float64)
        second=torch.zeros(128,128,device='cuda',dtype=torch.float64)
        image_total=total.clone(); image_second=second.clone()
        class_done=0
        for images,idx in loader:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                z=model.forward_features(images.cuda(non_blocking=True))
            assert z.shape[1:]==(197,768)
            patch=(z.float()[:,1:]@projection).double()
            assert torch.isfinite(patch).all()
            groups=[rows[i]['group'] for i in idx.tolist()]
            begin=0
            while begin < len(groups):
                end=begin+1
                while end<len(groups) and groups[end]==groups[begin]: end+=1
                if current is not None and current!=groups[begin]:
                    add_patient(acc,count,total,second)
                    total.zero_();second.zero_();count=0
                current=groups[begin]
                x=patch[begin:end].flatten(0,1)
                a,b=x.sum(0),x.T@x
                total+=a;second+=b;count+=len(x)
                image_total+=a;image_second+=b
                begin=end
            class_done+=len(images); images_done+=len(images)
            status('fit', class_id=c, class_images=class_done, class_total=len(rows))
        add_patient(acc,count,total,second)
        s=finish_accumulator(acc)
        assert s['patients']==support[str(c)]['patients']
        mu=image_total/(len(rows)*cfg['patches'])
        s.update(class_id=c,image_mu=mu,image_cov=sym(image_second/(len(rows)*cfg['patches'])-torch.outer(mu,mu)))
        stats.append({k:v.cpu() if torch.is_tensor(v) else v for k,v in s.items()})
        save(out/f'class_{c:02d}.private.pt',stats[-1])
        print(json.dumps(dict(event='class_fit',class_id=c,patients=s['patients'],images=len(rows))),flush=True)
    # Priors use within-class covariance averages, not cross-class mean scatter.
    base=stats[:cfg['task_sizes'][0]]
    reference={k:torch.stack([s[k] for s in base]).mean(0).cuda() for k in ['W','B','pooled']}
    eps=cfg['eps_relative']*float(reference['pooled'].diag().mean())
    assert eps>0
    save(out/'fitted.private.pt',dict(stats=stats,reference={k:v.cpu() for k,v in reference.items()},eps=eps))
    write_json(out/'FIT_SEAL.json',dict(fitted_classes=len(stats),images_extracted=images_done,
               tau=cfg['tau'],eps_relative=cfg['eps_relative'],eps=eps,
               clipping={str(s['class_id']):s['clipped_eigenvalues'] for s in stats},
               between_unidentified=[s['class_id'] for s in stats if not s['between_identified']],
               fitting_finished=time.time(),query_opened=False))
    del model
    torch.cuda.empty_cache()
    results=[];seen=[];offset=0;previous=None
    for stage,size in enumerate(cfg['task_sizes']):
        old=len(seen);seen+=cfg['order'][offset:offset+size];offset+=size
        rows=[r for r in val if r['label'] in seen]
        labels=torch.tensor([r['label'] for r in rows])
        cached={c:torch.load(prior/f'val_{c:02d}.private.pt',weights_only=True)[0] for c in seen}
        positions={c:0 for c in seen};features=[]
        for r in rows:
            c=r['label'];features.append(cached[c][positions[c]]);positions[c]+=1
        x=torch.stack(features)
        gpu_stats=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in s.items()} for s in stats[:offset]]
        fitted=models(gpu_stats,reference,cfg['patches'],cfg['tau'],eps)
        outputs={}
        for i in range(0,len(rows),cfg['batch_size']):
            status('evaluate',stage=stage,processed=i)
            values=scores(x[i:i+cfg['batch_size']].cuda(),fitted)
            for k,v in values.items():outputs.setdefault(k,[]).append(v.cpu())
        outputs={k:torch.cat(v) for k,v in outputs.items()}
        historical=torch.load(prior/f'stage_{stage}.private.pt',weights_only=True)
        assert historical['seen']==seen and torch.equal(historical['labels'],labels)
        outputs['ridge_historical']=historical['scores']['ridge']
        record=dict(stage=stage,seen=seen.copy(),metrics={k:metrics(v,labels,seen,old) for k,v in outputs.items()},paired={},drift={})
        pred={k:torch.tensor(seen)[v.argmax(-1)] for k,v in outputs.items()}
        for comparator in ['hierarchical_independent','patient_shrunk','patchbell_adapted_entropy','ridge_historical']:
            a,b=pred['hierarchical'],pred[comparator]
            record['paired'][comparator]=dict(rescue=int(((a==labels)&(b!=labels)).sum()),
                harm=int(((a!=labels)&(b==labels)).sum()),changed=int((a!=b).sum()))
        if previous is not None:
            old_seen,old_rows,old_scores=previous
            lookup={r['relative_path']:i for i,r in enumerate(rows)}
            ix=[lookup[r['relative_path']] for r in old_rows]
            for k,v in outputs.items():
                diff=float((v[ix,:len(old_seen)]-old_scores[k]).abs().max())
                record['drift'][k]=diff
                if k not in ['patchbell_adapted_entropy','ridge_historical']:
                    assert diff<1e-7,'Unexpected old-model score drift: '+k
        save(out/f'stage_{stage}.private.pt',dict(scores=outputs,labels=labels,seen=seen))
        results.append(record);write_json(out/'results.json',results)
        previous=seen.copy(),rows,outputs
    costs=dict(wall_seconds=time.time()-start,training_images_extracted=images_done,
               validation_features_reused=len(val),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               attempts=1,optimizer_updates=0,external_test_access=False)
    write_json(out/'costs.json',costs)
    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < cfg['output_budget_bytes']
    report(out)
    write_json(out/'status.private.json',dict(state='COMPLETE',started=start,finished=time.time(),stages=len(results),costs=costs))
    print('COMPLETE',flush=True)


def report(out):
    results=json.loads((out/'results.json').read_text())
    arms=list(results[0]['metrics'])
    lines=['# 病例层联合 patch 证据：固定开发实验','',
           'NIH author20 单标签开发划分；结果为探索性证据，非原 PatchBELL 精确复现。',
           '同一冻结 ViT-B/16、128 维投影、种子及六阶段；无测试集、参数扫描或训练更新。',
           '同参数比较 hierarchical 与 hierarchical_independent；二者只改变 patch 联合分布。', '',
           '| 方法 | 平均阶段 BA (%) | 最终 BA (%) |','|---|---:|---:|']
    for arm in arms:
        average=sum(r['metrics'][arm]['macro'] for r in results)/len(results)
        lines.append(f'| {arm} | {100*average:.3f} | {100*results[-1]["metrics"][arm]["macro"]:.3f} |')
    primary=results[-1]['paired']['hierarchical_independent']
    lines+=['',f'最终同参数相关性对照：救回 {primary["rescue"]}，损害 {primary["harm"]}，改变 {primary["changed"]} 张。',
            '', '病例间效应是同患者 patch 的共同偏移；R=I 不覆盖所有位置相关性。',
            'W 采用患者内无偏样本协方差的等患者平均；B 采用患者均值样本协方差减噪并截断负特征值。',
            '共同类内 W、patch 条件独立、PSD 截断及基类先验均是模型假设，不能把 B 的全部变化都解释为无关噪声。',
            '先验只取基类，tau=20、协方差地板=基类平均对角方差的 .001，预先固定；无参数选择。',
            '有患者 ID 且训练/开发患者不重叠，但开发集已反复使用，仅 300 张；指标按图像计算。',
            '每类保存两套协方差，内存大于单高斯；不声称与 16 个历史特征等预算或 TMI 级结论。',
            '原论文 DINOv3 / 1024维 / k=500 尚未复现；熵对照为同特征完整128维高斯加附录A4，加入共同数值地板。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['stage','arm','BA','micro','old_BA','new_BA'])
        for r in results:
            for k,m in r['metrics'].items():writer.writerow([r['stage'],k,m['macro'],m['micro'],m['old_macro'],m['new_macro']])


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        out=Path(cfg['output']);out.mkdir(exist_ok=True)
        path=out/'status.private.json'
        old=json.loads(path.read_text()) if path.exists() else {}
        old.update(state='FAILED',updated=time.time(),error=traceback.format_exc())
        write_json(path,old)
        traceback.print_exc()
        raise
