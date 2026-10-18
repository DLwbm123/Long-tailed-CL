"""Fixed DENO transfer, reusing immutable class statistics and development features."""
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'case_hierarchical1_20261018'))
from core import gaussian, logpdf, entropy_pool
from data import read_rows, Images
from deno import pool, kernel


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def save(path, value):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def metric(value, labels, seen, old):
    prediction=torch.tensor(seen)[value.argmax(-1)]
    counts=[int((labels==c).sum()) for c in seen]
    correct=[int(((labels==c)&(prediction==c)).sum()) for c in seen]
    recall=[a/b for a,b in zip(correct,counts)]
    return dict(macro=sum(recall)/len(recall),micro=float((prediction==labels).double().mean()),
                old_macro=sum(recall[:old])/old if old else None,
                new_macro=sum(recall[old:])/(len(seen)-old),counts=counts,correct=correct,
                recall=recall,confusion=[[int(((labels==c)&(prediction==p)).sum()) for p in seen] for c in seen])


def main(cfg):
    torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(cfg['output']);root.mkdir(exist_ok=False)
    write_json(root/'config.private.json',cfg)
    start=time.time();extracted=0
    def status(phase,**extra):
        assert time.time()-start < cfg['max_wall_seconds'],'Wall budget exhausted'
        assert extracted<=cfg['max_calibration_images'],'Calibration extraction budget exceeded'
        if torch.cuda.is_initialized():
            assert torch.cuda.max_memory_reserved()<=cfg['gpu_reserved_budget_bytes'],'GPU reserve budget exceeded'
        write_json(root/'status.private.json',dict(state='RUNNING',phase=phase,started=start,updated=time.time(),
                   new_training_images=extracted,**extra))
    status('data_check')
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    assert len(train)==cfg['train_count'] and len(val)==cfg['val_count']
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    prior=Path(cfg['prior_run'])
    original=json.loads((prior/'config.private.json').read_text())
    for key in ['seed','dim','order','task_sizes','manifest','images','weight','author_labels']:
        assert original[key]==cfg[key],'Historical binding mismatch: '+key
    assert json.loads((prior/'status.private.json').read_text())['state']=='COMPLETE'
    audit=json.loads((prior/'support.json').read_text())
    assert all(audit[str(c)]['images']==sum(r['label']==c for r in train) for c in cfg['order'])
    write_json(root/'support.json',audit)
    stats=[torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
    assert all(s['class_id']==c for s,c in zip(stats,cfg['order']))
    reference=torch.load(prior/'reference.private.pt',weights_only=True)['local']
    mu0,cov0,eps=reference
    base_count=cfg['task_sizes'][0]
    expected_mu=torch.stack([s['local'][0] for s in stats[:base_count]]).mean(0)
    assert torch.allclose(expected_mu,mu0,atol=1e-10)
    fitted=[gaussian(s['local'][0].cuda(),s['local'][1].cuda()+eps*torch.eye(cfg['dim'],device='cuda',dtype=torch.float64)) for s in stats]
    ref_model=gaussian(mu0.cuda(),cov0.cuda()+eps*torch.eye(cfg['dim'],device='cuda',dtype=torch.float64))
    projection=torch.load(prior/'projection.private.pt',weights_only=True).cuda()
    assert projection.shape==(768,128)
    generator=random.Random(cfg['seed']);calibration=[];cal_counts={}
    for c in cfg['order'][:base_count]:
        rows=sorted([r for r in train if r['label']==c],key=lambda r:r['relative_path'])
        first={}
        for row in rows:first.setdefault(row['group'],row)
        groups=generator.sample(sorted(first),min(cfg['patients_per_base_class'],len(first)))
        selected=[first[g] for g in sorted(groups)]
        calibration.extend(selected);cal_counts[str(c)]=len(selected)
    assert len(calibration)<=cfg['max_calibration_images']
    from safetensors.torch import load_file
    import timm
    model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    model.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    model.eval().requires_grad_(False).cuda()
    loader=DataLoader(Images(calibration,cfg),batch_size=cfg['batch_size'],num_workers=0,shuffle=False)
    class_squared={c:[] for c in cfg['order'][:base_count]}
    class_distances={c:[] for c in class_squared}
    pair_mask=~torch.eye(cfg['patches'],dtype=torch.bool,device='cuda')
    for images,indices in loader:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            z=model.forward_features(images.cuda())
        assert z.shape[1:]==(197,768)
        patch=z.float()[:,1:]@projection
        raw=torch.stack([logpdf(patch,m) for m in fitted[:base_count]],-1)
        ratios=raw-logpdf(patch,ref_model)[...,None]
        normalized=torch.nn.functional.normalize(patch.double(),dim=-1)
        distances=torch.cdist(normalized,normalized).square()
        for j,i in enumerate(indices.tolist()):
            c=calibration[i]['label']
            class_squared[c].append(float(ratios[j].square().mean()))
            class_distances[c].append(float(distances[j][pair_mask].median())/2)
        extracted+=len(images);status('calibration',processed=extracted)
    scale=(sum(sum(v)/len(v) for v in class_squared.values())/base_count)**.5
    bandwidth_squared=sum(sum(v)/len(v) for v in class_distances.values())/base_count
    assert scale>1e-12 and bandwidth_squared>1e-12
    bandwidth=bandwidth_squared**.5
    seal=dict(scale=scale,bandwidth=bandwidth,beta=cfg['beta'],tau=cfg['determinant_tau'],clip=cfg['evidence_clip'],
              calibration_counts=cal_counts,images_extracted=extracted,query_opened=False,frozen_at=time.time(),
              class_models='immutable historical patient-equal unshrunk Gaussian',reference='base-only fixed Gaussian')
    write_json(root/'FIT_SEAL.json',seal)
    del model
    torch.cuda.empty_cache()
    # Class-separable DENO scores can be computed once. Stages select the seen
    # prefix; entropy weights are recomputed for each stage's competitor set.
    cached={c:torch.load(prior/f'val_{c:02d}.private.pt',weights_only=True)[0] for c in cfg['order']}
    positions={c:0 for c in cfg['order']};features=[]
    for row in val:
        c=row['label'];features.append(cached[c][positions[c]]);positions[c]+=1
    features=torch.stack(features)
    labels=torch.tensor([r['label'] for r in val])
    all_outputs={};raw_blocks=[];ratios_blocks=[]
    infer_start=time.time()
    for i in range(0,len(val),cfg['batch_size']):
        x=features[i:i+cfg['batch_size']].cuda()
        raw=torch.stack([logpdf(x,m) for m in fitted],-1)
        reference_scores=logpdf(x,ref_model)
        values,ratios=pool(x,raw,reference_scores,scale,bandwidth,cfg['evidence_clip'],cfg['beta'],cfg['determinant_tau'])
        values['uniform_raw']=raw.mean(1)
        for key,value in values.items():all_outputs.setdefault(key,[]).append(value.cpu())
        raw_blocks.append(raw.cpu());ratios_blocks.append(ratios.cpu())
        status('scoring',processed=min(i+cfg['batch_size'],len(val)),total=len(val))
    scoring_seconds=time.time()-infer_start
    all_outputs={k:torch.cat(v) for k,v in all_outputs.items()}
    raw_all=torch.cat(raw_blocks);ratios_all=torch.cat(ratios_blocks)
    clipped=ratios_all.abs()>cfg['evidence_clip']
    numerical=[]
    for col,c in enumerate(cfg['order']):
        i=int((labels==c).nonzero()[0]);x=features[i:i+1].cuda()
        raw=raw_all[i:i+1,:,col:col+1].cuda()
        ref=logpdf(x,ref_model)
        doubled,_=pool(x.repeat_interleave(2,1),raw.repeat_interleave(2,1),ref.repeat_interleave(2,1),
                       scale,bandwidth,cfg['evidence_clip'],cfg['beta'],cfg['determinant_tau'])
        error=float(abs(doubled['deno_mixture'][0,0].cpu()-all_outputs['deno_mixture'][i,col]))
        numerical.append(dict(class_id=c,uniform_replication_error=error))
        assert error<1e-7,'Uniform replication property failed'
    x=features[:cfg['batch_size']].cuda();raw=raw_all[:len(x),:,:base_count].cuda()
    prefix,_=pool(x,raw,logpdf(x,ref_model),scale,bandwidth,cfg['evidence_clip'],cfg['beta'],cfg['determinant_tau'])
    prefix_error=float((prefix['deno_mixture'].cpu()-all_outputs['deno_mixture'][:len(x),:base_count]).abs().max())
    assert prefix_error<1e-10
    write_json(root/'NUMERICAL_CHECKS.json',dict(uniform_replication=numerical,prefix_recompute_error=prefix_error,
               clip_total=int(clipped.sum()),clip_denominator=clipped.numel(),
               per_class_clip=[int(clipped[:,:,i].sum()) for i in range(len(cfg['order']))]))
    results=[];offset=0;previous=None
    for stage,size in enumerate(cfg['task_sizes']):
        old=offset;offset+=size;seen=cfg['order'][:offset]
        ix=torch.tensor([i for i,r in enumerate(val) if r['label'] in seen])
        stage_labels=labels[ix]
        outputs={k:v[ix,:offset] for k,v in all_outputs.items()}
        outputs['patchbell_entropy']=entropy_pool(raw_all[ix,:,:offset])
        historical=torch.load(prior/f'stage_{stage}.private.pt',weights_only=True)
        assert historical['seen']==seen and torch.equal(historical['labels'],stage_labels)
        outputs['ridge_historical']=historical['scores']['ridge']
        assert torch.equal(outputs['uniform_raw'].argmax(-1),historical['scores']['uniform'].argmax(-1))
        assert torch.equal(outputs['reference_mean_raw'].argmax(-1),outputs['uniform_raw'].argmax(-1))
        assert torch.equal(outputs['identity_mixture'].argmax(-1),outputs['reference_mean_clipped'].argmax(-1))
        record=dict(stage=stage,seen=seen,metrics={k:metric(v,stage_labels,seen,old) for k,v in outputs.items()},paired={},drift={})
        preds={k:torch.tensor(seen)[v.argmax(-1)] for k,v in outputs.items()}
        for comparator in ['reference_mean_clipped','uniform_raw','patchbell_entropy','ridge_historical']:
            a,b=preds['deno_mixture'],preds[comparator]
            record['paired'][comparator]=dict(rescue=int(((a==stage_labels)&(b!=stage_labels)).sum()),
                    harm=int(((a!=stage_labels)&(b==stage_labels)).sum()),changed=int((a!=b).sum()))
        if previous is not None:
            old_ix,old_outputs=previous
            locations={int(v):i for i,v in enumerate(ix)}
            indices=[locations[int(v)] for v in old_ix]
            for k,v in outputs.items():
                diff=float((v[indices,:old]-old_outputs[k]).abs().max())
                record['drift'][k]=diff
                if k not in ['patchbell_entropy','ridge_historical']:assert diff<1e-7
        save(root/f'stage_{stage}.private.pt',dict(scores=outputs,labels=stage_labels,seen=seen))
        previous=ix,outputs
        results.append(record);write_json(root/'results.json',results)
    costs=dict(wall_seconds=time.time()-start,scoring_seconds=scoring_seconds,new_training_images=extracted,
               unique_validation_images=len(val),class_scores_computed_once=True,replication_checks=20,
               gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               optimizer_updates=0,attempts=1,new_test_access=False)
    write_json(root/'costs.json',costs)
    assert sum(p.stat().st_size for p in root.iterdir() if p.is_file())<cfg['output_budget_bytes']
    report(root,results,costs,seal)
    write_json(root/'status.private.json',dict(state='COMPLETE',started=start,finished=time.time(),stages=6,costs=costs))
    print('COMPLETE',flush=True)


def report(root,results,costs,seal):
    lines=['# 固定 DENO 聚合迁移：NIH 开发结果','',
           '同一冻结 ViT-B/16、128维特征、原等患者高斯、seed74002与六阶段；单标签开发集300张，非独立测试。',
           'DENO为固定参数迁移，非原论文训练复现；PatchBELL为同特征高斯/A4改写，非原DINOv3配置。',
           '主比较 deno_mixture 对 reference_mean_clipped，二者使用相同证据和截断。beta=.5、tau=1固定；带宽与尺度仅由基类训练标定。','',
           '| 方法 | 平均阶段BA (%) | 最终BA (%) |','|---|---:|---:|']
    for k in results[-1]['metrics']:
        avg=sum(r['metrics'][k]['macro'] for r in results)/6
        lines.append(f'| {k} | {100*avg:.3f} | {100*results[-1]["metrics"][k]["macro"]:.3f} |')
    lines+=['','| 最终相对基线 | 救回 | 损害 | 改变预测 |','|---|---:|---:|---:|']
    for k,v in results[-1]['paired'].items():lines.append(f'| {k} | {v["rescue"]} | {v["harm"]} | {v["changed"]} |')
    lines+=['',f'新增训练标定图像 {costs["new_training_images"]}；墙钟 {costs["wall_seconds"]:.2f} 秒；独立评分 {costs["scoring_seconds"]:.2f} 秒。',
            f'固定尺度 {seal["scale"]:.8g}，核带宽 {seal["bandwidth"]:.8g}。',
            '参考密度在非线性聚合中不自动抵消；截断本身可能影响预测，因此单列无截断/截断均值。',
            '旧类分数固定不保证分类不遗忘；新增类仍可竞争胜出。',
            '均匀全袋复制数值核验不等于任意选择性复制不变，也不保证临床正确。',
            '只有一种固定迁移配置、一个开发划分和一个种子；负结果不证明DENO方法族无效，正结果不证明临床泛化或新颖性。',
            '既有类统计、骨干、投影与特征生成成本未计入本轮新增墙钟；报告的是新增开销。',
            '未修改并行病例层实验，不扫参，不自动重试。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['stage','arm','BA','micro','old_BA','new_BA'])
        for r in results:
            for k,m in r['metrics'].items():w.writerow([r['stage'],k,m['macro'],m['micro'],m['old_macro'],m['new_macro']])


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        root=Path(cfg['output']);root.mkdir(exist_ok=True)
        p=root/'status.private.json';state=json.loads(p.read_text()) if p.exists() else {}
        state.update(state='FAILED',updated=time.time(),error=traceback.format_exc())
        write_json(p,state)
        traceback.print_exc()
        raise
