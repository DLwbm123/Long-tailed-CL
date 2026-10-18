"""One new feature pass for missing image-mean statistics; fixed readouts only."""
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from torch.utils.data import DataLoader
from core import weights, augmented, project_moments, ridge, gda_parameters, gda_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'case_hierarchical1_20261018'))
from data import read_rows, Images


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def save(path,value):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


def metrics(value,labels,seen,old):
    pred=torch.tensor(seen)[value.argmax(1)]
    counts=[int((labels==c).sum()) for c in seen]
    correct=[int(((labels==c)&(pred==c)).sum()) for c in seen]
    recall=[a/b for a,b in zip(correct,counts)]
    return dict(macro=sum(recall)/len(recall),micro=float((pred==labels).double().mean()),
                old_macro=sum(recall[:old])/old if old else None,
                new_macro=sum(recall[old:])/(len(seen)-old),counts=counts,correct=correct,recall=recall,
                confusion=[[int(((labels==c)&(pred==p)).sum()) for p in seen] for c in seen])


def main(cfg):
    torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(cfg['output']);root.mkdir(exist_ok=False)
    write_json(root/'config.private.json',cfg)
    start=time.time();new_train=0;new_val=0
    def status(phase,**kw):
        if time.time()-start>=cfg['max_wall_seconds']:raise TimeoutError('Frozen wall budget exhausted')
        assert new_train<=cfg['train_count'] and new_val<=cfg['val_count']
        if torch.cuda.is_initialized():
            assert torch.cuda.max_memory_reserved()<=cfg['gpu_reserved_budget_bytes'],'GPU memory budget exceeded'
        write_json(root/'status.private.json',dict(state='RUNNING',phase=phase,started=start,updated=time.time(),
                   new_training_images=new_train,new_validation_images=new_val,**kw))
    status('data_check')
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    assert len(train)==cfg['train_count'] and len(val)==cfg['val_count']
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    assert cfg['workers']==0 and cfg['batch_size']==64
    prior=Path(cfg['prior_run']);old=json.loads((prior/'config.private.json').read_text())
    for key in ['seed','dim','order','task_sizes','manifest','images','weight','author_labels']:
        assert old[key]==cfg[key],'Historical binding mismatch: '+key
    assert json.loads((prior/'status.private.json').read_text())['state']=='COMPLETE'
    original=[torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
    assert all(s['class_id']==c for s,c in zip(original,cfg['order']))
    audit=json.loads((prior/'support.json').read_text())
    write_json(root/'support.json',audit)
    projection=torch.load(prior/'projection.private.pt',weights_only=True).double().cuda()
    assert projection.shape==(768,128)
    assert torch.allclose(projection.T@projection,torch.eye(128,device='cuda',dtype=torch.float64),atol=2e-6)
    from safetensors.torch import load_file
    import timm
    encoder=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    encoder.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    encoder.eval().requires_grad_(False).cuda()
    def extract(rows):
        loader=DataLoader(Images(rows,cfg),batch_size=cfg['batch_size'],num_workers=0,shuffle=False)
        for images,idx in loader:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                z=encoder.forward_features(images.cuda())
            assert z.shape[1:]==(197,768)
            yield z[:,1:].float().mean(1).double(),z[:,0].double(),idx
    patch_grams=[];patch_targets=[];train_parity={}
    for c,s in zip(cfg['order'],original):
        rows=[r for r in train if r['label']==c]
        assert len(rows)==audit[str(c)]['images']
        assert len({r['group'] for r in rows})==audit[str(c)]['patients']==s['groups']
        w=weights([r['group'] for r in rows],'cuda')
        assert abs(float(w.sum())-1)<1e-12
        gram=torch.zeros(769,769,device='cuda',dtype=torch.float64)
        target=torch.zeros(769,device='cuda',dtype=torch.float64)
        cls_mean=torch.zeros(768,device='cuda',dtype=torch.float64);done=0
        for patch_mean,cls,idx in extract(rows):
            mass=w[idx.cuda()];x=augmented(patch_mean)
            gram+=x.T@(x*mass[:,None]);target+=(x*mass[:,None]).sum(0)
            cls_mean+=(cls*mass[:,None]).sum(0)
            done+=len(idx);new_train+=len(idx)
            if done==len(idx) or done%1024<cfg['batch_size']:
                status('fit_missing_moments',class_id=c,class_processed=done,class_total=len(rows))
        error=float((cls_mean.cpu()-s['ridge_xy'][:-1]).abs().max())
        assert error<=cfg['feature_atol'],'Training CLS parity failed'
        train_parity[str(c)]=error
        patch_grams.append(gram.cpu());patch_targets.append(target.cpu())
        save(root/f'class_{c:02d}.private.pt',dict(patch_mean_gram=gram.cpu(),patch_mean_target=target.cpu()))
        print(json.dumps(dict(event='class_complete',class_id=c,images=done,CLS_mean_error=error)),flush=True)
    assert new_train==cfg['train_count']
    families={'cls768':([s['ridge_xx'].cuda() for s in original],[s['ridge_xy'].cuda() for s in original]),
              'patchmean768':([g.cuda() for g in patch_grams],[v.cuda() for v in patch_targets])}
    for stem in ['cls','patchmean']:
        pairs=[project_moments(a,b,projection) for a,b in zip(*families[stem+'768'])]
        families[stem+'128']=([a for a,b in pairs],[b for a,b in pairs])
    stages=[];offset=0
    for stage,size in enumerate(cfg['task_sizes']):
        offset+=size
        heads={stem+'_ridge':ridge(a[:offset],b[:offset],cfg['ridge_alpha']).cpu() for stem,(a,b) in families.items()}
        stages.append(heads)
        save(root/f'heads_{stage}.private.pt',heads)
        status('fit_readouts',stage=stage)
    groups=[s['groups'] for s in original];base=cfg['task_sizes'][0]
    gda={};epsilons={}
    for stem in ['cls128','patchmean128']:
        fitted,eps=gda_parameters(families[stem][0],groups,base,cfg['gda_tau'],cfg['eps_relative'])
        gda[stem]=fitted;epsilons[stem]=eps
    save(root/'gda.private.pt',{k:[tuple(v.cpu() for v in part) for part in model] for k,model in gda.items()})
    write_json(root/'FIT_SEAL.json',dict(fitted_classes=20,stages=6,new_training_images=new_train,query_opened=False,
               frozen_at=time.time(),ridge_alpha=cfg['ridge_alpha'],gda_tau=cfg['gda_tau'],epsilons=epsilons,
               train_CLS_max_abs_error=train_parity))
    # One query extraction per class matches the historical cache's batch shape.
    query={};parity={}
    for c in cfg['order']:
        rows=[r for r in val if r['label']==c]
        blocks=[(p.cpu(),h.cpu()) for p,h,_ in extract(rows)]
        pm=torch.cat([p for p,h in blocks]);fresh_cls=torch.cat([h for p,h in blocks])
        cached_patch,_,cached_cls=torch.load(prior/f'val_{c:02d}.private.pt',weights_only=True)
        projected=pm@projection.cpu()
        cls_error=float((fresh_cls-cached_cls.double()).abs().max())
        pm_error=float((projected-cached_patch.double().mean(1)).abs().max())
        assert cls_error<=cfg['feature_atol'] and pm_error<=cfg['feature_atol'],'Validation feature parity failed'
        parity[str(c)]=dict(CLS_max_abs=cls_error,patch_mean128_max_abs=pm_error)
        query[c]=dict(cls768=cached_cls.double(),patchmean768=pm,
                      cls128=cached_cls.double()@projection.cpu(),patchmean128=projected)
        new_val+=len(rows);status('extract_validation',class_id=c)
    write_json(root/'PARITY.json',dict(validation=parity,feature_atol=cfg['feature_atol']))
    del encoder
    torch.cuda.empty_cache()
    contrasts=[('cls128_ridge','cls768_ridge'),('patchmean128_ridge','patchmean768_ridge'),
               ('patchmean768_ridge','cls768_ridge'),('patchmean128_ridge','cls128_ridge'),
               ('patchmean128_gda','patchmean128_ridge'),('cls128_gda','cls128_ridge')]
    results=[];offset=0
    for stage,size in enumerate(cfg['task_sizes']):
        old_count=offset;offset+=size;seen=cfg['order'][:offset]
        rows=[r for r in val if r['label'] in seen];labels=torch.tensor([r['label'] for r in rows])
        positions={c:0 for c in seen};rows_features={stem:[] for stem in families}
        for row in rows:
            c=row['label'];i=positions[c];positions[c]+=1
            for stem in families:rows_features[stem].append(query[c][stem][i])
        rows_features={k:torch.stack(v) for k,v in rows_features.items()}
        outputs={stem+'_ridge':augmented(x)@stages[stage][stem+'_ridge'] for stem,x in rows_features.items()}
        for stem in ['cls128','patchmean128']:
            outputs[stem+'_gda']=gda_score(rows_features[stem].cuda(),gda[stem][:offset]).cpu()
        historical=torch.load(prior/f'stage_{stage}.private.pt',weights_only=True)
        assert historical['seen']==seen and torch.equal(historical['labels'],labels)
        err=float((outputs['cls768_ridge']-historical['scores']['ridge']).abs().max())
        assert err<cfg['score_atol'] and torch.equal(outputs['cls768_ridge'].argmax(1),historical['scores']['ridge'].argmax(1))
        outputs['patchgaussian128_historical']=historical['scores']['uniform']
        record=dict(stage=stage,seen=seen,metrics={k:metrics(v,labels,seen,old_count) for k,v in outputs.items()},
                    contrasts={},historical_ridge_score_error=err)
        predictions={k:torch.tensor(seen)[v.argmax(1)] for k,v in outputs.items()}
        for candidate,reference in contrasts:
            a,b=predictions[candidate],predictions[reference]
            record['contrasts'][candidate+'-'+reference]=dict(delta_BA=record['metrics'][candidate]['macro']-record['metrics'][reference]['macro'],
                  rescue=int(((a==labels)&(b!=labels)).sum()),harm=int(((a!=labels)&(b==labels)).sum()),changed=int((a!=b).sum()))
        save(root/f'stage_{stage}.private.pt',dict(scores=outputs,labels=labels,seen=seen))
        results.append(record);write_json(root/'results.json',results)
        status('stage_complete',stage=stage)
    resource=dict(wall_seconds=time.time()-start,new_training_images=new_train,new_validation_images=new_val,
                  GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  optimizer_updates=0,attempts=1,historical_feature_creation_excluded=True)
    write_json(root/'costs.json',resource)
    assert sum(p.stat().st_size for p in root.iterdir() if p.is_file())<cfg['output_budget_bytes']
    report(root,results,resource)
    write_json(root/'status.private.json',dict(state='COMPLETE',started=start,finished=time.time(),stages=6,costs=resource))
    print('COMPLETE',flush=True)


def report(root,results,costs):
    lines=['# NIH同条件投影、表示与读出归因','',
           '固定患者隔离的既有单标签开发划分、种子74002、六阶段及相同冻结骨干；结果为探索性诊断。',
           '所有ridge使用原始特征、等患者等类权重、alpha=1和正则化截距，无参数选择。',
           '128维GDA使用对应相同图像向量与权重，tau=20、仅基类共享协方差目标及.001相对地板固定。','',
           '| 方法 | 平均阶段 BA (%) | 最终 BA (%) |','|---|---:|---:|']
    for k in results[-1]['metrics']:
        avg=sum(r['metrics'][k]['macro'] for r in results)/len(results)
        lines.append(f'| {k} | {100*avg:.3f} | {100*results[-1]["metrics"][k]["macro"]:.3f} |')
    lines+=['','| 最终成对对照 | BA变化 pp | 救回 | 损害 |','|---|---:|---:|---:|']
    for k,v in results[-1]['contrasts'].items():lines.append(f'| {k} | {100*v["delta_BA"]:+.3f} | {v["rescue"]} | {v["harm"]} |')
    lines+=['','投影比较对应固定子空间约束，表示比较只替换CLS/patch均值，128维头比较使用相同图像向量。',
            'GDA与ridge的正则族不同；仅能比较这些预先固定的规则，不能否定所有高斯或判别头。',
            '旧逐patch高斯与图像均值GDA改变了统计单位和评分汇总，不能归因纯换头。',
            'patch均值丢弃空间与分布信息；负结果不能证明完整patch集合无价值。',
            '没有改变病例层实验或DENO，没有新增种子、数据集或测试集访问；开发集已反复使用。',
            f'新增编码训练{costs["new_training_images"]}张、验证{costs["new_validation_images"]}张，墙钟{costs["wall_seconds"]:.2f}秒；历史成本另计。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['stage','arm','BA','micro','old_BA','new_BA'])
        for r in results:
            for k,m in r['metrics'].items():writer.writerow([r['stage'],k,m['macro'],m['micro'],m['old_macro'],m['new_macro']])


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        root=Path(cfg['output']);root.mkdir(exist_ok=True)
        p=root/'status.private.json';state=json.loads(p.read_text()) if p.exists() else {}
        state.update(state='FAILED',updated=time.time(),error=traceback.format_exc())
        write_json(p,state);traceback.print_exc();raise
