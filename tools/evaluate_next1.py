"""Sealed-until-coordinator evaluation and the fixed historical score audit."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from run_multilabel import ApartFeatures
from run_prototype_single import Images, manifests, extract, save
from next1_support import summarize


def run(config):
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def budget():
        if time.monotonic()-start>config['max_wall_seconds']:raise TimeoutError('Evaluation residence cap')
    save(root/'STATUS.json',dict(status='RUNNING',started=time.time(),optimizer_updates=0))
    try:
        if not config.get('historical_audit'):
            gate=json.loads(Path(config['evaluation_gate']).read_text())
            if gate['phase'] not in ('EVALUATE_MAIN','EVALUATE_ROBUSTNESS'):raise ValueError('Validation is still sealed')
        torch.set_num_threads(4);seed=config['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        encoder=ApartFeatures(config['legacy_repo'],config['weight'],8,'cuda:0',seed)
        from run_medical_v2 import transform
        train,val=manifests(config);ranked=sorted(config['order'],key=lambda c:(-sum(r['label']==c for r in train),c));tail=set(ranked[4:])
        for name, model in config['models'].items():
            reports=[];diagnostics=[]
            for task in range(1,5):
                budget();state=torch.load(Path(model)/f'stage_{task}.pt',map_location='cpu',weights_only=False)
                encoder.load_state_dict(state.get('model',state['adapter']),strict='model' in state);encoder.eval();seen=state['seen']
                rows=[r for r in val if r['label'] in seen]
                loader=DataLoader(Images(rows,config['images'],transform(False),seed+task*100003),batch_size=64,num_workers=4,
                    multiprocessing_context='spawn',persistent_workers=True,generator=torch.Generator().manual_seed(seed+task*2003))
                z,y=extract(encoder,loader,budget);scores=z@state['head'].numpy();pred=np.asarray(seen)[scores.argmax(1)]
                recalls={str(c):float((pred[y==c]==c).mean()) for c in seen}
                report=dict(task=task,seen=seen,validation_n=len(y),accuracy=float((pred==y).mean()),
                    balanced_accuracy=float(np.mean(list(recalls.values()))),per_class_recall=recalls,
                    per_class_n={str(c):int((y==c).sum()) for c in seen},
                    tail_recall=float(np.mean([recalls[str(c)] for c in seen if c in tail])) if tail.intersection(seen) else None)
                reports.append(report);diagnostics.append(dict(task=task,**summarize(scores,y,seen)))
                private=root/(name+f'_stage_{task}.private.npz')
                np.savez_compressed(private,scores=scores,labels=y,prediction=pred)
                del loader,state,z,scores
            forgetting=[];learning={}
            for c in config['order']:
                available=[r for r in reports if str(c) in r['per_class_recall']]
                values=[r['per_class_recall'][str(c)] for r in available]
                drop=max(values[:-1])-values[-1] if len(values)>1 else None
                if drop is not None:forgetting.append(drop)
                learning[str(c)]=dict(first_task=available[0]['task'],first_recall=values[0],final_recall=values[-1],
                    first_minus_final=values[0]-values[-1],standard_forgetting=drop)
            metrics=dict(stages=reports,average_incremental_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reports])),
                final_balanced_accuracy=reports[-1]['balanced_accuracy'],final_tail_recall=reports[-1]['tail_recall'],
                forgetting=float(np.mean(forgetting)),forgetting_classes=len(forgetting),test_accessed=False,independent_confirmation=False)
            if config.get('historical_audit'):
                expected=json.loads((Path(model)/'metrics.json').read_text())
                for a,b in zip(reports,expected['stages']):
                    if a['per_class_recall']!=b['per_class_recall']:raise ValueError('Historical recall reconstruction differs')
            save(root/(name+'.json'),dict(metrics=metrics,score_diagnostics=diagnostics,class_learning=learning))
        save(root/'STATUS.json',dict(status='COMPLETE',elapsed_seconds=time.monotonic()-start,optimizer_updates=0))
    except BaseException as exc:
        save(root/'STATUS.json',dict(status='INCOMPLETE',elapsed_seconds=time.monotonic()-start,error=str(exc),optimizer_updates=0));raise


def pair_tables(paths, names):
    result=[]
    for stage in range(1,5):
        values={n:np.load(Path(paths[n])/(n+f'_stage_{stage}.private.npz')) for n in names}
        for i,a in enumerate(names):
            for b in names[i+1:]:
                x,y=values[a],values[b];assert np.array_equal(x['labels'],y['labels'])
                ca=x['prediction']==x['labels'];cb=y['prediction']==y['labels']
                result.append(dict(stage=stage,a=a,b=b,n=len(ca),prediction_agreement=float((x['prediction']==y['prediction']).mean()),
                    both_correct=int((ca&cb).sum()),a_only_correct=int((ca&~cb).sum()),b_only_correct=int((~ca&cb).sum()),both_wrong=int((~ca&~cb).sum())))
        for v in values.values():v.close()
    return result


if __name__=='__main__':run(json.load(sys.stdin))
