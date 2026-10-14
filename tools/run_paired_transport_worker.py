"""Authorized train entry and sealed paired val aggregates; no prediction files."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import paired_moment_transport as pmt
import core1_competition as competition
import run_module_additions as training
import module_additions as modules
from lt_benchmark import benchmark_metrics,task_blocks
from run_multilabel import ApartFeatures
from run_prototype_single import Images,extract,manifests,save
from run_paired_transport_cycle import validate_job,check_barrier,checkpoint_metadata,STEPS,label


def paired(labels,seen,on,off,on_scores,off_scores,old_count):
    lookup={c:i for i,c in enumerate(seen)};y=np.array([lookup[int(c)] for c in labels])
    counts=[];confusions={}
    for name,pred in [('ON',on),('CONTROL',off)]:
        matrix=np.zeros((len(seen),len(seen)),dtype=int)
        for truth,guess in zip(y,pred):matrix[truth,guess]+=1
        confusions[name]=matrix.tolist()
    for i,c in enumerate(seen):
        pick=y==i;n=int(pick.sum());correct_on=on==y;correct_off=off==y
        other=np.arange(len(seen))!=i
        old_new=dict(role='old' if i<old_count else 'new')
        counts.append(dict(label=c,n=n,**old_new,correct_on=int((pick&correct_on).sum()),correct_control=int((pick&correct_off).sum()),
            rescued=int((pick&correct_on&~correct_off).sum()),damaged=int((pick&correct_off&~correct_on).sum()),
            wrong_to_other_wrong=int((pick&~correct_on&~correct_off&(on!=off)).sum()),
            true_margin_mean_on=float((on_scores[pick,i]-on_scores[pick][:,other].max(1)).mean()),
            true_margin_mean_control=float((off_scores[pick,i]-off_scores[pick][:,other].max(1)).mean())))
    return dict(classes=counts,confusion=confusions,predictions_exported=False)


def evaluate(c):
    validate_job(c);check_barrier(c)  # Before opening val.csv.
    output=Path(c['output']);started=time.monotonic()
    seed=c['seed'];torch.set_num_threads(4)
    def budget():
        if time.time()>=c['original_deadline']:raise RuntimeError('INCOMPLETE_ORIGINAL_DEADLINE')
    train,val=manifests(c)
    from run_medical_v2 import transform
    encoder=ApartFeatures(c['legacy_repo'],c['weight'],len(c['order']),'cuda:0',seed)
    tasks=task_blocks(c['order'],c['task_sizes']);reports=[]
    ranked=sorted(c['order'],key=lambda x:(-sum(r['label']==x for r in train),x));tail=set(ranked[len(ranked)//2:])
    historical=json.loads(Path(c['control_metrics']).read_text())
    for task in range(1,len(tasks)+1):
        budget();st=torch.load(output/f'stage_{task}.pt',map_location='cpu',weights_only=False)
        seen=c['order'][:sum(c['task_sizes'][:task])]
        if st['pmt_metadata']!=checkpoint_metadata(c,task,sum(STEPS[c['name']][:task])) or st['seen']!=seen:
            raise ValueError('PMT checkpoint/source/order/budget mismatch')
        bank={key:v.cuda() if torch.is_tensor(v) else v for key,v in st['bank'].items()}
        bank['components']=[{key:v.cuda() if torch.is_tensor(v) else v for key,v in row.items()} for row in bank['components']]
        local=None if st['module_memory'] is None else st['module_memory'].double().cuda()
        R,_=modules.metric(bank,c['module'],c['name'],seen,local)
        if st['bank'].get('_frame')!=f'task_{task}_sealed' or st['pmt_frame']!=st['bank']['_frame']:
            raise ValueError('Checkpoint coordinate frame differs')
        rebuilt,audit=competition.bank_head(bank,st['boundary_pair_weights'].double().cuda(),regularizer=R)
        saved=st['pmt_head_double'].cuda()
        error=float((rebuilt-saved).norm()/saved.norm().clamp_min(1e-30))
        if error>1e-8 or not torch.equal(st['pmt_head_double'].float(),st['head']):raise ValueError('PMT head checkpoint rebuild differs')
        rows=[r for r in val if r['label'] in seen]
        loader=DataLoader(Images(rows,c['images'],transform(False),seed+task*100003),batch_size=64,num_workers=0,shuffle=False)
        encoder.load_state_dict(st['model'],strict=True);encoder.eval()
        x,y=extract(encoder,loader,budget);scores=x@st['head'].numpy();on=scores.argmax(1)
        del x,bank,local,R,rebuilt,saved
        control=torch.load(Path(c['control_checkpoint_root'])/f'stage_{task}.pt',map_location='cpu',weights_only=False)
        if control['seen']!=seen or control['steps']!=sum(STEPS[c['name']][:task]) or control['module']!=c['module']:raise ValueError('Frozen control checkpoint differs')
        encoder.load_state_dict(control['model'],strict=True);encoder.eval()
        x,y0=extract(encoder,loader,budget)
        if not np.array_equal(y,y0):raise ValueError('Paired canonical val order changed')
        control_scores=x@control['head'].numpy();off=control_scores.argmax(1)
        recalls={str(cl):float((on[y==cl]==i).mean()) for i,cl in enumerate(seen)}
        control_recalls={str(cl):float((off[y==cl]==i).mean()) for i,cl in enumerate(seen)}
        if control_recalls!=historical['stages'][task-1]['per_class_recall']:raise ValueError('Paired control inference differs from published stage recalls')
        count={str(cl):int((y==cl).sum()) for cl in seen}
        paired_values=paired(y,seen,on,off,scores,control_scores,len(seen)-len(tasks[task-1]))
        cm=np.asarray(paired_values['confusion']['ON']);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(len(seen)),where=den>0)
        reports.append(dict(task=task,seen=seen,validation_n=len(y),accuracy=float((np.asarray(seen)[on]==y).mean()),
            balanced_accuracy=float(np.mean(list(recalls.values()))),per_class_recall=recalls,per_class_n=count,
            tail_recall=float(np.mean([recalls[str(cl)] for cl in seen if cl in tail])),macroF1=float(f1.mean()),
            paired=paired_values,checkpoint_rebuild_relative_error=error,solve_residual=audit['relative_residual']))
        del x,st,control,scores,control_scores
    old=c['order'][:-c['task_sizes'][-1]];new=c['order'][-c['task_sizes'][-1]:]
    forget=[max(r['per_class_recall'][str(cl)] for r in reports[:-1] if cl in r['seen'])-reports[-1]['per_class_recall'][str(cl)] for cl in old]
    metrics=dict(stages=reports,**benchmark_metrics(reports,tasks,{cl:sum(r['label']==cl for r in train) for cl in c['order']}),
        average_incremental_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reports])),final_balanced_accuracy=reports[-1]['balanced_accuracy'],
        final_tail_recall=reports[-1]['tail_recall'],forgetting=float(np.mean(forget)),
        final_old_recall=float(np.mean([reports[-1]['per_class_recall'][str(cl)] for cl in old])),
        final_new_recall=float(np.mean([reports[-1]['per_class_recall'][str(cl)] for cl in new])),final_macro_F1=reports[-1]['macroF1'],average_incremental_macro_F1=float(np.mean([r['macroF1'] for r in reports])),
        test_accessed=False,evaluation_split='development_validation',independent_confirmation=False,source_commit=c['source_commit'],
        pmt_version=pmt.VERSION,module=label(c['module'],c['pmt_mode']),checkpoint_head_rebuild_verified=True,
        paired_control_inference_recall_verified=True)
    save(output/'metrics.json',metrics)
    status=json.loads((output/'STATUS.json').read_text());status.update(status='COMPLETE',evaluation_seconds=time.monotonic()-started);save(output/'STATUS.json',status)


if __name__=='__main__':
    c=json.load(sys.stdin);validate_job(c)
    if c.get('operation')=='evaluate':evaluate(c)
    elif c.get('operation')=='train':training.run(c)
    else:raise ValueError('No other operation authorized')
