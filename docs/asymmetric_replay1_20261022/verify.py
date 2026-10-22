"""Independent CPU NumPy model/objective, metadata admission, sampling and cost acceptance."""
import hashlib
import json
import os
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def numpy_forward(x,state,scale):
    x=normalize(np.asarray(x,dtype=np.float64));h=x@state['down.weight'].T+state['down.bias']
    h=h/(1+np.exp(-h));z=normalize(x+h@state['up.weight'].T+state['up.bias'])
    return z,scale*z@normalize(state['head']).T


def cross_entropy(score,y):
    v=score-score.max(1,keepdims=True)
    return np.log(np.exp(v).sum(1))-v[np.arange(len(y)),y]


def numpy_loss(x,y,incoming_rows,old_count,state,arm,cfg):
    raw=numpy_forward(x,state,cfg['logit_scale'])[1];k=raw.shape[1]
    current=raw[:incoming_rows];cy=y[:incoming_rows]
    if arm=='ASYMMETRIC':current=current[:,old_count:];cy=cy-old_count
    else:assert arm=='ER'
    acquire=cross_entropy(current,cy).mean()
    weights=np.where(y[incoming_rows:]>=old_count,.5,1.)
    return float((k-old_count)/(2*k)*acquire+np.mean(weights*cross_entropy(raw[incoming_rows:],y[incoming_rows:])))


def summarize(stages,cfg,counts):
    arms=cfg['arms'];mean={a:float(np.mean([s['metrics'][a]['BA_percent'] for s in stages])) for a in arms}
    groups=dict(tail4=[7,14,15,16],tail7=[c for c,n in counts.items() if n<250],initial10=cfg['order'][:10],later10=cfg['order'][10:])
    frequency={a:{g:sum(stages[-1]['metrics'][a]['correct'][str(c)] for c in cs) for g,cs in groups.items()} for a in arms}
    final=lambda a:stages[-1]['metrics'][a]['total_correct'];m='ASYMMETRIC';c='ER';g=cfg['gates']
    gates=dict(final=final(m)>=g['final_min_correct'],mean=mean[m]>=g['mean_min_percent'],tail7=frequency[m]['tail7']>=g['tail7_min'],initial10=frequency[m]['initial10']>=g['initial10_min'],later10=frequency[m]['later10']>=g['later10_min'],vs_ER_final=final(m)>=final(c)+4,vs_ER_mean=mean[m]>=mean[c]+1,vs_ER_tail4=frequency[m]['tail4']>=frequency[c]['tail4']+2,vs_ER_later=frequency[m]['later10']>=frequency[c]['later10'])
    return dict(stages=stages,mean_BA_percent=mean,frequency=frequency),dict(gates=gates,proceed=all(gates.values()),no_parameter_rescue=True,evidence_scope='Stage-adapted asymmetric acquisition/all-class-memory consolidation versus matched class-balanced ER. Not a faithful online ER-ACE reproduction or novelty/clinical efficacy claim.')


def main(cfg):
    from threadpoolctl import threadpool_limits
    from data import read_rows
    threadpool_limits(4);torch.set_num_threads(4);start=time.time();root=Path(cfg['output'])
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    seals=json.loads((root/'STAGE_SEALS.json').read_text());admit=json.loads((root/'MEMORY_ADMISSIONS.json').read_text());audits=json.loads((root/'SAMPLING_AUDIT.json').read_text())
    fit=json.loads((root/'FIT_SEAL.json').read_text());opened=json.loads((root/'QUERY_OPEN.private.json').read_text())
    assert len(seals)==len(admit)==len(audits)==6 and fit['models']==12
    assert all(s['at']<fit['at']<opened['at'] and s['validation_encodes']==0 for s in seals)
    query=torch.load(root/'query.private.pt',map_location='cpu',weights_only=False);labels=query['labels'].numpy()
    assert labels.tolist()==[r['label'] for r in val]
    error=0.;mismatch=0;probe_error=0.;stages=[];previous={};probe_rows=0;score_rows=0;sample_current=0;sample_replay=0
    for task in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])];old=sum(cfg['task_sizes'][:task]);new=seen[old:]
        pack=torch.load(root/f'stage_{task}.private.pt',map_location='cpu',weights_only=False)
        assert pack['seen']==seals[task]['seen']==admit[task]['seen']==seen
        assert admit[task]['at']<audits[task]['first_update_at']<seals[task]['at']
        assert admit[task]['validation_encodes']==0 and admit[task]['rows']==seals[task]['memory_rows']==len(pack['memory_rows'])
        expected=cfg['base_updates']+task*cfg['later_updates'];assert all(v==expected for v in seals[task]['updates'].values())
        assert seals[task]['science_train_encodes']==sum(r['label'] in seen for r in train)
        selected=[]
        for c in seen:
            pp={}
            for i,r in enumerate(train):
                if r['label']==c:pp.setdefault(r['group'],[]).append(i)
            def rank(g,path=''):return hashlib.sha256(f"{cfg['seed']}:{c}:{g}:{path}".encode()).digest()
            for g in sorted(pp,key=lambda g:(rank(g),g))[:cfg['memory_per_class']]:selected.append(min(pp[g],key=lambda i:(rank(g,train[i]['relative_path']),i)))
        rows=pack['memory_rows'];assert sorted(r['id'] for r in rows)==sorted(selected)
        assert set(r['label'] for r in rows)==set(seen) and len(rows)==len({(r['label'],r['group']) for r in rows})
        for r,vector in zip(rows,pack['memory_features']):
            i=r['id'];assert r['relative_path']==train[i]['relative_path'] and r['label']==train[i]['label'] and r['group']==train[i]['group']
            if i in previous:assert np.array_equal(previous[i],vector.numpy())
            previous[i]=vector.numpy().copy()
        steps=cfg['base_updates'] if task==0 else cfg['later_updates'];audit=audits[task]
        assert audit['paired_same_draws'] and audit['paired_same_views']
        assert sum(audit['current_counts'].values())==steps*cfg['current_draws']
        assert sum(audit['replay_counts'].values())==steps*cfg['replay_draws']
        assert all(n==0 for c,n in audit['current_counts'].items() if int(c) not in new)
        sample_current+=sum(audit['current_counts'].values());sample_replay+=sum(audit['replay_counts'].values())
        p=pack['training_probe'];assert p['incoming_rows']==cfg['current_draws']*cfg['views_per_draw'] and p['old_count']==old
        assert len(p['ids'])==len(p['y'])==128
        for ix,(i,y) in enumerate(zip(p['ids'],p['y'].tolist())):
            assert train[i]['label']==seen[y]
            if ix<p['incoming_rows']:assert train[i]['label'] in new
            else:assert i in selected
        mask=np.isin(labels,seen);metrics={}
        for a in cfg['arms']:
            s={k:v.numpy().astype(np.float64) for k,v in pack['models'][a].items()};assert s['head'].shape==(len(seen),cfg['input_dimension'])
            if task==0:assert all(torch.equal(pack['models'][a][k],pack['models']['ER'][k]) for k in pack['models'][a])
            lv=numpy_loss(p['x'].numpy(),p['y'].numpy(),p['incoming_rows'],old,s,a,cfg);probe_error=max(probe_error,abs(lv-p['losses'][a]));probe_rows+=len(p['y'])
            score=numpy_forward(query['features'].numpy()[mask],s,cfg['logit_scale'])[1]
            saved=torch.load(root/f'scores_{task}_{a}.private.pt',map_location='cpu',weights_only=True).numpy()
            error=max(error,float(abs(score-saved).max()));mismatch+=int(sum(score.argmax(1)!=saved.argmax(1)));score_rows+=len(score)
            pred=np.array(seen)[score.argmax(1)];correct={str(c):int(((pred==c)&(labels[mask]==c)).sum()) for c in seen}
            metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/int(mask.sum()))
        stages.append(dict(task=task,seen=seen,metrics=metrics))
    assert error<=cfg['score_absolute_tolerance'] and mismatch==0 and probe_error<=cfg['score_absolute_tolerance']
    cost=json.loads((root/'COSTS.json').read_text());qual=json.loads((root/'QUALIFICATION.json').read_text())
    assert qual['state']=='PASS' and cost['science_train_encodes']==68058 and cost['qualification_encodes']==20 and cost['validation_encodes']==300 and cost['qualification_updates']==4
    total=cfg['base_updates']+5*cfg['later_updates']
    assert sample_current==sample_replay==total*32
    for a in cfg['arms']:
        assert cost['updates'][a]==total and cost['sampled_image_rows'][a]==total*64 and cost['training_view_rows'][a]==total*128
        assert cost['incoming_view_rows'][a]==cost['replay_view_rows'][a]==total*64
        assert cost['validation_model_rows'][a]==1350 and cost['training_probe_model_rows'][a]==768
        assert cost['qualification_model_rows'][a]==256 and cost['qualification_FP64_rows'][a]==cost['qualification_numpy_rows'][a]==256
    assert cost['encoder_parameter_updates']==0 and cost['head_initialization_feature_rows']==68058 and cost['memory_rows_final']==615
    resources=json.loads((root.parent/'guardian_resources.private.json').read_text())
    assert resources['peak_output_bytes']<cfg['output_budget_bytes'] and resources['RSS_sampled_peak_bytes']<cfg['RSS_budget_bytes']
    results,decision=summarize(stages,cfg,Counter(r['label'] for r in train))
    verdict=dict(state='PASS',score_groups=12,max_score_error=error,prediction_mismatches=mismatch,real_training_objective_max_error=probe_error,patient_overlap=0,memory_independent_reselection=True,old_features_immutable=True,current_memory_admitted_before_updates=True,current_positives_in_allseen_replay=True,first_stage_models_identical=True,paired_sampling_counts=True,probe_numpy_rows=probe_rows,score_numpy_rows=score_rows,new_encodes=0,new_optimization=0,wall_seconds=time.time()-start,scope='Independent endpoints and saved actual training probes; not independent replay of every optimization step.')
    for name,value in [('RESULTS.json',results),('DECISION.json',decision),('VERIFICATION.json',verdict)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    (root/'status.private.json').write_text(json.dumps(dict(state='COMPLETE_VERIFIED',method_gate_pass=decision['proceed'],delivery_pending=True,updated=time.time())))


if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
