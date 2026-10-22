"""Independent NumPy predictions, class risk, feasible arrivals and sealed-memory acceptance."""
import json,os,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch

def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def numpy_forward(x,s,scale):
    x=normalize(np.asarray(x,dtype=np.float64));h=x@s['down.weight'].T+s['down.bias']
    h=h/(1+np.exp(-h));z=normalize(x+h@s['up.weight'].T+s['up.bias'])
    return z,scale*z@normalize(s['head']).T+s['bias']
def ce(score,y):
    z=score-score.max(1,keepdims=True)
    return np.log(np.exp(z).sum(1))-z[np.arange(len(y)),y]
def numpy_risks(x,y,s,scale):
    values=ce(numpy_forward(x,s,scale)[1],y)
    return np.array([values[y==c].mean() for c in range(int(y.max())+1)]) if len(y) else np.array([])
def nstate(s):return {k:v.numpy().astype(np.float64) for k,v in s.items()}
def summarize(stages,cfg,counts):
    arms=cfg['arms'];main='FEASIBLE'
    mean={a:float(np.mean([s['metrics'][a]['BA_percent'] for s in stages])) for a in arms}
    groups=dict(tail4=[7,14,15,16],tail7=[c for c,n in counts.items() if n<250],initial10=cfg['order'][:10],later10=cfg['order'][10:])
    freq={a:{g:sum(stages[-1]['metrics'][a]['correct'][str(c)] for c in cs) for g,cs in groups.items()} for a in arms}
    final=lambda a:stages[-1]['metrics'][a]['total_correct']
    gates=dict(final=final(main)>=91,mean=mean[main]>=cfg['gates']['mean_min_percent'],tail7=freq[main]['tail7']>=25,initial10=freq[main]['initial10']>=44,later10=freq[main]['later10']>=41)
    for a in ['ER','PROJECT']:
        gates.update({f'vs_{a}_final':final(main)>=final(a)+4,f'vs_{a}_mean':mean[main]>=mean[a]+1,f'vs_{a}_tail4':freq[main]['tail4']>=freq[a]['tail4']+2,f'vs_{a}_later':freq[main]['later10']>=freq[a]['later10']})
    return dict(stages=stages,mean_BA_percent=mean,frequency=freq),dict(gates=gates,proceed=all(gates.values()),no_parameter_rescue=True,evidence_scope='Finite-memory class-CE feasible transitions; no unseen-patient risk guarantee, novelty claim or faithful GEM reproduction.')

def main(cfg):
    from data import read_rows
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);start=time.time();root=Path(cfg['output'])
    def stage_path(task):return (Path(cfg['resume_source']) if task<cfg['resume_task'] else root)/f'stage_{task}.private.pt'
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    seals=json.loads((root/'STAGE_SEALS.json').read_text());fit=json.loads((root/'FIT_SEAL.json').read_text());opened=json.loads((root/'QUERY_OPEN.private.json').read_text())
    assert len(seals)==6 and all(s['at']<fit['at']<opened['at'] and s['validation_encodes']==0 for s in seals)
    q=torch.load(root/'query.private.pt',map_location='cpu',weights_only=False)
    assert q['labels'].tolist()==[r['label'] for r in val]
    labels=q['labels'].numpy();error=0.;riskerror=0.;mismatches=0;previous={};stages=[];auditrows=0
    prior=None
    for t in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        p=torch.load(stage_path(t),map_location='cpu',weights_only=False)
        seen=cfg['order'][:sum(cfg['task_sizes'][:t+1])];assert p['seen']==seen==seals[t]['seen']
        assert seals[t]['science_train_encodes']==sum(r['label'] in seen for r in train)
        assert all(v==cfg['base_updates']+t*cfg['later_updates'] for v in seals[t]['opportunities'].values())
        selected=[]
        for c in seen:
            groups={}
            for i,r in enumerate(train):
                if r['label']==c:groups.setdefault(r['group'],[]).append(i)
            rank=lambda g,path='':hashlib.sha256(f"{cfg['seed']}:{c}:{g}:{path}".encode()).digest()
            for g in sorted(groups,key=lambda g:(rank(g),g))[:cfg['memory_per_class']]:
                selected.append(min(groups[g],key=lambda i:(rank(g,train[i]['relative_path']),i)))
        assert sorted(r['id'] for r in p['memory_rows'])==sorted(selected)
        assert len(p['memory_rows'])==len({(r['label'],r['group']) for r in p['memory_rows']})
        for row,vec in zip(p['memory_rows'],p['memory_features']):
            assert train[row['id']]['relative_path']==row['relative_path']
            if row['id'] in previous:assert np.array_equal(previous[row['id']],vec.numpy())
            previous[row['id']]=vec.numpy().copy()
        oldn=len(prior['memory_rows']) if prior else 0
        mx=p['memory_features'][:oldn].numpy();my=np.array([seen.index(r['label']) for r in p['memory_rows'][:oldn]])
        metrics={};mask=np.isin(labels,seen)
        for a in cfg['arms']:
            s=nstate(p['models'][a]);assert s['head'].shape==(len(seen),cfg['input_dimension'])
            if t:
                pre=numpy_risks(mx,my,nstate(prior['models'][a]),cfg['logit_scale'])
                admitted=numpy_risks(mx,my,nstate(p['arrival_models'][a]),cfg['logit_scale'])
                final=numpy_risks(mx,my,s,cfg['logit_scale']);ceiling=np.array(p['ceilings'][a])
                assert np.max(abs(ceiling-pre-cfg['stage_CE_allowance']))<cfg['risk_verification_tolerance']
                assert np.max(admitted-pre)<=cfg['expansion_CE_allowance']+cfg['risk_verification_tolerance']
                riskerror=max(riskerror,float(np.max(abs(final-np.array(p['final_old_risks'][a])))))
                if a=='FEASIBLE':assert np.all(final<=ceiling+cfg['risk_verification_tolerance'])
                auditrows+=3*oldn
            else:
                for k in s:assert torch.equal(p['models'][a][k],p['models']['ER'][k])
            probe=p['probe'];pr=ce(numpy_forward(probe['x'].numpy(),s,cfg['logit_scale'])[1],probe['y'].numpy()).mean()
            assert abs(pr-probe['CE'][a])<=cfg['score_absolute_tolerance']
            score=numpy_forward(q['features'].numpy()[mask],s,cfg['logit_scale'])[1]
            saved=torch.load(root/f'scores_{t}_{a}.private.pt',map_location='cpu',weights_only=True).numpy()
            error=max(error,float(abs(score-saved).max()));mismatches+=int((score.argmax(1)!=saved.argmax(1)).sum())
            pred=np.array(seen)[score.argmax(1)];correct={str(c):int(((pred==c)&(labels[mask]==c)).sum()) for c in seen}
            metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/int(mask.sum()))
        if t:
            tr=p['transition_probe'];a='FEASIBLE';bs=nstate(tr['before']);after=nstate(tr['after'])
            r=numpy_risks(mx,my,after,cfg['logit_scale'])
            assert np.all(r<=np.array(p['ceilings'][a])+cfg['risk_verification_tolerance'])
            bv=np.concatenate([bs[k].ravel() for k in tr['parameter_order']]);av=np.concatenate([after[k].ravel() for k in tr['parameter_order']])
            assert np.max(abs(av-(bv-cfg['learning_rate']*tr['factor']*tr['direction'].numpy())))<2e-6
            G=tr['constraint_gradients'].numpy().astype(np.float64);direction=tr['direction'].numpy().astype(np.float64)
            H=G/np.maximum(np.linalg.norm(G,axis=1,keepdims=True),1e-12)
            assert np.min(H@direction)>=-cfg['projection_direction_tolerance']*2
            auditrows+=oldn
        stages.append(dict(task=t,seen=seen,metrics=metrics));prior=p
    assert error<=cfg['score_absolute_tolerance'] and mismatches==0 and riskerror<=cfg['risk_verification_tolerance']
    result,decision=summarize(stages,cfg,Counter(r['label'] for r in train))
    assert result==json.loads((root/'RESULTS.json').read_text()) and decision==json.loads((root/'DECISION.json').read_text())
    cost=json.loads((root/'COSTS.json').read_text())
    assert cost['new_science_train_encodes']==6940 and cost['reused_sealed_stage_train_encodes']==61118
    assert cost['science_train_encodes']==68058 and cost['validation_encodes']==300 and cost['encoder_parameter_updates']==0
    for a in cfg['arms']:
        assert cost['arms'][a]['opportunities']==4500 and cost['arms'][a]['training_view_rows']==576000 and cost['validation_model_rows'][a]==1350
    receipt=dict(state='PASS',score_max_absolute_error=error,prediction_disagreements=mismatches,class_risk_max_error=riskerror,metadata_memory_selection=True,old_features_unchanged=True,patient_overlap=0,first_stage_states_identical=True,arrival_CE_allowance_verified=True,FEASIBLE_final_per_stage_ceiling_verified=True,transition_probe_verified=True,verification_memory_forward_rows=auditrows,new_encoding=0,optimization=0,seconds=time.time()-start,scope='All 18 saved scores, six stage risk ledgers and one saved real transition per later stage; intermediate steps checked online, not independently replayed.')
    (root/'VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
