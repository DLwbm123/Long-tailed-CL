"""Independent CPU NumPy native-attention acceptance, without encoding/optimization."""
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from core import ARMS
from data import read_rows
from run import numpy_scores, write


def summarize(stages,cfg,counts):
    mean={a:float(np.mean([s['metrics'][a]['BA_percent'] for s in stages])) for a in ARMS}
    groups=dict(tail4=[7,14,15,16],tail7=[c for c,n in counts.items() if n<250],initial10=cfg['order'][:10],later10=cfg['order'][10:])
    freq={a:{g:sum(stages[-1]['metrics'][a]['correct'][str(c)] for c in cs) for g,cs in groups.items()} for a in ARMS}
    final=lambda a:stages[-1]['metrics'][a]['total_correct']
    a,m=ARMS
    gates=dict(final=final(m)>=91,mean=mean[m]>=36.005291005291,tail7=freq[m]['tail7']>=25,initial10=freq[m]['initial10']>=44,later10=freq[m]['later10']>=41,
               vs_frozen_final=final(m)>=final(a)+4,vs_frozen_mean=mean[m]>=mean[a]+1,
               vs_frozen_tail4=freq[m]['tail4']>=freq[a]['tail4']+2,vs_frozen_later=freq[m]['later10']>=freq[a]['later10'])
    return dict(stages=stages,mean_BA_percent=mean,frequency=freq),dict(gates=gates,proceed=all(gates.values()),no_parameter_rescue=True,
          scope='Native upper-layer latent replay adaptation; repeated development val, not clinical test or novelty proof.')


def main(cfg):
    from safetensors.torch import load_file
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4)
    tick=time.time();root=Path(cfg['output'])
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    seals=json.loads((root/'STAGE_SEALS.json').read_text());fit=json.loads((root/'FIT_SEAL.json').read_text());opened=json.loads((root/'QUERY_OPEN.private.json').read_text())
    assert len(seals)==6 and fit['models']==12 and all(s['at']<fit['at']<opened['at'] and s['validation_encodes']==0 for s in seals)
    query=torch.load(root/'query.private.pt',map_location='cpu',weights_only=False)
    assert query['labels'].tolist()==[r['label'] for r in val]
    assert query['tokens'].shape==(300,197,768) and query['tokens'].dtype==torch.float16
    original=load_file(cfg['weight']);lower=torch.load(root/'prefix_final.private.pt',map_location='cpu',weights_only=False)
    assert set(lower)=={k for k in original if not k.startswith(('head.','blocks.10.','blocks.11.','norm.'))}
    assert all(torch.equal(v,original[k]) for k,v in lower.items())
    initial=torch.load(root/'initial_upper.private.pt',map_location='cpu',weights_only=False)
    for k,v in initial.items():
        source='blocks.'+str(int(k.split('.')[1])+10)+'.'+'.'.join(k.split('.')[2:]) if k.startswith('blocks.') else k
        assert torch.equal(v,original[source])
    del original,lower
    previous={};error=0.;probe_error=0.;mismatches=0;rows_count=0;probe_rows=0;stages=[]
    for task in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        pack=torch.load(root/f'stage_{task}.private.pt',map_location='cpu',weights_only=False)
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])]
        assert pack['seen']==seals[task]['seen']==seen
        assert seals[task]['science_train_encodes']==sum(r['label'] in seen for r in train)
        assert all(n==cfg['base_updates']+task*cfg['later_updates'] for n in seals[task]['updates'].values())
        selected=[]
        for c in seen:
            patients={}
            for i,r in enumerate(train):
                if r['label']==c:patients.setdefault(r['group'],[]).append(i)
            def rank(g,path=''):return hashlib.sha256(f"{cfg['seed']}:{c}:{g}:{path}".encode()).digest()
            for g in sorted(patients,key=lambda g:(rank(g),g))[:cfg['memory_per_class']]:
                selected.append(min(patients[g],key=lambda i:(rank(g,train[i]['relative_path']),i)))
        assert sorted(r['id'] for r in pack['memory_rows'])==sorted(selected)
        assert len(pack['memory_rows'])==len({(r['label'],r['group']) for r in pack['memory_rows']})
        assert pack['memory_tokens'].shape==(len(selected),197,768) and pack['memory_tokens'].dtype==torch.float16
        for row,token in zip(pack['memory_rows'],pack['memory_tokens']):
            i=row['id'];assert row['relative_path']==train[i]['relative_path'] and row['group']==train[i]['group']
            if i in previous:assert torch.equal(previous[i],token)
            previous[i]=token.clone()
        mask=np.isin(query['labels'].numpy(),seen);labels=query['labels'].numpy()[mask]
        tokens=query['tokens'].numpy()[mask];metrics={}
        for a in ARMS:
            w=pack['models'][a]
            assert w['head'].shape==(len(seen),1536)
            same=all(torch.equal(v,w['upper.'+k]) for k,v in initial.items())
            assert same if a=='FROZEN_UPPER' else not same
            expected=torch.load(root/f'scores_{task}_{a}.private.pt',map_location='cpu',weights_only=False).numpy()
            result=[]
            for start in range(0,len(tokens),cfg['verification_batch']):
                assert time.time()<cfg['hard_deadline_epoch']-60
                result.append(numpy_scores(tokens[start:start+cfg['verification_batch']],w,cfg['layernorm_epsilon']))
            score=np.concatenate(result);rows_count+=len(score)
            error=max(error,float(abs(score-expected).max()))
            mismatches+=int((score.argmax(1)!=expected.argmax(1)).sum())
            pred=np.array(seen)[score.argmax(1)]
            correct={str(c):int(((pred==c)&(labels==c)).sum()) for c in seen}
            metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/len(labels))
            probe=pack['training_probe'];ref=numpy_scores(probe['tokens'].numpy(),w,cfg['layernorm_epsilon']);probe_rows+=len(ref)
            probe_error=max(probe_error,float(abs(ref-probe['scores'][a].numpy()).max()))
        stages.append(dict(task=task,seen=seen,metrics=metrics))
    assert error<=cfg['score_absolute_tolerance'] and probe_error<=cfg['score_absolute_tolerance'] and mismatches==0
    result,decision=summarize(stages,cfg,Counter(r['label'] for r in train))
    assert result==json.loads((root/'RESULTS.json').read_text()) and decision==json.loads((root/'DECISION.json').read_text())
    cost=json.loads((root/'COSTS.json').read_text())
    assert (cost['qualification_encodes'],cost['science_train_encodes'],cost['validation_encodes'])==(20,68058,300)
    assert cost['qualification_interface_full_encoder_rows']==cost['qualification_interface_prefix_rows']==2
    for a in ARMS:
        ac=cost['arms'][a];qc=cost['qualification_arms'][a]
        assert ac['updates']==4500 and ac['logical_draws']==ac['head_forward_rows']==ac['head_backward_rows']==ac['upper_forward_rows']==288000
        assert ac['upper_backward_rows']==(288000 if a=='PLASTIC_UPPER' else 0)
        assert ac['initialization_rows']==68058 and ac['validation_rows']==1350 and ac['FP64_probe_rows']==12
        assert qc['updates']==2 and qc['logical_draws']==qc['upper_forward_rows']==128
        assert qc['initialization_rows']==20 and qc['FP64_probe_rows']==qc['numpy_probe_rows']==2
    assert rows_count==2700 and probe_rows==24
    write(root/'VERIFICATION.json',dict(state='PASS',score_max_error=error,prediction_mismatches=mismatches,
          training_probe_error=probe_error,model_count=12,numpy_validation_model_rows=rows_count,numpy_probe_model_rows=probe_rows,
          memory_images=len(previous),memory_exactly_immutable=True,metadata_reselection=True,patient_overlap=0,
          prefix_exactly_unchanged=True,frozen_upper_exactly_unchanged=True,main_upper_changed=True,seals_and_counts=True,
          new_encodes=0,optimization_updates=0,internal_seconds=time.time()-tick))


if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
