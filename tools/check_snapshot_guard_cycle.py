"""CPU-only orchestration self-check; no images, checkpoints, CUDA or optimizer."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import run_snapshot_guard_cycle as cycle


def main():
    root=Path(os.environ['Q251_ROOT']);c=json.loads((root/'CONFIG.private.json').read_text())
    old=json.loads((root/'ACCEPTED_CONFIG.private.json').read_text())
    result=json.loads((root/'ACCEPTED_PREFLIGHT.json').read_text());reuse=json.loads((root/'SOURCE_REUSE.json').read_text())
    cycle.validate_acceptance(c,old,result,reuse)
    rejected=0
    def rejects(fn):
        nonlocal rejected
        try:fn()
        except (ValueError,FileNotFoundError):rejected+=1;return
        raise AssertionError('Invalid scope admitted')
    for mut in (lambda x:x.update(gpus=[2,3]),lambda x:x.update(formal_training_authorized=False),
                lambda x:x['jobs'][0].update(lr=.0004),lambda x:x['jobs'][0].update(weight='different'),
                lambda x:x['jobs'][0].update(guard_solver='different')):
        bad=copy.deepcopy(c);mut(bad);rejects(lambda:cycle.validate_acceptance(bad,old,result,reuse))
    for key,value in [('parameters_exactly_unchanged',False),('optimizer_updates',1),('validation_accessed',True),('source_commit','bad')]:
        bad=copy.deepcopy(result);bad['preflights'][0]['result'][key]=value;rejects(lambda:cycle.validate_acceptance(c,old,bad,reuse))
    bad=copy.deepcopy(result);bad['preflights'][0]['result']['guard_highest_bank']['minimum_raw_slack']=-1e-3
    rejects(lambda:cycle.validate_acceptance(c,old,bad,reuse))
    calls=[]
    with tempfile.TemporaryDirectory(dir=root/'tmp') as td:
        tc=copy.deepcopy(c);tc.update(root=td,launched_at=time.time(),deadline=time.time()+60)
        Path(td,'logs').mkdir()
        for j in tc['jobs']:j['output']=str(Path(td,j['job_name']))
        rejects(lambda:cycle.training_barrier(tc))
        class MockProcess:
            pid=999999
            returncode=0
            def __init__(self,args,**kw):
                assert args==[cycle.sys.executable,'-u',tc['worker_entry']]
                assert kw['env']['CUDA_VISIBLE_DEVICES'] in ('0','1')
            def communicate(self,data=None,timeout=None):
                j=json.loads(data);out=Path(j['output']);calls.append((j['job_name'],j['operation']))
                if j['operation']=='train':
                    out.mkdir()
                    cycle.save(out/'STATUS.json',dict(status='TRAINED',steps=j['expected_steps'],stages=len(j['task_sizes']),policy_updates=0))
                    for filename,fn in [('GUARD_TRAINING_SEAL.json',cycle.metadata),('TRAINING_SEAL.json',cycle.checkpoint_metadata)]:
                        cycle.save(out/filename,[fn(j,t,sum(cycle.STEPS[j['name']][:t])) for t in range(1,len(j['task_sizes'])+1)])
                else:
                    assert j['operation']=='evaluate'
                    p=json.loads(Path(j['evaluation_gate']).read_text());assert p['training_sealed_at'] and p['phase']=='EVALUATE'
                    assert len([x for x in calls if x[1]=='train'])==4
                    s=json.loads((out/'STATUS.json').read_text());s['status']='COMPLETE';cycle.save(out/'STATUS.json',s)
                return None,None
        with patch.object(cycle.subprocess,'Popen',MockProcess):cycle.run(tc)
        state=json.loads(Path(td,'PROGRAM_STATE.json').read_text())
        assert state['status']=='COMPLETE' and not state['active'] and not state['failures'] and len(state['costs'])==8
        assert len(calls)==8 and {op for _,op in calls}=={'train','evaluate'}
        # Seal/source drift must be rejected before evaluation.
        for j in tc['jobs']:
            p=Path(j['output'],'STATUS.json');s=json.loads(p.read_text());s['status']='TRAINED';cycle.save(p,s)
        cycle.training_barrier(tc)
        p=Path(tc['jobs'][0]['output'],'GUARD_TRAINING_SEAL.json');v=json.loads(p.read_text());v[0]['source_commit']='bad';cycle.save(p,v)
        rejects(lambda:cycle.training_barrier(tc))
    with tempfile.TemporaryDirectory(dir=root/'tmp') as td:
        tc['root']=td;Path(td,'logs').mkdir();calls.clear()
        class FailedProcess(MockProcess):
            returncode=7
            def communicate(self,data=None,timeout=None):
                j=json.loads(data);calls.append((j['job_name'],j['operation']));return None,None
        with patch.object(cycle.subprocess,'Popen',FailedProcess):cycle.run(tc)
        s=json.loads(Path(td,'PROGRAM_STATE.json').read_text())
        assert s['status']=='INCOMPLETE' and s['failures'] and not s['active'] and all(op=='train' for _,op in calls)
        assert len(s['costs'])==len(calls) and len(calls)<=2
    print(json.dumps(dict(status='PASS_CPU_ORCHESTRATION_ONLY',negative_scope_and_seal_checks=rejected,
        mocked_successful_operations=8,failure_stops_dependencies=True,no_preflight_rerun=True,
        GPU_operations=0,medical_images=0,medical_checkpoints=0,optimizer_updates=0,val_test_access=0)))

if __name__=='__main__':main()
