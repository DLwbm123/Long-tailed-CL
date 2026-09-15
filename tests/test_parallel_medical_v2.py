"""Real subprocess overlap/failure checks plus the sealed six-trajectory barrier."""
import importlib.util,json,os,sys,tempfile,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('parallel',Path(__file__).resolve().parents[1]/'tools/parallel_medical_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp);entry=out/'worker.py'
    entry.write_text('import time\ntime.sleep(.1)\n')
    config=dict(output=tmp,formal_started_at=time.time(),protocol_sha256='fixed',code_commit='fixed')
    events=m.run_queue(config,str(entry));active=set();peak=0;finished=[]
    for e in events:
        if e['event']=='start':active.add(e['pair']);peak=max(peak,len(active))
        else:active.remove(e['pair']);finished.append(e['pair'])
        assert len(active)<=2
    assert peak==2 and sorted(finished)==sorted(m.PAIRS)
    try:m.collect(config)
    except FileNotFoundError:pass
    else:raise AssertionError('early test release')
    assert not (out/'ALL_CHECKPOINTS_LOCK.json').exists()
    for pair in m.PAIRS:
        run=out/pair;run.mkdir();seed,branch=pair.split('_');entries=[]
        for task in range(3):
            cp=run/f'session{task}.pt';cp.touch();entries.append(dict(seed=int(seed),branch=branch,session=task,epoch=10,path=str(cp),sha256='fixture'))
        m.save(run/'TRAJECTORY_COMPLETE.json',dict(pair=pair,initial_hash=seed,s0_hash=seed,sessions=[],checkpoints=entries))
    m.collect(config);assert len(json.loads((out/'ALL_CHECKPOINTS_LOCK.json').read_text())['checkpoints'])==18
    (out/'ALL_CHECKPOINTS_LOCK.json').unlink()
    p=out/'1995_C/TRAJECTORY_COMPLETE.json';v=json.loads(p.read_text());v['s0_hash']='wrong';m.save(p,v)
    try:m.collect(config)
    except AssertionError:pass
    else:raise AssertionError('paired mismatch accepted')
    assert not (out/'ALL_CHECKPOINTS_LOCK.json').exists()
    entry.write_text("import os,time,sys\nif os.environ['Q8_PAIR']=='1993_B':sys.exit(1)\ntime.sleep(2)\n")
    try:m.run_queue(config,str(entry))
    except RuntimeError:pass
    else:raise AssertionError('failed worker accepted')
    e=json.loads((out/'parallel_scheduler.json').read_text())['events'];assert len([x for x in e if x['event']=='start'])==2
print('TWO_WORKER_QUEUE_AND_18_CHECKPOINT_BARRIER_PASS')
