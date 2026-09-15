"""Neutral-launcher entry points for bounded training, then a distinct evaluation process."""
import errno
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

def worker(config):
    import torch
    torch.set_num_threads(4)
    from run_medical_v2 import write_json
    output=Path(config['output'])
    try:
        if os.environ['Q8_STAGE']=='0':
            from execute_medical_v2 import train
            train(config)
        else:
            from evaluate_medical_v2 import evaluate
            result=evaluate(config);write_json(output/'FINAL_STATUS.json',result)
    except Exception as error:
        infrastructure=isinstance(error,OSError) and error.errno in (errno.EIO,errno.ESTALE,errno.ETIMEDOUT)
        write_json(output/'FAILURE.json',dict(status='INFRASTRUCTURE_INTERRUPTION' if infrastructure else 'BLOCKED_TECHNICAL',
                                             error=str(error),traceback=traceback.format_exc(),time=time.time(),stage=os.environ['Q8_STAGE']))
        traceback.print_exc()
        return 75 if infrastructure else 1
    return 0

def supervise(config,worker_entry):
    output=Path(config['output']);recovery=0
    for stage in ('0','1'):
        while True:
            env=dict(os.environ,Q8_STAGE=stage)
            with (output/('train.log' if stage=='0' else 'test_batch.log')).open('a') as log:
                process=subprocess.Popen([sys.executable,'-u',worker_entry],env=env,stdout=log,stderr=subprocess.STDOUT)
                (output/'worker.pid').write_text(str(process.pid))
                status=process.wait()
            if status==0:break
            # A partially read test set never triggers a new batch evaluation.
            if status==75 and stage=='0' and recovery<1 and time.time()<config['formal_started_at']+86400:
                recovery+=1
                (output/'recovery.json').write_text(json.dumps(dict(attempt=recovery,time=time.time(),same_config=True)))
                continue
            (output/'STOPPED.json').write_text(json.dumps(dict(stage=stage,exit_code=status,recoveries=recovery,time=time.time())))
            return status
    return 0
