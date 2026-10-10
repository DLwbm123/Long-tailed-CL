"""CPU checks of successful adoption, failure propagation, and config equality."""
import json
from pathlib import Path
import tempfile
import time

from module_dispatch_handoff import wait_external


def check():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);output=root/'runs'/'candidate';output.mkdir(parents=True)
        job=dict(output=str(output),expected_steps=276,original_deadline=time.time()+60)
        (root/'DISPATCH_HANDOFF.private.json').write_text(json.dumps(dict(job=job)))
        (output/'STATUS.json').write_text(json.dumps(dict(status='TRAINED',steps=276,policy_updates=0)))
        state=root/'EXTERNAL_TRAIN.private.json'
        state.write_text(json.dumps(dict(status='COMPLETE',cost=dict(returncode=0))))
        assert wait_external(job)==0
        state.write_text(json.dumps(dict(status='INCOMPLETE',cost=dict(returncode=7))))
        assert wait_external(job)==7
        try:wait_external(dict(job,expected_steps=277))
        except ValueError:pass
        else:raise AssertionError('Changed training budget accepted')
        state.write_text(json.dumps(dict(status='COMPLETE',cost=dict(returncode=0))))
        (output/'STATUS.json').write_text(json.dumps(dict(status='TRAINED',steps=275,policy_updates=0)))
        try:wait_external(job)
        except ValueError:pass
        else:raise AssertionError('Incomplete external training accepted')
    print('PASS: unchanged config, exact training barrier, failure returncode, no duplicate training')


if __name__=='__main__':check()
