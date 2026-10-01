"""Coordinator import must reject missing campaign authority before launching children."""
import json
import os
from pathlib import Path
import runpy
import tempfile
import time


def selfcheck():
    saved={k:os.environ.get(k) for k in ('N78_ROOT','N78_MODE')}
    try:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'private').mkdir();(root/'public').mkdir()
            os.environ.update(N78_ROOT=str(root),N78_MODE='probe')
            (root/'public/RESOURCE_LEDGER.json').write_text(json.dumps(dict(GPU_process_residence_seconds=0)))
            for authority,budget,allowed in [(False,7200,False),(True,None,False),(True,7200,True)]:
                cfg=dict(experiment='NB-RL-A4',campaign_budget_authorized=authority,wall_T0_unix=time.time())
                if budget is not None:cfg['gpu_budget_seconds']=budget
                (root/'private/INPUT.json').write_text(json.dumps(cfg))
                try:module=runpy.run_path(str(Path(__file__).with_name('nb_rl_a1_parallel.py')))
                except AssertionError as error:
                    assert not allowed and 'BLOCKED_MISSING_CAMPAIGN_BUDGET' in str(error)
                else:
                    assert allowed and module['GPU_LIMIT']==7200 and not module['PROCESS']
                    module['limits']()
                    module['BASE']['GPU_process_residence_seconds']=7141
                    try:module['limits']()
                    except AssertionError as error:assert 'INCOMPLETE_BUDGET' in str(error)
                    else:raise AssertionError('Cumulative residence cap was not enforced')
    finally:
        for k,v in saved.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    return dict(status='PASS',missing_authority_rejected=True,missing_budget_rejected=True,
                smaller_campaign_cap_enforced=True,no_children_launched=True)


if __name__=='__main__':print(json.dumps(selfcheck()))
