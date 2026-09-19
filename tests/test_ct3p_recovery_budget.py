"""CPU-only recovery dispatch check; no real subprocess or training."""
import hashlib,importlib.util,json,os,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'tools/execute_ct3p_recovery.py'
spec=importlib.util.spec_from_file_location('driver',path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

def check(limit,authorization):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);pub=root/'output/public';pub.mkdir(parents=True);(root/'output/private').mkdir()
        def save(name,value):(pub/name).write_text(json.dumps(value))
        (root/'runtime_locked.json').write_text('{}')
        save('RECOVERY_LOCK_R2.json',dict(GPU_limit_seconds=limit,budget_authorization=authorization,recovery_driver_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        save('RECOVERY_ADMISSION_R2.json',dict(status='PASS'))
        save('RESOURCE_AND_ACCESS_LEDGER.json',dict(GPU_process_residence_seconds=20000 if limit is None else 6178,CPU_analytic_seconds=64))
        with patch.dict(os.environ,P22_ROOT=tmp),patch.object(driver.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
            driver.main()
        commands=[x.args[0] for x in run.call_args_list]
        assert len(commands)==4
        assert all((c[0]=='timeout')==(limit is not None) for c in commands[:3])
        assert commands[-1][:3]==['timeout','-k','10']
        receipts=json.loads((pub/'PHASE_RECEIPTS_R2.json').read_text())
        assert all((x['hard_timeout_seconds'] is None)==(limit is None) for x in receipts[:3])
        assert (pub/'PIPELINE_COMPLETE.json').exists()

if __name__=='__main__':
    check(None,'USER_APPROVED_CT3P_NO_GPU_HOUR_LIMIT')
    check(18000,'USER_APPROVED_CT3P_CUMULATIVE_5H')
    try:check(None,'UNAPPROVED')
    except AssertionError:pass
    else:raise AssertionError('unapproved removal accepted')
    print('PASS: explicit uncapped GPU dispatch, retained CPU cap, legacy bounded dispatch, authorization guard')
