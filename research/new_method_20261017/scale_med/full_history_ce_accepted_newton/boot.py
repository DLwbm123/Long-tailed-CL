"""Receipts survive Python exceptions; the coordinator meters imports and fatal exits."""
import time
START=time.monotonic()
import importlib.util,json,os,resource,sys,traceback
from pathlib import Path
from environment import ENV_KEYS
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['root']);phase=cfg['mode'];out=root/('engineering/precheck' if phase=='PREFLIGHT' else 'run');out.mkdir(exist_ok=True)
resource.setrlimit(resource.RLIMIT_CPU,(int(cfg['cpu_budget']),int(cfg['cpu_budget'])))
def write(name,value):
    p=out/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2,allow_nan=False));q.replace(p)
exit_code=1;module=None;current='IMPORT';env={k:os.environ.get(k) for k in ENV_KEYS};env.update(executable=sys.executable,cwd=str(Path.cwd()))
try:
    write('PROGRESS.json',dict(phase=current,real_data_access=False))
    spec=importlib.util.spec_from_file_location('worker',str(root/'run.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.torch.set_num_threads(4)
    env.update(torch_version=module.torch.__version__,threads=module.torch.get_num_threads());write('ENVIRONMENT.private.json',env)
    if phase=='RUN':
        current='ENVIRONMENT_CHECK';matches=env==json.loads((root/'engineering/precheck/ENVIRONMENT.private.json').read_text())
        write('ENVIRONMENT_CHECK.json',dict(matches_precheck=matches,checked_keys=list(env),torch_version=module.torch.__version__,threads=module.torch.get_num_threads()))
        assert matches,'precheck/run environment mismatch'
    current='OPTIMIZER_INITIALIZATION';write('PROGRESS.json',dict(phase=current,real_data_access=False));tick=time.monotonic()
    dummy=module.torch.zeros((1,1),dtype=module.torch.float64,requires_grad=True)
    optimizer=module.torch.optim.LBFGS([dummy],lr=1,max_iter=2000,max_eval=3000,tolerance_grad=1e-9,tolerance_change=1e-14,history_size=20,line_search_fn='strong_wolfe')
    del optimizer,dummy
    write('INITIALIZATION.json',dict(wall_seconds=time.monotonic()-tick,constructor_only_no_optimizer_step=True,inside_global_budget=True,torch_version=module.torch.__version__))
    current='SYNTHETIC_SELFCHECK';write('PROGRESS.json',dict(phase=current,real_data_access=False))
    if phase=='PREFLIGHT':module.selfcheck(out/'SELFCHECK.json')
    else:
        current='SCIENTIFIC_PIPELINE';write('PROGRESS.json',dict(phase=current,real_data_access='guarded_by_worker'));module.main(cfg)
    exit_code=0
except BaseException as error:
    write('FAILURE.json',dict(error=type(error).__name__,phase=current,decision='UNRESOLVED_ENGINEERING_NO_EXTENSION',detailed_selfcheck_receipt=(out/'SELFCHECK.json').exists(),detailed_fit_receipts=[p.name for p in out.glob('*FIT*.json')],no_retry=True))
    write('FAILURE.private.json',dict(traceback=traceback.format_exc()))
finally:
    usage=resource.getrusage(resource.RUSAGE_SELF)
    gpu=bool(module and module.torch.cuda.is_initialized())
    write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.monotonic()-START,cpu_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024,GPU_initialized=gpu,peak_GPU_allocated_bytes=module.torch.cuda.max_memory_allocated() if gpu else 0,phase=current,scope='all_worker_imports_initialization_and_pipeline'))
    write('EXIT.json',dict(exit_code=exit_code,wall_seconds=time.monotonic()-START,phase=current))
sys.exit(exit_code)
