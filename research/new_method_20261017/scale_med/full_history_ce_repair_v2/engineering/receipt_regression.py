import time,resource,json,importlib.util,os
from pathlib import Path
start=time.monotonic();resource.setrlimit(resource.RLIMIT_CPU,(120,120));root=Path(os.environ['ENGINEERING_ROOT']);out=root/'engineering/receipt_check';out.mkdir()
spec=importlib.util.spec_from_file_location('worker',str(root/'run.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.torch.set_num_threads(4)
original=m.fit
def forced_fit(x,y,p,k,lam,device,check,limit=60,receipt=None):return original(x,y,p,k,lam,device,check,-1,receipt)
m.fit=forced_fit
try:m.selfcheck(out/'SELFCHECK.json')
except AssertionError:pass
else:raise AssertionError('expected synthetic budget failure was not observed')
r=json.loads((out/'SELFCHECK.json').read_text());f=json.loads((out/'SELFCHECK_FIT.json').read_text());assert r['state']=='FAILED' and r['fit']['status']=='UNRESOLVED' and f['stop_reason']=='HEAD_BUDGET_LIMIT'
m.fit=original;x=m.torch.ones((3,2),dtype=m.torch.float64);y=m.torch.arange(3);p=m.torch.ones(3,dtype=m.torch.float64)/3
def raise_error():raise RuntimeError('synthetic injected exception')
try:m.fit(x,y,p,3,.01,'cpu',raise_error,10,out/'EXCEPTION_FIT.json')
except RuntimeError:pass
else:raise AssertionError('expected injected exception was not observed')
e=json.loads((out/'EXCEPTION_FIT.json').read_text());assert e['state']=='FAILED' and e['error']=='RuntimeError' and 'optimizer_trace' in e
usage=resource.getrusage(resource.RUSAGE_SELF);result=dict(forced_budget_failure_saved_before_assert=True,python_exception_fit_receipt_saved=True,real_data_access=False,wall_seconds=time.monotonic()-start,cpu_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024)
(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
