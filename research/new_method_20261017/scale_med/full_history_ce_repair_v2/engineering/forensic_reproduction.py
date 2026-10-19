import time,resource,json,os,importlib.util
from pathlib import Path
root=Path(os.environ['ENGINEERING_ROOT']);out=root/'engineering';start=time.monotonic()
resource.setrlimit(resource.RLIMIT_CPU,(180,180))
spec=importlib.util.spec_from_file_location('legacy',os.environ['LEGACY_SOURCE']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.torch.set_num_threads(4)
receipt=dict(phase='legacy_cold_reproduction',import_seconds=time.monotonic()-start,thread_count=m.torch.get_num_threads(),torch_version=m.torch.__version__,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),cold_NFS_bytecode_cache=True,real_data_access=False)
original_init=m.torch.optim.LBFGS.__init__
def traced_init(self,*args,**kwargs):
 t=time.monotonic();original_init(self,*args,**kwargs);receipt['LBFGS_constructor_seconds']=time.monotonic()-t
m.torch.optim.LBFGS.__init__=traced_init
original_fit=m.fit
def traced_fit(*args,**kwargs):
 w,info=original_fit(*args,**kwargs);receipt['synthetic_fit']=info;(out/'LEGACY_COLD.json').write_text(json.dumps(receipt,indent=2));return w,info
m.fit=traced_fit
try:receipt['checks']=m.selfcheck();receipt['passed']=True
except Exception as e:receipt.update(passed=False,error=type(e).__name__)
finally:
 receipt.update(wall_seconds=time.monotonic()-start,cpu_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 (out/'LEGACY_COLD.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='synthetic_fit'}));print(json.dumps({k:v for k,v in receipt.get('synthetic_fit',{}).items() if k!='optimizer_trace'}))
