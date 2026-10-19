"""One environment for forensic precheck and the scientific worker."""
import os
ENV_KEYS=['PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TMPDIR','PYTHONPYCACHEPREFIX','RUN_CONFIG']
def environment(root):
    env=os.environ.copy()
    env.update(PYTHONPATH='/tmp/q570:/tmp/q540:/tmp/q581:/tmp/q530',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TMPDIR=str(root/'tmp'),PYTHONPYCACHEPREFIX=str(root/'tmp/runtime_cache'),RUN_CONFIG=str(root/'config.private.json'))
    return env
