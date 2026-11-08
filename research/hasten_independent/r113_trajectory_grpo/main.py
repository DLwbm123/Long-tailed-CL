import os
import runpy
import sys
from pathlib import Path

source = Path(os.environ['Q_SOURCE']).resolve()
sys.path[:0] = [str(source / 'deps.private'), str(source.parent)]
runpy.run_module(source.name + '.' + os.environ['Q_MODULE'], run_name='__main__')
