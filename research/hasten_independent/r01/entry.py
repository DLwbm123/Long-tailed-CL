"""Neutral entry; private paths arrive through the environment only."""
import os,sys,runpy
sys.path.insert(0,os.environ['RUN_SOURCE'])
runpy.run_path(os.path.join(os.environ['RUN_SOURCE'],'runner.py'),run_name='__main__')
