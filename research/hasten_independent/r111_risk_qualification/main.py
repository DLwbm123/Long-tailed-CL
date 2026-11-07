import os,sys,runpy
from pathlib import Path
src=Path(os.environ["Q_SOURCE"]).resolve();sys.path.insert(0,str(src.parent));runpy.run_module(src.name+"."+os.environ["Q_MODULE"],run_name="__main__")
