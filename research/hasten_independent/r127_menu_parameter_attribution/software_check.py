from pathlib import Path
from .core import fixtures,write

def run(cfg):write(Path(cfg['output'])/'SOFTWARE_CHECK.json',fixtures())
