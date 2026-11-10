from pathlib import Path
from .core import fixtures,interface_fixtures,contrast_fixtures,write

def run(cfg):write(Path(cfg['output'])/'SOFTWARE_CHECK.json',{**fixtures(),**interface_fixtures(),**contrast_fixtures()})
