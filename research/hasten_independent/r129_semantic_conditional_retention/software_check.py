from pathlib import Path
from .core import fixtures,interface_fixtures,semantic_fixtures,write

def run(cfg):write(Path(cfg['output'])/'SOFTWARE_CHECK.json',{'base':fixtures(),'interface':interface_fixtures(),'semantic':semantic_fixtures(),'new_model_calls':0})
