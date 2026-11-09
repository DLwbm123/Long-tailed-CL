from pathlib import Path
from .core import fixtures,write,interface_fixtures


def run(cfg):
    proof={**fixtures(),**interface_fixtures()};write(Path(cfg['output'])/'SOFTWARE_CHECK.json',proof)
    return proof
