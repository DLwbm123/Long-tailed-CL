from pathlib import Path
from .core import fixtures,write


def run(cfg):
    proof=fixtures();write(Path(cfg['output'])/'SOFTWARE_CHECK.json',proof)
    return proof
