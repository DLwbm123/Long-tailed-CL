"""Run with PYTHONPATH=tools python tests/test_isic_g1_controlled_mixup.py."""
import tempfile
from pathlib import Path
from threadpoolctl import threadpool_limits
from isic_g1_controlled_mixup import engineering, ROOT

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as d, threadpool_limits(limits=4):
        print(engineering(Path(d),ROOT/'tools/frozen_medical_v3.py'))
