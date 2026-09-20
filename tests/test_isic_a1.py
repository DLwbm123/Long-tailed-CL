"""CPU engineering check; no dataset, model, or GPU required."""
import tempfile
from pathlib import Path
from threadpoolctl import threadpool_limits
from isic_a1_attribution import engineering
from report_locked_holdout_r1 import selfcheck

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as d,threadpool_limits(limits=4):
        selfcheck()
        print(engineering(Path(d)))
