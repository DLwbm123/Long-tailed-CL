import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from core1_competition import self_check

def test_quadratic_competition():
    self_check()
