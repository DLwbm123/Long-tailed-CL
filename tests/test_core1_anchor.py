import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from core1_anchor import self_check

def test_historical_competition_anchor():
    self_check()
