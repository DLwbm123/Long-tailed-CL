"""CPU-only synthetic protocol checks; no private inputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from next2h_heads import self_check
if __name__=='__main__':self_check()
