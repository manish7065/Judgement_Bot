import os 
from pathlib import Path

ROOT = Path(__file__).resolve()

print(ROOT)
print(ROOT.parents[1])
print(ROOT.parents[0])

