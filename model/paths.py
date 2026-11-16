"""Local folders the model code reads and writes, resolved from the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRAIN = ROOT / "train"
EXPORT = ROOT / "export"
