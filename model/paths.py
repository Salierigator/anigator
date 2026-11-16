"""Local folders the model code reads and writes, resolved from the repo root.

Every architecture model/<name>/ owns train/<name>/ (training data, checkpoints) and
export/<name>/ (the bundle its serve.py reads).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def train_dir(name: str) -> Path:
    return ROOT / "train" / name


def export_dir(name: str) -> Path:
    return ROOT / "export" / name
