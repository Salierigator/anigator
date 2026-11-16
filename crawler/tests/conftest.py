"""The crawler is a folder of flat scripts, not a package: put it on sys.path."""
import sys
from pathlib import Path

CRAWLER = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CRAWLER), str(CRAWLER / "filter")]
