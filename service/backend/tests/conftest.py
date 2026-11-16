"""Make `import app...` work from any CWD and force mock mode (no model) for the tests."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # service/backend/
os.environ["MOCK_MODE"] = "1"                                 # must be set before app.main is imported
