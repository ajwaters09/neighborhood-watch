"""Shared pytest setup: the repo root on sys.path, so tests import `src.*` the way the app and
notebooks do, and a stand-in `src.db_connect` where databricks-sdk isn't installed (nothing under
test connects to anything).

    python -m pip install -r requirements-dev.txt
    python -m pytest
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from src import db_connect  # noqa: F401
except ImportError:
    sys.modules["src.db_connect"] = types.SimpleNamespace(
        UC_CATALOG="c", UC_SCHEMA="s", UC_CRIMES="c.s.x", UC_311="c.s.y")
