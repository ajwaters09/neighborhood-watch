"""Puts the repo root on sys.path whenever anything under `webapp` is imported, so the app imports
`src.*` the same way the notebooks do, whatever the process's working directory.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
