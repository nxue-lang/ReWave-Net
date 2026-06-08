from __future__ import annotations

import sys
from pathlib import Path


def add_project_src_to_path() -> None:
    """Allow scripts to import the local src-layout package without installation."""
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
