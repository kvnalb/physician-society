"""Load project-root `.env` for local runs (file is gitignored)."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_local_dotenv(*, override: bool = False) -> None:
    """
    Load ``<repo>/.env`` if present. Does not override variables already set
    in the process environment unless ``override=True``.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    path = _PROJECT_ROOT / ".env"
    if path.is_file():
        load_dotenv(path, override=override)
