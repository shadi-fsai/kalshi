"""Byte-compile the entrypoint and page scripts.

The page scripts call Streamlit at module top level, so they can't simply be
imported outside a script run. ``py_compile`` parses and compiles them (catching
syntax/indentation errors) without executing them. The behavioral coverage for
the pages lives in ``test_app_smoke.py`` via Streamlit's AppTest.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    ROOT / "app.py",
    ROOT / "app_pages" / "find.py",
    ROOT / "app_pages" / "watch.py",
    ROOT / "app_pages" / "portfolio.py",
]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(script: Path):
    assert script.exists(), f"missing script: {script}"
    py_compile.compile(str(script), doraise=True)
