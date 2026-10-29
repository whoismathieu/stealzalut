"""PyInstaller-aware path resolution.

When running from a PyInstaller one-file bundle, static assets are
extracted to a temporary directory whose path is stored in
``sys._MEIPASS``.  This module provides helpers that transparently
resolve asset paths in both development and packaged environments.
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_base_dir() -> Path:
    """Return the application base directory.

    * **Bundled mode** (PyInstaller): ``sys._MEIPASS`` — the temporary
      extraction directory.
    * **Development mode**: the ``stealzalut`` package root
      (``src/stealzalut/``).
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return Path(str(meipass))
    # In dev: this file is at src/stealzalut/utils/paths.py
    return Path(__file__).resolve().parent.parent


def get_asset_path(*parts: str) -> Path:
    """Resolve a path relative to the ``assets/`` directory.

    Args:
        *parts: Path segments under ``assets/`` (e.g. ``"icons"``, ``"logo.png"``).

    Returns:
        Absolute path to the requested asset.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    path = get_base_dir() / "assets" / Path(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Asset not found: {path}")
    return path


def get_styles_path(filename: str = "theme.qss") -> Path:
    """Resolve the path to a Qt stylesheet file.

    Args:
        filename: Stylesheet filename inside ``gui/styles/``.

    Returns:
        Absolute path to the stylesheet.
    """
    return get_base_dir() / "gui" / "styles" / filename

