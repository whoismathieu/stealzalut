"""Themes: one stylesheet template, one small palette file per theme.

``gui/styles/theme.qss`` is written once, with tokens (``@bg``, ``@accent``,
``@radius``…) where a colour or a radius goes.  A theme is a JSON file that
gives those tokens a value::

    {"name": "Graphite", "tokens": {"bg": "#1b1b1c", "accent": "#5b8def", ...}}

Adding a theme is adding a file: in ``gui/styles/themes/`` (shipped with the
application) or in the user's own folder (:func:`user_theme_folder`), which
needs no rebuild.  A theme only has to list the tokens it changes: the others
come from :data:`DEFAULT_THEME`, so a token added to the template later never
breaks an older theme.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication

from stealzalut.utils.paths import get_styles_path

logger = logging.getLogger(__name__)

#: Theme every other one inherits its missing tokens from.
DEFAULT_THEME = "graphite"

_TOKEN = re.compile(r"@([a-z_]+)")
#: What a token may hold: a colour, ``transparent`` or a length in pixels.
_VALUE = re.compile(r"#[0-9a-fA-F]{6}|#[0-9a-fA-F]{8}|transparent|\d{1,3}px")

_ARROW = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
    '<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
_TICK = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
    '<path d="M2.5 6.2 5 8.6 9.5 3.6" fill="none" stroke="{colour}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


@dataclass(frozen=True, slots=True)
class Theme:
    """A named set of token values.

    Attributes:
        key: File name without extension; what preferences store.
        name: Name shown in the menu.
        tokens: Value of every token of the template.
    """

    key: str
    name: str
    tokens: dict[str, str]


def user_theme_folder() -> Path:
    """Folder where the user can drop theme files of their own."""
    location = QStandardPaths.StandardLocation.AppConfigLocation
    return Path(QStandardPaths.writableLocation(location)) / "themes"


def _read(path: Path) -> tuple[str, dict[str, str]]:
    """Name and tokens of one theme file; raises ``ValueError`` when it is not one."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(str(exc)) from exc
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(tokens, dict):
        raise ValueError('no "tokens" object')
    for token, value in tokens.items():
        if not isinstance(value, str) or not _VALUE.fullmatch(value):
            raise ValueError(f"token {token!r}: {value!r} is not a colour or a length in px")
    name = data.get("name")
    return (name if isinstance(name, str) and name else path.stem.capitalize()), tokens


def available_themes(folders: list[Path] | None = None) -> dict[str, Theme]:
    """Every usable theme, by key, the default one first.

    Args:
        folders: Where to look, in order; a later folder replaces a theme of
            the same key.  Defaults to the shipped folder then the user's.

    Returns:
        The themes, each completed with the default theme's tokens.  A file
        that cannot be read as a theme is logged and left out.

    Raises:
        ValueError: The default theme itself is missing or unreadable.
    """
    if folders is None:
        folders = [get_styles_path("themes"), user_theme_folder()]
    raw: dict[str, tuple[str, dict[str, str]]] = {}
    for folder in folders:
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                raw[path.stem] = _read(path)
            except ValueError as exc:
                logger.warning("Theme %s ignored: %s", path, exc)
    if DEFAULT_THEME not in raw:
        raise ValueError(f"default theme {DEFAULT_THEME!r} not found in {folders}")
    base = raw[DEFAULT_THEME][1]
    keys = [DEFAULT_THEME, *sorted(key for key in raw if key != DEFAULT_THEME)]
    return {key: Theme(key, raw[key][0], {**base, **raw[key][1]}) for key in keys}


def _write_icons(theme: Theme, folder: Path) -> None:
    """Arrows and tick in the theme's colours (a stylesheet takes images from files)."""
    folder.mkdir(parents=True, exist_ok=True)
    dim, on_accent = theme.tokens["dim"], theme.tokens["accent_text"]
    icons = {
        "down.svg": _ARROW.format(path="M2 3.5 5 6.5 8 3.5", colour=dim),
        "up.svg": _ARROW.format(path="M2 6.5 5 3.5 8 6.5", colour=dim),
        "check.svg": _TICK.format(colour=on_accent),
    }
    for name, content in icons.items():
        (folder / name).write_text(content, encoding="utf-8")


def build_stylesheet(theme: Theme, icon_folder: Path | None = None) -> str:
    """The template with the theme's values in place of the tokens.

    Args:
        theme: Theme to render.
        icon_folder: Where its icons are written; a per-theme folder in the
            system's temporary directory when ``None``.

    Raises:
        ValueError: The template uses a token the theme does not define.
    """
    if icon_folder is None:
        icon_folder = Path(tempfile.gettempdir()) / "stealzalut" / "icons" / theme.key
    _write_icons(theme, icon_folder)
    values = {**theme.tokens, "icons": icon_folder.as_posix()}
    template = get_styles_path("theme.qss").read_text(encoding="utf-8")
    unknown = sorted({token for token in _TOKEN.findall(template) if token not in values})
    if unknown:
        raise ValueError(f"theme {theme.key!r} lacks the tokens {unknown}")
    return _TOKEN.sub(lambda match: values[match.group(1)], template)


def apply_theme(app: QApplication, key: str, folders: list[Path] | None = None) -> Theme | None:
    """Style the application with the theme *key*, or the default one if unknown.

    Never raises: an application without its stylesheet still works, so a
    failure is logged and ``None`` returned.
    """
    try:
        themes = available_themes(folders)
        theme = themes.get(key) or themes[DEFAULT_THEME]
        if theme.key != key:
            logger.info("Theme %r unknown, using %r", key, theme.key)
        app.setStyleSheet(build_stylesheet(theme))
    except (OSError, ValueError) as exc:
        logger.warning("Theme not applied: %s", exc)
        return None
    return theme
