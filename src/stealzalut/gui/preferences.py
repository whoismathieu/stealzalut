"""What the user chose in the window and expects to find again next time."""

from __future__ import annotations

from PySide6.QtCore import QSettings

#: Key of the theme (see :mod:`stealzalut.gui.theming`).
THEME = "appearance/theme"
#: Grey behind the images (see ``BACKDROPS`` in the image view).
IMAGE_BACKDROP = "appearance/image_backdrop"


class Preferences:
    """Small key/value store of interface choices.

    Without a store the values only live as long as the object: that is what
    a window built without preferences gets (tests, tools), so that nothing
    is read from or written to the user's own settings.
    """

    def __init__(self, store: QSettings | None = None) -> None:
        self._store = store
        self._memory: dict[str, str] = {}

    @classmethod
    def persistent(cls) -> Preferences:
        """Preferences kept by the system under the application's name."""
        return cls(QSettings())

    def get(self, key: str, default: str) -> str:
        """Stored value of *key*, or *default*."""
        if self._store is None:
            return self._memory.get(key, default)
        return str(self._store.value(key, default))

    def set(self, key: str, value: str) -> None:
        """Store *value* under *key*."""
        if self._store is None:
            self._memory[key] = value
        else:
            self._store.setValue(key, value)
