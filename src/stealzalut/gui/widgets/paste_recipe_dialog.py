"""Dialog to paste the answer of the user's AI.

The whole answer can be pasted (reasoning included): the recipe is the
last ```json block.  Reading errors are shown in the dialog, which stays
open so the text can be fixed.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from stealzalut.core.recipe import Recipe, RecipeError, parse_recipe


def _looks_like_answer(text: str) -> bool:
    return "stealzalut.recipe" in text or "```json" in text


class PasteRecipeDialog(QDialog):
    """Paste an AI answer; :attr:`recipe` holds the parsed recipe on accept."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Coller la réponse de l'IA")
        self.setObjectName("pasteRecipeDialog")
        self.resize(640, 460)
        self.recipe: Recipe | None = None

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Colle ici toute la réponse de ton IA : le texte autour ne gêne pas, "
            "StealZaLut lit le dernier bloc <code>```json</code>."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._editor = QPlainTextEdit()
        self._editor.setObjectName("recipeEditor")
        self._editor.setPlaceholderText("Réponse de l'IA…")
        layout.addWidget(self._editor, stretch=1)

        self._error = QLabel()
        self._error.setObjectName("conflictBanner")
        self._error.setWordWrap(True)
        self._error.hide()
        layout.addWidget(self._error)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setText("Appliquer")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        clipboard = QGuiApplication.clipboard().text()
        if _looks_like_answer(clipboard):
            self._editor.setPlainText(clipboard)

    @property
    def text(self) -> str:
        return self._editor.toPlainText()

    def set_text(self, text: str) -> None:
        self._editor.setPlainText(text)

    @property
    def error_text(self) -> str:
        return self._error.text() if self._error.isVisibleTo(self) else ""

    def _on_accept(self) -> None:
        try:
            self.recipe = parse_recipe(self.text)
        except RecipeError as exc:
            self._error.setText(f"⚠ {exc}")
            self._error.show()
            return
        self.accept()
