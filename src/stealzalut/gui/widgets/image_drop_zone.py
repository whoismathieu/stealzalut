"""Drag-and-drop image widget.

Displays a placeholder label when empty, or the image fitted to the zone
when one has been loaded (via drag-and-drop, paste or file dialog); the image
can be zoomed and panned (see
:class:`~stealzalut.gui.widgets.image_view.ImageView`).  A picture dragged
from a web page or pasted from the clipboard is accepted like a file (see
:mod:`stealzalut.gui.incoming_image`).  The path of
the loaded file is shown at the bottom-left of the zone, selectable and
copyable (Ctrl+C or the copy button).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QMimeData, Qt, QTimer, Signal
from PySide6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from stealzalut.gui.incoming_image import IncomingImageError, Offer, fetch, read_offer
from stealzalut.gui.overlay import highlight_zone
from stealzalut.gui.widgets.image_view import ImageView
from stealzalut.utils.image_io import (
    SUPPORTED_EXTENSIONS,
    create_thumbnail,
    load_image_rgb,
    rgb_to_bgr,
)

logger = logging.getLogger(__name__)

_COPY_LABEL = "Copier"
_COPIED_LABEL = "Copié ✓"
_COPIED_FEEDBACK_MS = 1500

#: Longest edge of the copy on display: the size the preview is rendered at,
#: so that both stay sharp when zoomed.
DISPLAY_MAX_SIZE = 1280

_RAW_FILTER_EXTS = (
    "*.cr2 *.cr3 *.nef *.arw *.raf *.rw2 *.orf *.pef *.dng *.raw "
    "*.crw *.nrw *.srf *.sr2 *.rwl *.3fr *.fff *.iiq *.mrw *.x3f"
)

_FILTER = (
    "All Supported Images ({all_ext});;"
    f"Camera RAW & DNG ({_RAW_FILTER_EXTS});;"
    "Standard Images (*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp);;"
    "Apple HEIC / HEIF (*.heic *.heif *.hif);;"
    "All Files (*.*)"
).format(all_ext=" ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS)))


class ImageDropZone(QWidget):
    """Widget accepting image files via drag-and-drop or click-to-browse.

    Signals:
        image_loaded (object): Emitted with the loaded RGB uint8 ndarray
            when a valid image is successfully loaded.
        image_cleared: Emitted when the image is removed.
        load_failed (str): A drop or a paste gave no image; the reason, in
            French, for the status bar.

    The widget stores the **full-resolution RGB image** internally and
    only displays a down-scaled copy.  A click that is not a drag opens the
    file dialog.
    """

    image_loaded = Signal(object)
    image_cleared = Signal()
    load_failed = Signal(str)

    def __init__(
        self,
        label_text: str = "Dépose une image\nou clique pour la choisir",
        max_preview: int = 280,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)

        self._label_text = label_text
        self._max_preview = max_preview
        self._image_rgb: NDArray[np.uint8] | None = None
        self._image_path: Path | None = None
        self._thumb_rgb: NDArray[np.uint8] | None = None
        # Temporary files written for pictures that came from the web or the clipboard
        self._fetched: list[Path] = []

        # Accept drops
        self.setAcceptDrops(True)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = ImageView(label_text)
        self._label.setMinimumSize(200, 160)
        self._label.setObjectName("dropZoneLabel")
        self._label.clicked.connect(self._open_file_dialog)
        self._label.menu_entries.append(("Coller une image (Ctrl+V)", self.paste_from_clipboard))
        layout.addWidget(self._label)

        # Path row (bottom-left): read-only field + copy button, hidden when empty
        self._path_row = QWidget()
        path_layout = QHBoxLayout(self._path_row)
        path_layout.setContentsMargins(0, 4, 0, 0)
        path_layout.setSpacing(6)

        self._path_field = QLineEdit()
        self._path_field.setReadOnly(True)
        self._path_field.setObjectName("pathField")
        # No keyboard focus at startup (avoids a highlighted field on launch)
        self._path_field.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._path_field.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        path_layout.addWidget(self._path_field, stretch=1)

        self._copy_btn = QToolButton()
        self._copy_btn.setText(_COPY_LABEL)
        self._copy_btn.setToolTip("Copier le chemin dans le presse-papiers")
        self._copy_btn.setObjectName("copyPathButton")
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.clicked.connect(self.copy_path_to_clipboard)
        path_layout.addWidget(self._copy_btn)

        layout.addWidget(self._path_row)
        self._path_row.hide()

        # Restores the copy button label after the "copied" feedback
        self._copy_feedback_timer = QTimer(self)
        self._copy_feedback_timer.setSingleShot(True)
        self._copy_feedback_timer.setInterval(_COPIED_FEEDBACK_MS)
        self._copy_feedback_timer.timeout.connect(lambda: self._copy_btn.setText(_COPY_LABEL))

        self.setObjectName("imageDropZone")

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def image_rgb(self) -> NDArray[np.uint8] | None:
        """The full-resolution RGB image, or ``None`` if empty."""
        return self._image_rgb

    @property
    def image_path(self) -> Path | None:
        """Path of the loaded image file, or ``None``."""
        return self._image_path

    @property
    def path_text(self) -> str:
        """The displayed path of the loaded file (empty when no image)."""
        return self._path_field.text()

    @property
    def view(self) -> ImageView:
        """The zoomable display of the image."""
        return self._label

    @property
    def has_image(self) -> bool:
        """Whether an image is currently loaded."""
        return self._image_rgb is not None

    def clear_image(self) -> None:
        """Remove the loaded image and reset to placeholder."""
        self._image_rgb = None
        self._image_path = None
        self._thumb_rgb = None
        self._label.set_rgb(None)
        self._set_path_display(None)
        self.image_cleared.emit()

    def copy_path_to_clipboard(self) -> None:
        """Copy the loaded file's path to the system clipboard."""
        if not self.path_text:
            return
        QGuiApplication.clipboard().setText(self.path_text)
        self._copy_btn.setText(_COPIED_LABEL)
        self._copy_feedback_timer.start()

    def load_from_path(self, path: str | Path) -> bool:
        """Load an image from *path*.

        Returns:
            ``True`` if the image was loaded successfully.
        """
        path = Path(path)
        try:
            rgb = load_image_rgb(path)
            bgr = rgb_to_bgr(np.asarray(rgb))
            self._image_rgb = rgb
            self._image_path = path
            self._update_thumbnail(bgr)
            self._set_path_display(path)
            self.image_loaded.emit(rgb)
            logger.info("Loaded image: %s (%s)", path.name, rgb.shape)
            return True
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Failed to load image %s: %s", path, exc)
            return False

    # ── Drag-and-drop ────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        # A supported file, or anything a browser gives for a picture
        if read_offer(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        offer = read_offer(event.mimeData())
        if not offer:
            event.ignore()
            return
        event.acceptProposedAction()
        if offer.local_file is not None:
            self.load_offer(offer)
        else:
            # Fetching may take a moment: let the browser finish its drag first
            QTimer.singleShot(0, lambda: self.load_offer(offer))

    # ── Paste, and pictures that are not files ───────────────────────────

    def paste_from_clipboard(self) -> bool:
        """Load the picture held by the clipboard (image, file or web address)."""
        return self.load_mime(QGuiApplication.clipboard().mimeData())

    def discard_fetched(self) -> None:
        """Delete the temporary files this zone wrote (when the window closes)."""
        for path in self._fetched:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Could not delete %s: %s", path, exc)
        self._fetched.clear()

    def load_mime(self, mime: QMimeData) -> bool:
        """Load the picture *mime* carries; ``False`` (and :attr:`load_failed`) if none."""
        return self.load_offer(read_offer(mime))

    def load_offer(self, offer: Offer) -> bool:
        """Load the picture of *offer*, fetching it from the web when it is an address."""
        if not offer:
            self.load_failed.emit("Aucune image dans ce qui a été déposé ou collé.")
            return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            path = fetch(offer)
        except IncomingImageError as exc:
            self.load_failed.emit(str(exc))
            return False
        except OSError as exc:
            self.load_failed.emit(f"Image non enregistrée temporairement : {exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        if path != offer.local_file and path not in self._fetched:
            self._fetched.append(path)
        if self.load_from_path(path):
            return True
        self.load_failed.emit("Cette image n'a pas pu être lue.")
        return False

    # ── Click to browse ──────────────────────────────────────────────────

    def _open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir une image",
            "",
            _FILTER,
        )
        if path:
            self.load_from_path(path)

    # ── Internal ─────────────────────────────────────────────────────────

    def _set_path_display(self, path: Path | None) -> None:
        """Show *path* in the path row, or hide the row when ``None``."""
        if path is None:
            self._path_field.clear()
            self._path_field.setToolTip("")
            self._path_row.hide()
            return
        text = str(path.absolute())
        self._path_field.setText(text)
        self._path_field.setToolTip(text)
        # Long paths: scroll so the file name (end of the path) stays visible
        self._path_field.setCursorPosition(len(text))
        self._copy_btn.setText(_COPY_LABEL)
        self._path_row.show()

    def set_highlight(self, mask: NDArray[np.bool_] | None) -> None:
        """Highlight a colour zone on the thumbnail (``None`` = plain image).

        Args:
            mask: Zone mask at any resolution with the image's aspect ratio.
        """
        if self._thumb_rgb is None:
            return
        shown = self._thumb_rgb if mask is None else highlight_zone(self._thumb_rgb, mask)
        self._show_rgb(shown)

    def _update_thumbnail(self, bgr_image: NDArray[np.uint8]) -> None:
        """Generate and display a thumbnail from the BGR image."""
        thumb = create_thumbnail(bgr_image, max(self._max_preview, DISPLAY_MAX_SIZE))
        rgb: NDArray[np.uint8] = np.ascontiguousarray(thumb[..., ::-1])  # BGR → RGB
        self._thumb_rgb = rgb
        self._show_rgb(rgb)

    def _show_rgb(self, rgb: NDArray[np.uint8]) -> None:
        self._label.set_rgb(rgb)
