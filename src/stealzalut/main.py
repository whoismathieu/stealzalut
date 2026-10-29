"""StealZaLut entry point.

Initialises application settings, logging, the PySide6 QApplication,
applies the active theme stylesheet, and launches the main window.
"""

from __future__ import annotations

import logging
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from stealzalut import __version__
from stealzalut.config import Settings, load_settings
from stealzalut.gui.main_window import MainWindow
from stealzalut.utils.paths import get_styles_path

logger = logging.getLogger("stealzalut")


def setup_logging(settings: Settings) -> None:
    """Configure standard logging for the application."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if settings.log_file is not None:
        try:
            settings.log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))
        except OSError as exc:
            print(f"Warning: could not open log file {settings.log_file}: {exc}", file=sys.stderr)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def load_stylesheet(app: QApplication, filename: str = "theme.qss") -> None:
    """Load and apply a Qt stylesheet to the application."""
    try:
        qss_path = get_styles_path(filename)
        if qss_path.exists():
            stylesheet = qss_path.read_text(encoding="utf-8")
            app.setStyleSheet(stylesheet)
            logger.debug("Applied stylesheet from %s", qss_path)
        else:
            logger.warning("Stylesheet not found at %s, proceeding without style", qss_path)
    except Exception as exc:
        logger.warning("Failed to load stylesheet: %s", exc)


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Create and configure the QApplication singleton."""
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(list(argv) if argv is not None else sys.argv)

    app.setApplicationName("StealZaLut")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("StealZaLut")
    return app


def main(argv: Sequence[str] | None = None, env_path: Path | None = None) -> int:
    """Main application lifecycle runner."""
    # Allow Python to handle Ctrl+C cleanly
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    settings = load_settings(env_path)
    setup_logging(settings)
    logger.info("Starting StealZaLut v%s", __version__)

    app = create_application(argv)

    # Let Python interpreter run periodically to catch OS signals
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(500)

    # Apply style
    if settings.app_theme == "dark":
        load_stylesheet(app, "theme.qss")

    window = MainWindow(settings=settings)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
