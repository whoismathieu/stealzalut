"""Tests for stealzalut.main — Application lifecycle and bootstrap."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication

from stealzalut import __version__
from stealzalut.config import Settings
from stealzalut.main import create_application, load_stylesheet, setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_stream(self) -> None:
        settings = Settings(log_level="DEBUG")
        setup_logging(settings)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_logging_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "app.log"
        settings = Settings(log_level="INFO", log_file=log_file)
        setup_logging(settings)
        logger = logging.getLogger("stealzalut.test")
        logger.info("Test message")

        # Flush handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert log_file.exists()
        assert "Test message" in log_file.read_text(encoding="utf-8")


class TestApplicationBootstrap:
    """Tests for create_application and stylesheet loading."""

    def test_create_application_properties(self) -> None:
        app = create_application(["test_app"])
        assert isinstance(app, QApplication)
        assert app.applicationName() == "StealZaLut"
        assert app.applicationVersion() == __version__
        assert app.organizationName() == "StealZaLut"

    def test_load_stylesheet_existing(self) -> None:
        app = create_application([])
        load_stylesheet(app, "theme.qss")
        # Should have applied a stylesheet containing dark theme rules
        assert len(app.styleSheet()) > 0
        assert "#1a1b1e" in app.styleSheet()

    def test_load_stylesheet_missing_does_not_crash(self) -> None:
        app = create_application([])
        # Reset stylesheet
        app.setStyleSheet("")
        load_stylesheet(app, "nonexistent.qss")
        # Should smoothly log warning and leave stylesheet empty
        assert app.styleSheet() == ""

