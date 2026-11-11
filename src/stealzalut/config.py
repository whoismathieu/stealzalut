"""Application configuration loaded from .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings.

    All values have sensible defaults and can be overridden via the
    ``.env`` file or environment variables.
    """

    # LUT export defaults
    lut_default_size: int = 33
    lut_output_dir: Path = field(default_factory=lambda: Path.home() / "Documents")

    # Color transfer
    color_transfer_method: str = "look"

    # GUI
    app_theme: Literal["dark", "light"] = "dark"
    preview_max_size: int = 512

    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None


def load_settings(env_path: Path | None = None) -> Settings:
    """Load settings from a ``.env`` file, falling back to defaults.

    Args:
        env_path: Explicit path to the ``.env`` file.
                  If ``None``, ``python-dotenv`` searches from the cwd upward.

    Returns:
        A frozen ``Settings`` instance.
    """
    load_dotenv(dotenv_path=env_path)

    lut_output_raw = os.getenv("LUT_OUTPUT_DIR", "").strip()
    log_file_raw = os.getenv("LOG_FILE", "").strip()

    return Settings(
        lut_default_size=int(os.getenv("LUT_DEFAULT_SIZE", "33")),
        lut_output_dir=Path(lut_output_raw) if lut_output_raw else Path.home() / "Documents",
        color_transfer_method=os.getenv("COLOR_TRANSFER_METHOD", "look"),
        app_theme=os.getenv("APP_THEME", "dark"),  # type: ignore[arg-type]
        preview_max_size=int(os.getenv("PREVIEW_MAX_SIZE", "512")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=Path(log_file_raw) if log_file_raw else None,
    )

