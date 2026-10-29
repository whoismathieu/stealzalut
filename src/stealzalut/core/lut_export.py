"""3D LUT export to the ``.cube`` file format.

The ``.cube`` format (originally by Iridas, now an industry standard)
is a plain-text file containing:

1. A **header** with metadata (title, grid size, domain bounds).
2. A **data section** listing one RGB triplet per line, in the order
   R-fastest / G-middle / B-slowest.

This module handles formatting and writing, with optional progress
reporting for integration with UI workers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def format_cube_header(
    size: int,
    title: str = "StealZaLut",
) -> str:
    """Generate the header section of a ``.cube`` file.

    Args:
        size: LUT grid resolution per axis.
        title: Title metadata string (embedded in the file).

    Returns:
        Formatted header string ending with a blank line separator.
    """
    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",  # blank line before data
    ]
    return "\n".join(lines) + "\n"


def write_cube_file(
    path: str | Path,
    grid: NDArray[np.float64],
    size: int,
    title: str = "StealZaLut",
    precision: int = 7,
    progress_callback: Callable[[int], None] | None = None,
) -> Path:
    """Write a transformed 3D LUT grid to a ``.cube`` file.

    Args:
        path: Output file path.  The ``.cube`` extension is appended
              automatically if missing.
        grid: Transformed LUT grid of shape ``(size³, 3)`` with values
              in [0.0, 1.0].
        size: Grid resolution per axis.  Must match ``grid.shape[0]``.
        title: Title string written into the file header.
        precision: Number of decimal digits for colour values.
        progress_callback: Optional callable receiving an ``int``
            percentage (0–100) as writing progresses.

    Returns:
        The resolved output :class:`~pathlib.Path`.

    Raises:
        ValueError: If ``grid.shape`` does not match ``(size³, 3)``.
    """
    expected_entries = size**3
    if grid.shape != (expected_entries, 3):
        raise ValueError(
            f"Grid shape {grid.shape} does not match expected "
            f"({expected_entries}, 3) for LUT size {size}"
        )

    path = Path(path)
    if path.suffix.lower() != ".cube":
        path = path.with_suffix(".cube")
    path.parent.mkdir(parents=True, exist_ok=True)

    header = format_cube_header(size, title)
    total = len(grid)
    fmt_str = f"%.{precision}f"

    # Chunked writing for progress reporting
    chunk_size = max(1, total // 100)  # ~1 % per chunk

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = grid[start:end]

            # numpy.savetxt is significantly faster than Python loops
            lines = "\n".join(
                " ".join(fmt_str % v for v in row) for row in chunk
            )
            fh.write(lines)
            fh.write("\n")

            if progress_callback is not None:
                pct = int((end / total) * 100)
                progress_callback(min(pct, 100))

    return path


def read_cube_size(path: str | Path) -> int:
    """Read the ``LUT_3D_SIZE`` value from an existing ``.cube`` file.

    Args:
        path: Path to the ``.cube`` file.

    Returns:
        The grid resolution per axis.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If ``LUT_3D_SIZE`` is not found in the header.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cube file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("LUT_3D_SIZE"):
                parts = stripped.split()
                if len(parts) >= 2:
                    return int(parts[1])
                break

    raise ValueError(f"LUT_3D_SIZE not found in {path}")

