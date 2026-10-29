"""Tests for stealzalut.core.lut_export — .cube file writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stealzalut.core.lut_engine import generate_neutral_grid
from stealzalut.core.lut_export import format_cube_header, read_cube_size, write_cube_file


class TestFormatCubeHeader:
    """Tests for format_cube_header()."""

    def test_contains_title(self) -> None:
        header = format_cube_header(33, title="MyLUT")
        assert 'TITLE "MyLUT"' in header

    def test_contains_size(self) -> None:
        header = format_cube_header(33)
        assert "LUT_3D_SIZE 33" in header

    def test_contains_domain(self) -> None:
        header = format_cube_header(17)
        assert "DOMAIN_MIN 0.0 0.0 0.0" in header
        assert "DOMAIN_MAX 1.0 1.0 1.0" in header

    def test_default_title(self) -> None:
        header = format_cube_header(17)
        assert "StealZaLut" in header


class TestWriteCubeFile:
    """Tests for write_cube_file()."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """A .cube file must be created on disk."""
        grid = generate_neutral_grid(17)
        out = write_cube_file(tmp_path / "test.cube", grid, size=17)
        assert out.exists()
        assert out.suffix == ".cube"

    def test_appends_extension(self, tmp_path: Path) -> None:
        """The .cube extension is added when missing."""
        grid = generate_neutral_grid(17)
        out = write_cube_file(tmp_path / "test", grid, size=17)
        assert out.suffix == ".cube"

    def test_file_content_header(self, tmp_path: Path) -> None:
        """The file must start with a valid header."""
        grid = generate_neutral_grid(17)
        out = write_cube_file(tmp_path / "test.cube", grid, size=17)
        content = out.read_text(encoding="utf-8")
        assert content.startswith('TITLE "StealZaLut"')
        assert "LUT_3D_SIZE 17" in content

    def test_file_line_count(self, tmp_path: Path) -> None:
        """Total lines = header (4 lines + 1 blank) + size³ data lines."""
        size = 17
        grid = generate_neutral_grid(size)
        out = write_cube_file(tmp_path / "test.cube", grid, size=size)
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        header_lines = 4  # TITLE, LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX
        # Allow for the blank separator line
        data_lines = len(lines) - header_lines - 1
        assert data_lines == size**3

    def test_mismatched_grid_raises(self, tmp_path: Path) -> None:
        """A grid that does not match the declared size must raise."""
        grid = np.zeros((100, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="does not match expected"):
            write_cube_file(tmp_path / "bad.cube", grid, size=17)

    def test_progress_callback(self, tmp_path: Path) -> None:
        """The progress callback must be invoked and reach 100."""
        grid = generate_neutral_grid(17)
        progress_values: list[int] = []
        write_cube_file(
            tmp_path / "test.cube",
            grid,
            size=17,
            progress_callback=progress_values.append,
        )
        assert len(progress_values) > 0
        assert progress_values[-1] == 100

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Intermediate directories must be created automatically."""
        grid = generate_neutral_grid(17)
        deep_path = tmp_path / "a" / "b" / "c" / "output.cube"
        out = write_cube_file(deep_path, grid, size=17)
        assert out.exists()


class TestReadCubeSize:
    """Tests for read_cube_size()."""

    def test_reads_size(self, tmp_path: Path) -> None:
        grid = generate_neutral_grid(33)
        out = write_cube_file(tmp_path / "test.cube", grid, size=33)
        assert read_cube_size(out) == 33

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_cube_size(tmp_path / "nonexistent.cube")

    def test_missing_header_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.cube"
        bad_file.write_text("TITLE \"bad\"\n0.0 0.0 0.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="LUT_3D_SIZE not found"):
            read_cube_size(bad_file)

