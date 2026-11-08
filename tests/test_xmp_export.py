"""Tests for stealzalut.core.xmp_export — Lightroom Classic profile export."""

from __future__ import annotations

import hashlib
import re
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from stealzalut.core.lut_engine import generate_neutral_grid
from stealzalut.core.lut_export import write_cube_file
from stealzalut.core.xmp_export import (
    B85_ALPHABET,
    HEADER_SIZE,
    TABLE_DATA_BYTES,
    TRAILER_SIZE,
    UNCOMPRESSED_PAYLOAD_SIZE,
    b85_decode_le,
    b85_encode_le,
    build_xmp_table_payload,
    cube_to_xmp,
    format_xmp_document,
    read_cube_file,
    read_xmp_profile,
    trilinear_resample_3d,
    write_xmp_profile,
    xmp_to_cube,
)


class TestBase85Codec:
    """Tests for Adobe Camera Raw Little-Endian Base85 codec."""

    def test_alphabet_length_and_characters(self) -> None:
        """Alphabet must contain exactly 85 unique characters."""
        assert len(B85_ALPHABET) == 85
        assert len(set(B85_ALPHABET)) == 85

    def test_encode_decode_roundtrip_random(self) -> None:
        """Arbitrary byte payloads must round-trip with byte-for-byte fidelity."""
        rng = np.random.default_rng(1234)
        for size in [1, 2, 3, 4, 5, 7, 8, 16, 127, 256, 1024]:
            payload = rng.bytes(size)
            encoded = b85_encode_le(payload)
            decoded = b85_decode_le(encoded)
            assert decoded == payload

    def test_empty_bytes(self) -> None:
        """Encoding empty bytes produces empty string and decodes to empty bytes."""
        assert b85_encode_le(b"") == ""
        assert b85_decode_le("") == b""

    def test_known_size_prefix(self) -> None:
        """Adobe Camera Raw's table size prefix (196652 = 0x3002c) must encode to 'Lir00'."""
        size_bytes = struct.pack("<I", 196652)
        encoded = b85_encode_le(size_bytes)
        assert encoded == "Lir00"
        assert b85_decode_le("Lir00") == size_bytes

    def test_remainder_handling(self) -> None:
        """Lengths not divisible by 4 (remainder 1, 2, 3) must encode and decode cleanly."""
        for rem in [1, 2, 3]:
            data = b"ABCDEFG"[:rem]
            enc = b85_encode_le(data)
            assert len(enc) == rem + 1
            assert b85_decode_le(enc) == data


class TestTrilinearResample3D:
    """Tests for 3D LUT trilinear interpolation."""

    def test_same_size_returns_copy(self) -> None:
        """Resampling when in_size == out_size returns an identical copy."""
        grid = generate_neutral_grid(17)
        resampled = trilinear_resample_3d(grid, in_size=17, out_size=17)
        assert np.array_equal(grid, resampled)
        assert resampled is not grid

    @pytest.mark.parametrize("size", [17, 33, 64])
    def test_resample_neutral_grid_matches_identity(self, size: int) -> None:
        """Resampling a neutral identity grid of any size to 32³ preserves identity."""
        grid = generate_neutral_grid(size)
        resampled = trilinear_resample_3d(grid, in_size=size, out_size=32)

        # Expected 32³ identity grid in standard .cube layout
        axis = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        b_g, g_g, r_g = np.meshgrid(axis, axis, axis, indexing="ij")
        expected = np.stack([r_g.ravel(), g_g.ravel(), b_g.ravel()], axis=-1)

        assert resampled.shape == (32**3, 3)
        assert np.allclose(resampled, expected, atol=1e-7)


class TestBuildXmpTablePayload:
    """Tests for build_xmp_table_payload()."""

    def test_exact_payload_length(self) -> None:
        """Payload must be exactly 196,652 bytes."""
        grid = generate_neutral_grid(17)
        payload = build_xmp_table_payload(grid, size=17)
        assert len(payload) == UNCOMPRESSED_PAYLOAD_SIZE
        assert len(payload) == 196652

    def test_header_structure(self) -> None:
        """Header must be 16 bytes: uint32(1, 1, 3, 32)."""
        grid = generate_neutral_grid(33)
        payload = build_xmp_table_payload(grid, size=33)
        header_vals = struct.unpack("<4I", payload[:HEADER_SIZE])
        assert header_vals == (1, 1, 3, 32)

    def test_trailer_structure(self) -> None:
        """Trailer must be 28 bytes: uint32(0, 1, 0, 0, 0), double(2.0)."""
        grid = generate_neutral_grid(17)
        payload = build_xmp_table_payload(grid, size=17)
        trailer = payload[-TRAILER_SIZE:]
        u32_vals = struct.unpack("<5I", trailer[:20])
        double_val = struct.unpack("<d", trailer[20:28])[0]
        assert u32_vals == (0, 1, 0, 0, 0)
        assert double_val == 2.0

    def test_identity_grid_deltas_are_all_zeros(self) -> None:
        """For an exact neutral identity grid, all table delta values must be 0."""
        axis = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        b_g, g_g, r_g = np.meshgrid(axis, axis, axis, indexing="ij")
        grid32 = np.stack([r_g.ravel(), g_g.ravel(), b_g.ravel()], axis=-1)

        payload = build_xmp_table_payload(grid32, size=32)
        table_bytes = payload[HEADER_SIZE : HEADER_SIZE + TABLE_DATA_BYTES]
        table_i16 = np.frombuffer(table_bytes, dtype=np.int16)
        assert np.all(table_i16 == 0)

    def test_channel_and_axis_alignment(self) -> None:
        """Modifying only Red in .cube must produce deltas ONLY on channel 0 (Red) along axis 0."""
        axis = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        b_g, g_g, r_g = np.meshgrid(axis, axis, axis, indexing="ij")

        # Only modify red output by -0.1
        red_modified = np.maximum(0.0, r_g.ravel() - 0.1)
        grid = np.stack([red_modified, g_g.ravel(), b_g.ravel()], axis=-1)

        payload = build_xmp_table_payload(grid, size=32)
        table_bytes = payload[HEADER_SIZE : HEADER_SIZE + TABLE_DATA_BYTES]
        tbl = np.frombuffer(table_bytes, dtype="<i2").reshape((32, 32, 32, 3))

        # Channel 0 (Red delta) must be negative or zero everywhere
        assert np.all(tbl[..., 0] <= 0)
        # Channels 1 (Green) and 2 (Blue) must be exactly zero everywhere
        assert np.all(tbl[..., 1] == 0)
        assert np.all(tbl[..., 2] == 0)

        # Axis 0 in tbl corresponds to input Red coordinate
        # At index (16, 0, 0), Red delta should be ~ -0.1 * 65535 = -6554
        expected_delta = round(-0.1 * 65535.0)
        assert abs(tbl[16, 0, 0, 0] - expected_delta) <= 1

    def test_invalid_grid_shape_raises(self) -> None:
        """Passing an incorrect shape must raise ValueError."""
        bad_grid = np.zeros((100, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="does not match expected"):
            build_xmp_table_payload(bad_grid, size=32)


class TestFormatXmpDocument:
    """Tests for format_xmp_document()."""

    def test_contains_attributes_and_tags(self) -> None:
        """XML must contain all required crs attributes and name/group tags."""
        md5_dummy = "A" * 32
        base85_dummy = "Lir00dummy"
        doc = format_xmp_document(
            table_md5=md5_dummy,
            table_base85=base85_dummy,
            name="MyProfile",
            group="MyGroup",
            profile_uuid="12345678123456781234567812345678",
        )
        assert 'crs:PresetType="Look"' in doc
        assert 'crs:SupportsAmount="True"' in doc
        assert 'crs:UUID="12345678123456781234567812345678"' in doc
        assert f'crs:RGBTable="{md5_dummy}"' in doc
        assert f'crs:Table_{md5_dummy}="{base85_dummy}"' in doc
        assert "<crs:Name>" in doc
        assert '<rdf:li xml:lang="x-default">MyProfile</rdf:li>' in doc
        assert "<crs:Group>" in doc
        assert '<rdf:li xml:lang="x-default">MyGroup</rdf:li>' in doc


class TestWriteXmpProfile:
    """Tests for write_xmp_profile()."""

    def test_creates_file_and_appends_extension(self, tmp_path: Path) -> None:
        """File is created with .xmp extension automatically appended."""
        grid = generate_neutral_grid(17)
        out = write_xmp_profile(tmp_path / "test_profile", grid, size=17)
        assert out.exists()
        assert out.suffix == ".xmp"

    def test_progress_callback(self, tmp_path: Path) -> None:
        """Progress callback is called and reaches 100%."""
        grid = generate_neutral_grid(17)
        progress: list[int] = []
        write_xmp_profile(
            tmp_path / "test_progress.xmp",
            grid,
            size=17,
            progress_callback=progress.append,
        )
        assert len(progress) > 0
        assert progress[-1] == 100

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created automatically."""
        grid = generate_neutral_grid(17)
        deep_path = tmp_path / "sub" / "folder" / "profile.xmp"
        out = write_xmp_profile(deep_path, grid, size=17)
        assert out.exists()

    def test_file_payload_is_valid_and_decodable(self, tmp_path: Path) -> None:
        """Exported .xmp profile must be decodable and match fingerprint."""
        grid = generate_neutral_grid(33)
        out = write_xmp_profile(
            tmp_path / "valid.xmp",
            grid,
            size=33,
            name="ValidProfile",
            group="StealZaLut",
        )
        content = out.read_text(encoding="utf-8")

        # Extract RGBTable MD5 and Table Base85 string
        m_md5 = re.search(r'crs:RGBTable="([0-9A-F]{32})"', content)
        assert m_md5 is not None
        md5_in_xml = m_md5.group(1)

        m_table = re.search(r'crs:Table_' + md5_in_xml + r'="([^"]+)"', content)
        assert m_table is not None
        b85_str = m_table.group(1)

        # Base85 decode
        framed = b85_decode_le(b85_str)
        assert len(framed) > 4
        uncomp_size = struct.unpack("<I", framed[:4])[0]
        assert uncomp_size == UNCOMPRESSED_PAYLOAD_SIZE

        # Decompress zlib payload
        decomp = zlib.decompress(framed[4:])
        assert len(decomp) == UNCOMPRESSED_PAYLOAD_SIZE

        # Verify MD5 fingerprint matches uncompressed payload
        calculated_md5 = hashlib.md5(decomp).hexdigest().upper()
        assert calculated_md5 == md5_in_xml


class TestRoundtripAndConversion:
    """Tests for cube_to_xmp(), read_xmp_profile(), and xmp_to_cube()."""

    def test_cube_to_xmp_and_back_roundtrip(self, tmp_path: Path) -> None:
        """A .cube file converted to .xmp and read back reproduces original values."""
        cube_path = tmp_path / "original.cube"
        grid = generate_neutral_grid(33)
        # Apply distinct color transformation
        grid[:, 0] = np.clip(grid[:, 0] * 1.05, 0.0, 1.0)
        grid[:, 1] = np.power(grid[:, 1], 1.1)
        grid[:, 2] = np.clip(grid[:, 2] * 0.95 + 0.02, 0.0, 1.0)

        write_cube_file(cube_path, grid, size=33, title="OriginalLook")

        xmp_path = cube_to_xmp(cube_path, profile_name="ConvertedLook")
        assert xmp_path.exists()

        # Read back via read_xmp_profile (returns 32³ grid)
        recovered_grid, rec_size, rec_name, _ = read_xmp_profile(xmp_path)
        assert rec_size == 32
        assert rec_name == "ConvertedLook"

        # Resample original 33³ grid to 32³ to compare
        expected_grid32 = trilinear_resample_3d(grid, in_size=33, out_size=32)
        max_err = np.abs(expected_grid32 - recovered_grid).max()
        assert max_err <= 1.6e-5

        # Also test xmp_to_cube
        cube_back = xmp_to_cube(xmp_path, tmp_path / "recovered.cube")
        assert cube_back.exists()
        grid_from_file, size_from_file = read_cube_file(cube_back)
        assert size_from_file == 32
        assert np.abs(expected_grid32 - grid_from_file).max() <= 1.6e-5

    def test_read_cube_file_missing_file_raises(self, tmp_path: Path) -> None:
        """Attempting to read a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_cube_file(tmp_path / "nonexistent.cube")

    def test_read_cube_file_missing_header_raises(self, tmp_path: Path) -> None:
        """A file missing the LUT_3D_SIZE keyword raises ValueError."""
        bad_cube = tmp_path / "bad.cube"
        bad_cube.write_text("TITLE Test\n0.1 0.2 0.3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="LUT_3D_SIZE keyword not found"):
            read_cube_file(bad_cube)
