"""Lightroom Classic Profile (.xmp) export with embedded 3D LUT.

This module serialises colour-transform 3D LUT grids into native Adobe
Camera Raw / Lightroom Classic ``.xmp`` profile files (PresetType="Look").

The binary payload and encoding were reverse-engineered from Adobe Camera
Raw (CameraRaw.dll and Camera Raw.8bi):
1. **Grid size**: Resampled to a 32×32×32 cube if needed.
2. **Channel layout**: 3 channels per point stored in RGB order (Red, Green, Blue).
   Row order: Red slowest (outer, stride 32²), Green middle (stride 32),
   Blue fastest (inner, stride 1).
3. **Values**: Signed 16-bit deltas from identity:
   ``delta = round((target - identity) * 65535.0)`` clamped to [-32768, 32767].
4. **Binary structure**:
   - 16-byte header: uint32(1), uint32(1), uint32(3), uint32(32).
   - 196,608-byte table (32³ × 3 × 2 bytes).
   - 28-byte trailer: uint32(0), uint32(1), uint32(0), uint32(0), uint32(0), double(2.0).
   - Total uncompressed size: exactly 196,652 bytes.
5. **Compression**: Standard zlib compression with level 9.
6. **Framing**: Prefixed with 4-byte little-endian uncompressed size (196652 = 0x3002c).
7. **Encoding**: Custom Little-Endian Base85 encoding using Adobe's 85-char alphabet.
8. **Fingerprint**: Uppercase hex MD5 digest of the uncompressed 196,652-byte payload.
"""

from __future__ import annotations

import hashlib
import re
import struct
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Adobe Camera Raw custom Base85 alphabet (found at RVA in CameraRaw.dll)
B85_ALPHABET = (
    b"0123456789abcdefghijklmnopqrstuvwxyz"
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?`'|()[]{}@%$#"
)
_CHAR_TO_VAL = {c: i for i, c in enumerate(B85_ALPHABET)}

# Fixed table parameters for Camera Raw Look profiles
TABLE_GRID_SIZE = 32
TABLE_CHANNELS = 3
TABLE_DATA_BYTES = TABLE_GRID_SIZE**3 * TABLE_CHANNELS * 2  # 196,608 bytes
HEADER_SIZE = 16
TRAILER_SIZE = 28
UNCOMPRESSED_PAYLOAD_SIZE = HEADER_SIZE + TABLE_DATA_BYTES + TRAILER_SIZE  # 196,652 bytes


def b85_encode_le(data: bytes) -> str:
    """Encode bytes using Adobe Camera Raw's custom Little-Endian Base85.

    Unlike standard RFC 1924 or Ascii85 (which are big-endian where the first
    character represents the highest power :math:`85^4`), Adobe's algorithm
    processes 4-byte chunks in little-endian order where the first character
    represents :math:`85^0`.

    Args:
        data: Arbitrary binary data to encode.

    Returns:
        The encoded string consisting of characters from :data:`B85_ALPHABET`.
    """
    n = len(data)
    chunks = n // 4
    rem = n % 4

    out: list[bytes] = []
    # Full 4-byte uint32 chunks -> 5 characters
    for i in range(chunks):
        val = int.from_bytes(data[i * 4 : (i + 1) * 4], "little")
        c0 = val % 85
        val //= 85
        c1 = val % 85
        val //= 85
        c2 = val % 85
        val //= 85
        c3 = val % 85
        val //= 85
        c4 = val % 85
        out.append(
            bytes(
                [
                    B85_ALPHABET[c0],
                    B85_ALPHABET[c1],
                    B85_ALPHABET[c2],
                    B85_ALPHABET[c3],
                    B85_ALPHABET[c4],
                ]
            )
        )

    # Remainder (1 to 3 bytes) -> (rem + 1) characters
    if rem > 0:
        val = int.from_bytes(data[chunks * 4 :], "little")
        rem_chars = []
        for _ in range(rem + 1):
            rem_chars.append(B85_ALPHABET[val % 85])
            val //= 85
        out.append(bytes(rem_chars))

    return b"".join(out).decode("latin1")


def b85_decode_le(s: str) -> bytes:
    """Decode Adobe Camera Raw's custom Little-Endian Base85 string.

    Args:
        s: String encoded with :func:`b85_encode_le`.

    Returns:
        The decoded raw byte sequence.
    """
    b = s.encode("latin1")
    n = len(b)
    chunks = n // 5
    rem = n % 5

    out = bytearray()
    for i in range(chunks):
        chunk = b[i * 5 : (i + 1) * 5]
        val = 0
        mult = 1
        for c in chunk:
            val += _CHAR_TO_VAL[c] * mult
            mult *= 85
        out.extend(val.to_bytes(4, "little"))

    if rem > 0:
        chunk = b[chunks * 5 :]
        val = 0
        mult = 1
        for c in chunk:
            val += _CHAR_TO_VAL[c] * mult
            mult *= 85
        out.extend(val.to_bytes(4, "little")[: rem - 1])

    return bytes(out)


def trilinear_resample_3d(
    grid: NDArray[np.float64],
    in_size: int,
    out_size: int = TABLE_GRID_SIZE,
) -> NDArray[np.float64]:
    """Resample a 3D LUT grid to *out_size* along each axis using trilinear interpolation.

    Args:
        grid: Flat array of shape ``(in_size³, 3)`` in row order (B slowest, R fastest).
        in_size: Input grid resolution per axis.
        out_size: Target resolution per axis (default 32 for Camera Raw profiles).

    Returns:
        Flat array of shape ``(out_size³, 3)`` with values in [0.0, 1.0].
    """
    if in_size == out_size:
        return grid.copy()

    # Reshape to (in_size, in_size, in_size, 3) where axes are (B, G, R)
    vol = grid.reshape((in_size, in_size, in_size, 3))

    coords = np.linspace(0.0, in_size - 1.0, out_size, dtype=np.float64)
    # Target grid coordinates for (B, G, R)
    b_idx, g_idx, r_idx = np.meshgrid(coords, coords, coords, indexing="ij")

    # Integer coordinates and fractional weights for each axis
    b0 = np.floor(b_idx).astype(int)
    b1 = np.minimum(b0 + 1, in_size - 1)
    wb = (b_idx - b0)[..., np.newaxis]

    g0 = np.floor(g_idx).astype(int)
    g1 = np.minimum(g0 + 1, in_size - 1)
    wg = (g_idx - g0)[..., np.newaxis]

    r0 = np.floor(r_idx).astype(int)
    r1 = np.minimum(r0 + 1, in_size - 1)
    wr = (r_idx - r0)[..., np.newaxis]

    # 8 corner samples
    c000 = vol[b0, g0, r0]
    c001 = vol[b0, g0, r1]
    c010 = vol[b0, g1, r0]
    c011 = vol[b0, g1, r1]
    c100 = vol[b1, g0, r0]
    c101 = vol[b1, g0, r1]
    c110 = vol[b1, g1, r0]
    c111 = vol[b1, g1, r1]

    # Interpolate along R
    c00 = c000 * (1.0 - wr) + c001 * wr
    c01 = c010 * (1.0 - wr) + c011 * wr
    c10 = c100 * (1.0 - wr) + c101 * wr
    c11 = c110 * (1.0 - wr) + c111 * wr

    # Interpolate along G
    c0 = c00 * (1.0 - wg) + c01 * wg
    c1 = c10 * (1.0 - wg) + c11 * wg

    # Interpolate along B
    resampled = c0 * (1.0 - wb) + c1 * wb
    result: NDArray[np.float64] = np.asarray(resampled.reshape((-1, 3)), dtype=np.float64)
    return result


def build_xmp_table_payload(
    grid: NDArray[np.float64],
    size: int,
) -> bytes:
    """Build the exact uncompressed 196,652-byte binary payload for Camera Raw.

    Args:
        grid: Transformed 3D LUT grid of shape ``(size³, 3)`` in standard
              .cube row order (B slowest, G middle, R fastest) with RGB channels
              per row.
        size: Grid resolution per axis. If not 32, it will be resampled to 32.

    Returns:
        The 196,652-byte uncompressed payload buffer.
    """
    if size != TABLE_GRID_SIZE:
        grid32 = trilinear_resample_3d(grid, in_size=size, out_size=TABLE_GRID_SIZE)
    else:
        grid32 = grid

    expected_len = TABLE_GRID_SIZE**3
    if grid32.shape != (expected_len, 3):
        raise ValueError(
            f"Grid shape {grid32.shape} does not match expected ({expected_len}, 3)"
        )

    # Reshape .cube flat array into 3D volume with shape (B, G, R, 3)
    vol32 = grid32.reshape((TABLE_GRID_SIZE, TABLE_GRID_SIZE, TABLE_GRID_SIZE, 3))

    # Adobe Camera Raw / Lightroom Look tables are indexed with R slowest, G middle, B fastest.
    # Transpose axes 0 and 2 from (B, G, R) to (R, G, B):
    table_rgb = np.transpose(vol32, (2, 1, 0, 3))

    # Coordinates for identity lattice in (R, G, B) order
    coords = np.linspace(0.0, 1.0, TABLE_GRID_SIZE, dtype=np.float64)
    r_g, g_g, b_g = np.meshgrid(coords, coords, coords, indexing="ij")
    ident = np.stack([r_g, g_g, b_g], axis=-1)

    # Signed 16-bit deltas in RGB channel order: delta = round((target - identity) * 65535.0)
    deltas = np.clip(
        np.round((table_rgb - ident) * 65535.0),
        -32768,
        32767,
    ).astype(np.int16)

    # Ensure C-contiguous bytes: R slowest, G middle, B fastest, [delta_R, delta_G, delta_B]
    table_bytes = np.ascontiguousarray(deltas).tobytes()

    # 16-byte header: uint32(1), uint32(1), uint32(3), uint32(32)
    header = struct.pack("<4I", 1, 1, TABLE_CHANNELS, TABLE_GRID_SIZE)

    # 28-byte trailer: uint32(0), uint32(1), uint32(0), uint32(0), uint32(0), double(2.0)
    trailer = struct.pack("<5Id", 0, 1, 0, 0, 0, 2.0)

    payload = header + table_bytes + trailer
    if len(payload) != UNCOMPRESSED_PAYLOAD_SIZE:
        raise RuntimeError(
            f"Constructed payload size {len(payload)} != expected {UNCOMPRESSED_PAYLOAD_SIZE}"
        )

    return payload


def format_xmp_document(
    table_md5: str,
    table_base85: str,
    name: str = "StealZaLut",
    group: str = "StealZaLut",
    profile_uuid: str | None = None,
) -> str:
    """Format the XMP XML document embedding the Camera Raw 3D LUT profile.

    Args:
        table_md5: MD5 fingerprint (32 hex characters uppercase) of the uncompressed payload.
        table_base85: Custom Base85 encoded compressed data string.
        name: Profile display name in Lightroom Classic.
        group: Profile group (folder) in Lightroom Classic.
        profile_uuid: Optional 32-hex character UUID. If omitted, a random UUID is generated.

    Returns:
        The formatted XML text for the ``.xmp`` profile.
    """
    if profile_uuid is None:
        profile_uuid = uuid.uuid4().hex.upper()
    else:
        profile_uuid = profile_uuid.replace("-", "").upper()

    xmp_tk = "Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        "
    xml_lines = [
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="{xmp_tk}">',
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        '  <rdf:Description rdf:about=""',
        '    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"',
        '   crs:PresetType="Look"',
        '   crs:Cluster=""',
        f'   crs:UUID="{profile_uuid}"',
        '   crs:SupportsAmount="True"',
        '   crs:SupportsColor="True"',
        '   crs:SupportsMonochrome="True"',
        '   crs:SupportsHighDynamicRange="True"',
        '   crs:SupportsNormalDynamicRange="True"',
        '   crs:SupportsSceneReferred="True"',
        '   crs:SupportsOutputReferred="True"',
        '   crs:RequiresRGBTables="False"',
        '   crs:ShowInPresets="True"',
        '   crs:ShowInQuickActions="False"',
        '   crs:CameraModelRestriction=""',
        '   crs:Copyright=""',
        '   crs:ContactInfo=""',
        '   crs:Version="18.7"',
        '   crs:ProcessVersion="15.4"',
        '   crs:ConvertToGrayscale="False"',
        f'   crs:RGBTable="{table_md5}"',
        f'   crs:Table_{table_md5}="{table_base85}"',
        '   crs:HasSettings="True">',
        "   <crs:Name>",
        "    <rdf:Alt>",
        f"     <rdf:li xml:lang=\"x-default\">{name}</rdf:li>",
        "    </rdf:Alt>",
        "   </crs:Name>",
        "   <crs:ShortName>",
        "    <rdf:Alt>",
        '     <rdf:li xml:lang="x-default"/>',
        "    </rdf:Alt>",
        "   </crs:ShortName>",
        "   <crs:SortName>",
        "    <rdf:Alt>",
        '     <rdf:li xml:lang="x-default"/>',
        "    </rdf:Alt>",
        "   </crs:SortName>",
        "   <crs:Group>",
        "    <rdf:Alt>",
        f"     <rdf:li xml:lang=\"x-default\">{group}</rdf:li>",
        "    </rdf:Alt>",
        "   </crs:Group>",
        "   <crs:Description>",
        "    <rdf:Alt>",
        '     <rdf:li xml:lang="x-default"/>',
        "    </rdf:Alt>",
        "   </crs:Description>",
        "  </rdf:Description>",
        " </rdf:RDF>",
        "</x:xmpmeta>",
        "",
    ]
    return "\n".join(xml_lines)


def write_xmp_profile(
    path: str | Path,
    grid: NDArray[np.float64],
    size: int,
    name: str = "StealZaLut",
    group: str = "StealZaLut",
    progress_callback: Callable[[int], None] | None = None,
) -> Path:
    """Generate and write a Lightroom Classic profile (.xmp) from a 3D LUT grid.

    Args:
        path: Output file path. ``.xmp`` extension is automatically appended if missing.
        grid: Transformed 3D LUT grid of shape ``(size³, 3)`` with values in [0.0, 1.0].
        size: Resolution per axis of *grid*.
        name: Name of the profile in Lightroom Classic.
        group: Group / folder in Lightroom Classic profile browser.
        progress_callback: Optional callback receiving completion percentage (0–100).

    Returns:
        The resolved output :class:`~pathlib.Path`.
    """
    path = Path(path)
    if path.suffix.lower() != ".xmp":
        path = path.with_suffix(".xmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback is not None:
        progress_callback(10)

    # 1. Build uncompressed binary payload
    payload = build_xmp_table_payload(grid, size=size)

    if progress_callback is not None:
        progress_callback(30)

    # 2. Compute fingerprint (MD5 of uncompressed 196,652 bytes)
    table_md5 = hashlib.md5(payload).hexdigest().upper()

    # 3. Compress payload with zlib level 9
    compressed = zlib.compress(payload, level=9)

    if progress_callback is not None:
        progress_callback(50)

    # 4. Prepend 4-byte little-endian uncompressed size prefix
    framed = struct.pack("<I", len(payload)) + compressed

    # 5. Encode in custom Little-Endian Base85
    table_base85 = b85_encode_le(framed)

    if progress_callback is not None:
        progress_callback(75)

    # 6. Format XML document
    xmp_text = format_xmp_document(
        table_md5=table_md5,
        table_base85=table_base85,
        name=name,
        group=group,
    )

    # 7. Write to disk
    path.write_text(xmp_text, encoding="utf-8")

    if progress_callback is not None:
        progress_callback(100)

    return path


def read_cube_file(path: str | Path) -> tuple[NDArray[np.float64], int]:
    """Read a ``.cube`` file into a NumPy array and return its grid resolution.

    Args:
        path: Path to the ``.cube`` file.

    Returns:
        A tuple ``(grid, size)`` where ``grid`` has shape ``(size³, 3)`` and
        ``size`` is the integer grid resolution per axis.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid 3D LUT cube file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cube file not found: {path}")

    size: int | None = None
    samples: list[list[float]] = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("LUT_3D_SIZE"):
                parts = stripped.split()
                if len(parts) >= 2:
                    size = int(parts[1])
                continue
            # Ignore 1D sizes or domain keywords for 3D parsing
            if stripped.startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE")):
                continue

            parts = stripped.split()
            if len(parts) == 3:
                try:
                    samples.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue

    if size is None:
        raise ValueError(f"LUT_3D_SIZE keyword not found in {path}")

    expected_count = size**3
    if len(samples) != expected_count:
        raise ValueError(
            f"Expected {expected_count} samples for LUT size {size}, found {len(samples)} in {path}"
        )

    grid = np.array(samples, dtype=np.float64)
    return grid, size


def cube_to_xmp(
    cube_path: str | Path,
    xmp_path: str | Path | None = None,
    profile_name: str | None = None,
    group_name: str = "StealZaLut",
    progress_callback: Callable[[int], None] | None = None,
) -> Path:
    """Convert an existing ``.cube`` file directly into a Lightroom Classic profile (.xmp).

    Args:
        cube_path: Path to the source ``.cube`` file.
        xmp_path: Optional output path. Defaults to same name as ``cube_path`` with ``.xmp``.
        profile_name: Name of profile displayed in Lightroom. Defaults to file stem.
        group_name: Group name in Lightroom profile browser.
        progress_callback: Optional callback receiving progress percentage (0–100).

    Returns:
        The generated :class:`~pathlib.Path` of the ``.xmp`` profile.
    """
    cube_path = Path(cube_path)
    xmp_path = cube_path.with_suffix(".xmp") if xmp_path is None else Path(xmp_path)

    if profile_name is None:
        profile_name = cube_path.stem

    grid, size = read_cube_file(cube_path)
    return write_xmp_profile(
        path=xmp_path,
        grid=grid,
        size=size,
        name=profile_name,
        group=group_name,
        progress_callback=progress_callback,
    )


def read_xmp_profile(path: str | Path) -> tuple[NDArray[np.float64], int, str, str]:
    """Read a Lightroom Classic ``.xmp`` profile and extract its 3D LUT grid.

    Args:
        path: Path to the ``.xmp`` profile file.

    Returns:
        A tuple ``(grid, size, name, group)`` where ``grid`` has shape ``(32³, 3)``
        in standard .cube order (B slowest, G middle, R fastest) with RGB channels,
        and ``size`` is 32.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid Camera Raw Look profile.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"XMP file not found: {path}")

    text = path.read_text(encoding="utf-8")
    m_md5 = re.search(r'crs:RGBTable="([0-9A-F]{32})"', text)
    if not m_md5:
        raise ValueError(f"crs:RGBTable attribute not found in {path}")
    md5 = m_md5.group(1)

    m_table = re.search(r'crs:Table_' + md5 + r'="([^"]+)"', text)
    if not m_table:
        raise ValueError(f"crs:Table_{md5} data not found in {path}")
    b85_str = m_table.group(1)

    framed = b85_decode_le(b85_str)
    if len(framed) < 4:
        raise ValueError("Decoded Base85 data is truncated")

    payload = zlib.decompress(framed[4:])
    if len(payload) != UNCOMPRESSED_PAYLOAD_SIZE:
        raise ValueError(
            f"Decompressed payload size {len(payload)} != expected {UNCOMPRESSED_PAYLOAD_SIZE}"
        )

    # Table contains 98304 int16 values: shape (32, 32, 32, 3) in (R, G, B) order
    tbl_raw = np.frombuffer(payload[HEADER_SIZE : HEADER_SIZE + TABLE_DATA_BYTES], dtype="<i2")
    deltas = tbl_raw.reshape((TABLE_GRID_SIZE, TABLE_GRID_SIZE, TABLE_GRID_SIZE, 3))

    coords = np.linspace(0.0, 1.0, TABLE_GRID_SIZE, dtype=np.float64)
    r_g, g_g, b_g = np.meshgrid(coords, coords, coords, indexing="ij")
    ident = np.stack([r_g, g_g, b_g], axis=-1)

    # Reconstruct target RGB in (R, G, B) order
    target_rgb = ident + deltas.astype(np.float64) / 65535.0

    # Transpose back from (R, G, B) to standard .cube volume (B, G, R)
    vol_cube = np.transpose(target_rgb, (2, 1, 0, 3))
    grid = np.asarray(vol_cube.reshape((-1, 3)), dtype=np.float64)

    # Extract name and group
    m_name = re.search(r"<crs:Name>\s*<rdf:Alt>\s*<rdf:li[^>]*>([^<]+)</rdf:li>", text)
    name = m_name.group(1) if m_name else path.stem

    m_group = re.search(r"<crs:Group>\s*<rdf:Alt>\s*<rdf:li[^>]*>([^<]+)</rdf:li>", text)
    group = m_group.group(1) if m_group else "StealZaLut"

    return grid, TABLE_GRID_SIZE, name, group


def xmp_to_cube(
    xmp_path: str | Path,
    cube_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Convert an existing Lightroom Classic ``.xmp`` profile back into a ``.cube`` file.

    Args:
        xmp_path: Path to the source ``.xmp`` profile.
        cube_path: Optional output path. Defaults to same name with ``.cube``.
        title: Title written in the .cube header. Defaults to profile name.

    Returns:
        The generated :class:`~pathlib.Path` of the ``.cube`` file.
    """
    xmp_path = Path(xmp_path)
    cube_path = xmp_path.with_suffix(".cube") if cube_path is None else Path(cube_path)

    grid, size, name, _ = read_xmp_profile(xmp_path)
    if title is None:
        title = name

    from stealzalut.core.lut_export import write_cube_file

    return write_cube_file(cube_path, grid=grid, size=size, title=title)
