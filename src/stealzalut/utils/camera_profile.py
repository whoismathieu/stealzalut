"""Adobe's camera profile (.dcp) of a DNG file, and the colour pipeline it defines.

Lightroom and Camera Raw render a raw file through a *camera profile*.  The
one named « Adobe Standard » is a ``.dcp`` file Adobe installs on the machine
for each camera it knows.  It holds what a neutral development needs and what
no generic library carries:

- two colour matrices and two forward matrices (camera → XYZ D50), one per
  calibration illuminant, mixed according to the white balance of the shot;
- a hue/saturation table applied before the exposure, and a « look » table
  applied after it, both indexed in the HSV of linear ProPhoto RGB;
- optionally a tone curve and an exposure offset.

This module reads that file and reproduces the colour steps of the Adobe DNG
SDK (``dng_color_spec``, ``dng_hue_sat_map``), so that a DNG developed here
gets the colours Lightroom gives it on Adobe Standard.  The profile is read
where Adobe installed it and never copied: on a machine without it, the
caller falls back on the file's own matrices.

No Qt, no image decoding: only arithmetic on arrays.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

# ── TIFF / DNG / DCP tags ────────────────────────────────────────────────
TAG_SUB_IFDS = 330
TAG_UNIQUE_CAMERA_MODEL = 50708
TAG_COLOR_MATRIX_1 = 50721
TAG_COLOR_MATRIX_2 = 50722
TAG_ANALOG_BALANCE = 50727
TAG_AS_SHOT_NEUTRAL = 50728
TAG_BASELINE_EXPOSURE = 50730
TAG_CALIBRATION_ILLUMINANT_1 = 50778
TAG_CALIBRATION_ILLUMINANT_2 = 50779
TAG_PROFILE_NAME = 50936
TAG_HUE_SAT_MAP_DIMS = 50937
TAG_HUE_SAT_MAP_DATA_1 = 50938
TAG_HUE_SAT_MAP_DATA_2 = 50939
TAG_PROFILE_TONE_CURVE = 50940
TAG_FORWARD_MATRIX_1 = 50964
TAG_FORWARD_MATRIX_2 = 50965
TAG_LOOK_TABLE_DIMS = 50981
TAG_LOOK_TABLE_DATA = 50982
TAG_HUE_SAT_MAP_ENCODING = 51107
TAG_LOOK_TABLE_ENCODING = 51108
TAG_BASELINE_EXPOSURE_OFFSET = 51109
TAG_PROFILE_GAIN_TABLE_MAP = 52525

#: numpy type of each TIFF field type (rationals are read as pairs).
_FIELD_TYPES = {
    1: "u1",
    3: "u2",
    4: "u4",
    5: "u4",
    6: "i1",
    7: "u1",
    8: "i2",
    9: "i4",
    10: "i4",
    11: "f4",
    12: "f8",
}
#: Largest field read (the look table of a profile is about 170 kB).
_MAX_FIELD_BYTES = 8_000_000

#: Colour temperature (K) of the EXIF light sources a profile is calibrated for.
ILLUMINANT_KELVIN = {
    1: 5500.0,  # daylight
    2: 4150.0,  # fluorescent
    3: 2850.0,  # tungsten
    4: 5500.0,  # flash
    9: 5500.0,  # fine weather
    10: 6500.0,  # cloudy
    11: 7500.0,  # shade
    12: 6400.0,  # daylight fluorescent
    13: 5050.0,  # day white fluorescent
    14: 4150.0,  # cool white fluorescent
    15: 3525.0,  # white fluorescent
    17: 2850.0,  # standard light A
    18: 4874.0,  # standard light B
    19: 6774.0,  # standard light C
    20: 5500.0,  # D55
    21: 6500.0,  # D65
    22: 7500.0,  # D75
    23: 5000.0,  # D50
    24: 3200.0,  # ISO studio tungsten
}

#: XYZ (D50) → linear ProPhoto RGB (ROMM), the working space of the pipeline.
XYZ_D50_TO_PROPHOTO: F64 = np.array(
    [
        [1.3459433, -0.2556075, -0.0511118],
        [-0.5445989, 1.5081673, 0.0205351],
        [0.0, 0.0, 1.2118128],
    ]
)
#: xy of D50, where the search for the white point starts.
_D50_XY = (0.3457, 0.3585)


def read_tiff_tags(path: Path, wanted: frozenset[int]) -> dict[int, Any]:
    """Values of the *wanted* tags of a TIFF-structured file (DNG or DCP).

    The first directory and its sub-directories are read; the first value met
    wins.  A text field is returned as ``str``, a numeric one as a float64 or
    integer array (rationals already divided).

    Raises:
        ValueError: The file is not TIFF-structured, or is truncated.
        OSError: The file cannot be read.
    """
    found: dict[int, Any] = {}
    with path.open("rb") as handle:
        header = handle.read(8)
        order = {b"II": "<", b"MM": ">"}.get(header[:2])
        # 42 is TIFF and DNG; 0x4352 ("CR") is the magic number of a .dcp profile
        if order is None or struct.unpack(order + "H", header[2:4])[0] not in (42, 0x4352):
            raise ValueError(f"{path.name} is not a TIFF, DNG or DCP file")
        size = path.stat().st_size

        def directory(offset: int, depth: int) -> None:
            if not 8 <= offset < size:
                return
            handle.seek(offset)
            (count,) = struct.unpack(order + "H", handle.read(2))
            entries = handle.read(12 * min(count, 1024))
            subs: list[int] = []
            for index in range(len(entries) // 12):
                tag, kind, number, raw = struct.unpack_from(order + "HHI4s", entries, 12 * index)
                if tag != TAG_SUB_IFDS and (tag not in wanted or tag in found):
                    continue
                value = field(kind, number, raw)
                if value is None:
                    continue
                if tag == TAG_SUB_IFDS:
                    subs = [int(v) for v in np.atleast_1d(value)]
                else:
                    found[tag] = value
            if depth < 2:
                for sub in subs:
                    directory(sub, depth + 1)

        def field(kind: int, number: int, raw: bytes) -> Any:
            if kind == 2:
                length, dtype = number, None
            elif kind in _FIELD_TYPES:
                dtype = np.dtype(order + _FIELD_TYPES[kind])
                length = dtype.itemsize * number * (2 if kind in (5, 10) else 1)
            else:
                return None
            if length > _MAX_FIELD_BYTES:
                return None
            if length <= 4:
                data = raw[:length]
            else:
                handle.seek(struct.unpack(order + "I", raw)[0])
                data = handle.read(length)
            if len(data) != length:
                raise ValueError(f"{path.name} is truncated")
            if dtype is None:
                return data.split(b"\0", 1)[0].decode("latin-1")
            values = np.frombuffer(data, dtype=dtype)
            if kind in (5, 10):
                pairs = values.reshape(-1, 2).astype(np.float64)
                safe = np.where(pairs[:, 1] == 0, 1.0, pairs[:, 1])
                return pairs[:, 0] / safe
            if kind in (11, 12):
                return values.astype(np.float64)
            return values.astype(np.int64)

        directory(struct.unpack(order + "I", header[4:8])[0], 0)
    return found


# ── What a DNG says about its own colour ─────────────────────────────────


@dataclass(frozen=True, slots=True)
class DngColour:
    """Colour description stored in a DNG file.

    Attributes:
        camera_model: ``UniqueCameraModel``, what Adobe matches a profile on.
        analog_balance: Gain already applied to each camera channel.
        as_shot_neutral: Camera values of a neutral grey under the shot's light.
        baseline_exposure: Stops to add so that the picture is normally exposed.
        gain_table_map: The file carries a ``ProfileGainTableMap`` (Apple ProRAW).
    """

    camera_model: str
    analog_balance: F64
    as_shot_neutral: F64
    baseline_exposure: float
    gain_table_map: bool


_DNG_TAGS = frozenset(
    {
        TAG_UNIQUE_CAMERA_MODEL,
        TAG_ANALOG_BALANCE,
        TAG_AS_SHOT_NEUTRAL,
        TAG_BASELINE_EXPOSURE,
        TAG_PROFILE_GAIN_TABLE_MAP,
    }
)


def _vector(tags: dict[int, Any], tag: int, length: int) -> F64 | None:
    value = tags.get(tag)
    if not isinstance(value, np.ndarray) or value.size != length:
        return None
    out = value.astype(np.float64)
    return out if bool(np.all(np.isfinite(out))) else None


def read_dng_colour(path: Path) -> DngColour | None:
    """Colour tags of the DNG at *path*, or ``None`` when it is not one or lacks them.

    A file without ``AsShotNeutral`` (white balance stored as a chromaticity,
    or not at all) gives ``None``: the profile pipeline needs the neutral.
    """
    if path.suffix.lower() != ".dng":
        return None
    try:
        tags = read_tiff_tags(path, _DNG_TAGS)
    except (OSError, ValueError, struct.error) as exc:
        logger.debug("Could not read the colour tags of %s: %s", path, exc)
        return None
    model = tags.get(TAG_UNIQUE_CAMERA_MODEL)
    neutral = _vector(tags, TAG_AS_SHOT_NEUTRAL, 3)
    if not isinstance(model, str) or not model or neutral is None or bool(np.any(neutral <= 0)):
        return None
    balance = _vector(tags, TAG_ANALOG_BALANCE, 3)
    if balance is None or bool(np.any(balance <= 0)):
        balance = np.ones(3)
    exposure = _vector(tags, TAG_BASELINE_EXPOSURE, 1)
    return DngColour(
        camera_model=model.strip(),
        analog_balance=balance,
        as_shot_neutral=neutral,
        baseline_exposure=float(exposure[0]) if exposure is not None else 0.0,
        gain_table_map=TAG_PROFILE_GAIN_TABLE_MAP in tags,
    )


# ── The profile ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HsvTable:
    """A hue / saturation / value table of a profile.

    Attributes:
        data: ``(values, hues, saturations, 3)``: hue shift in degrees,
            saturation factor, value factor.
        srgb_value: The value axis is indexed through the sRGB curve.
    """

    data: F32
    srgb_value: bool = False


@dataclass(frozen=True, slots=True)
class CameraProfile:
    """What a ``.dcp`` file holds, for cameras calibrated under one or two lights.

    Attributes:
        name: ``ProfileName`` (« Adobe Standard »).
        camera_model: ``UniqueCameraModel`` the profile was made for.
        kelvin: Temperature of each calibration light (one or two).
        color_matrices: XYZ → camera, one per light.
        forward_matrices: White-balanced camera → XYZ D50, one per light.
        hue_sat_maps: Table applied before the exposure, one per light, or none.
        look_table: Table applied after the exposure, or ``None``.
        tone_curve: ``(n, 2)`` points of the profile's own curve, or ``None``
            (the default curve of the host then applies).
        baseline_exposure_offset: Stops added to the file's BaselineExposure.
    """

    name: str
    camera_model: str
    kelvin: tuple[float, ...]
    color_matrices: tuple[F64, ...]
    forward_matrices: tuple[F64, ...]
    hue_sat_maps: tuple[HsvTable, ...]
    look_table: HsvTable | None
    tone_curve: F64 | None
    baseline_exposure_offset: float


_PROFILE_TAGS = frozenset(
    {
        TAG_UNIQUE_CAMERA_MODEL,
        TAG_COLOR_MATRIX_1,
        TAG_COLOR_MATRIX_2,
        TAG_CALIBRATION_ILLUMINANT_1,
        TAG_CALIBRATION_ILLUMINANT_2,
        TAG_PROFILE_NAME,
        TAG_HUE_SAT_MAP_DIMS,
        TAG_HUE_SAT_MAP_DATA_1,
        TAG_HUE_SAT_MAP_DATA_2,
        TAG_PROFILE_TONE_CURVE,
        TAG_FORWARD_MATRIX_1,
        TAG_FORWARD_MATRIX_2,
        TAG_LOOK_TABLE_DIMS,
        TAG_LOOK_TABLE_DATA,
        TAG_HUE_SAT_MAP_ENCODING,
        TAG_LOOK_TABLE_ENCODING,
        TAG_BASELINE_EXPOSURE_OFFSET,
    }
)


def _table(
    tags: dict[int, Any], dims_tag: int, data_tag: int, encoding_tag: int
) -> HsvTable | None:
    dims, data = tags.get(dims_tag), tags.get(data_tag)
    if not isinstance(dims, np.ndarray) or not isinstance(data, np.ndarray) or dims.size != 3:
        return None
    hues, sats, values = (int(v) for v in dims)
    if hues < 1 or sats < 2 or values < 1 or data.size != hues * sats * values * 3:
        return None
    encoding = tags.get(encoding_tag)
    srgb = isinstance(encoding, np.ndarray) and encoding.size == 1 and int(encoding[0]) == 1
    return HsvTable(data.astype(np.float32).reshape(values, hues, sats, 3), bool(srgb))


def read_camera_profile(path: Path) -> CameraProfile | None:
    """The profile stored at *path*, or ``None`` when the pipeline cannot use it.

    A profile is usable when it has, for each calibration light, a colour
    matrix and a forward matrix, and a light whose temperature is known.
    """
    try:
        tags = read_tiff_tags(path, _PROFILE_TAGS)
    except (OSError, ValueError, struct.error) as exc:
        logger.debug("Could not read the profile %s: %s", path, exc)
        return None
    lights = []
    for light, colour, forward, table in (
        (TAG_CALIBRATION_ILLUMINANT_1, TAG_COLOR_MATRIX_1, TAG_FORWARD_MATRIX_1, 1),
        (TAG_CALIBRATION_ILLUMINANT_2, TAG_COLOR_MATRIX_2, TAG_FORWARD_MATRIX_2, 2),
    ):
        code = tags.get(light)
        kelvin = ILLUMINANT_KELVIN.get(int(code[0])) if isinstance(code, np.ndarray) else None
        cm, fm = _vector(tags, colour, 9), _vector(tags, forward, 9)
        if kelvin is None or cm is None or fm is None:
            continue
        data_tag = TAG_HUE_SAT_MAP_DATA_1 if table == 1 else TAG_HUE_SAT_MAP_DATA_2
        hsm = _table(tags, TAG_HUE_SAT_MAP_DIMS, data_tag, TAG_HUE_SAT_MAP_ENCODING)
        lights.append((kelvin, cm.reshape(3, 3), fm.reshape(3, 3), hsm))
    if not lights:
        return None
    lights.sort(key=lambda item: item[0])
    maps = tuple(item[3] for item in lights)
    curve = tags.get(TAG_PROFILE_TONE_CURVE)
    offset = _vector(tags, TAG_BASELINE_EXPOSURE_OFFSET, 1)
    return CameraProfile(
        name=str(tags.get(TAG_PROFILE_NAME, path.stem)),
        camera_model=str(tags.get(TAG_UNIQUE_CAMERA_MODEL, "")).strip(),
        kelvin=tuple(item[0] for item in lights),
        color_matrices=tuple(item[1] for item in lights),
        forward_matrices=tuple(item[2] for item in lights),
        hue_sat_maps=cast("tuple[HsvTable, ...]", maps) if all(maps) else (),
        look_table=_table(tags, TAG_LOOK_TABLE_DIMS, TAG_LOOK_TABLE_DATA, TAG_LOOK_TABLE_ENCODING),
        tone_curve=(
            curve.reshape(-1, 2)
            if isinstance(curve, np.ndarray) and curve.size >= 4 and curve.size % 2 == 0
            else None
        ),
        baseline_exposure_offset=float(offset[0]) if offset is not None else 0.0,
    )


def adobe_profile_folders() -> list[Path]:
    """Folders where Adobe installs its « Adobe Standard » camera profiles."""
    relative = Path("Adobe") / "CameraRaw" / "CameraProfiles" / "Adobe Standard"
    if sys.platform == "win32":
        roots = [Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))]
        if os.environ.get("APPDATA"):
            roots.append(Path(os.environ["APPDATA"]))
    elif sys.platform == "darwin":
        roots = [
            Path("/Library/Application Support"),
            Path.home() / "Library" / "Application Support",
        ]
    else:
        roots = []
    return [root / relative for root in roots]


def find_adobe_standard(
    camera_model: str, folders: list[Path] | None = None
) -> CameraProfile | None:
    """Adobe's « Adobe Standard » profile for *camera_model*, if it is installed.

    Adobe names the file after the camera, sometimes with the maker in front
    (« Apple iPhone16,2 back telephoto camera Adobe Standard.dcp »): files
    whose name ends with the model are tried, and the one whose own
    ``UniqueCameraModel`` is the camera's is kept.
    """
    wanted = camera_model.strip().casefold()
    if not wanted:
        return None
    for folder in adobe_profile_folders() if folders is None else folders:
        if not folder.is_dir():
            continue
        suffix = f"{wanted} adobe standard.dcp"
        for path in sorted(folder.glob("*.dcp")):
            if not path.name.casefold().endswith(suffix):
                continue
            profile = read_camera_profile(path)
            if profile is not None and profile.camera_model.casefold() == wanted:
                logger.debug("Camera profile: %s", path)
                return profile
    return None


# ── White point and matrices (DNG specification, chapter 6) ──────────────

_LOCUS_KELVIN: F64 = np.arange(1500.0, 25001.0, dtype=np.float64)


def _planckian_uv(kelvin: F64) -> tuple[F64, F64]:
    """CIE 1960 uv of the black body at *kelvin* (Krystek's approximation)."""
    t = kelvin
    u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) / (
        1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t
    )
    v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) / (
        1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t
    )
    return u, v


_LOCUS_UV = _planckian_uv(_LOCUS_KELVIN)


def correlated_temperature(x: float, y: float) -> float:
    """Correlated colour temperature (K) of the chromaticity ``(x, y)``."""
    denominator = -2.0 * x + 12.0 * y + 3.0
    u, v = 4.0 * x / denominator, 6.0 * y / denominator
    distance = (_LOCUS_UV[0] - u) ** 2 + (_LOCUS_UV[1] - v) ** 2
    return float(_LOCUS_KELVIN[int(np.argmin(distance))])


def _first_light_share(profile: CameraProfile, kelvin: float) -> float:
    """Weight of the warmer calibration light at *kelvin* (linear in 1 / T)."""
    if len(profile.kelvin) < 2:
        return 1.0
    warm, cold = profile.kelvin[0], profile.kelvin[-1]
    if kelvin <= warm:
        return 1.0
    if kelvin >= cold:
        return 0.0
    return float((1.0 / kelvin - 1.0 / cold) / (1.0 / warm - 1.0 / cold))


def _mixed(matrices: tuple[F64, ...], share: float) -> F64:
    if len(matrices) < 2:
        return matrices[0]
    return share * matrices[0] + (1.0 - share) * matrices[-1]


@dataclass(frozen=True, slots=True)
class Development:
    """The colour steps of one file under one profile, ready to be applied.

    Attributes:
        camera_to_prophoto: 3×3, camera values → linear ProPhoto RGB, the
            neutral of the shot landing on white.
        hue_sat_map: Table applied before the exposure, or ``None``.
        look_table: Table applied after the exposure, or ``None``.
        tone_curve: ``(n, 2)`` points of the profile's curve, or ``None``.
        exposure_ev: BaselineExposure of the file plus the profile's offset.
        kelvin: Colour temperature found for the shot (what the host displays).
        profile_name: Name of the profile, for the log and the interface.
    """

    camera_to_prophoto: F64
    hue_sat_map: HsvTable | None
    look_table: HsvTable | None
    tone_curve: F64 | None
    exposure_ev: float
    kelvin: float
    profile_name: str


def plan_development(profile: CameraProfile, dng: DngColour) -> Development:
    """Resolve *profile* for the white balance stored in *dng*.

    The white point is the chromaticity whose camera response is the file's
    neutral; since the colour matrix itself depends on the temperature of
    that point, it is found by iteration, as the DNG specification describes.
    The forward matrix then maps the white-balanced camera values to XYZ D50.

    Raises:
        ValueError: The matrices cannot be inverted for this neutral.
    """
    balance = np.diag(dng.analog_balance)
    x, y = _D50_XY
    share = 1.0
    for _ in range(30):
        share = _first_light_share(profile, correlated_temperature(x, y))
        xyz_to_camera = balance @ _mixed(profile.color_matrices, share)
        try:
            xyz = np.linalg.solve(xyz_to_camera, dng.as_shot_neutral)
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular colour matrix") from exc
        total = float(xyz.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("the neutral of the file has no white point under this profile")
        new_x, new_y = float(xyz[0] / total), float(xyz[1] / total)
        moved = abs(new_x - x) + abs(new_y - y)
        x, y = new_x, new_y
        if moved < 1e-7:
            break
    kelvin = correlated_temperature(x, y)
    share = _first_light_share(profile, kelvin)

    reference_neutral = dng.as_shot_neutral / dng.analog_balance
    reference_neutral = reference_neutral / reference_neutral.max()
    camera_to_xyz = (
        _mixed(profile.forward_matrices, share)
        @ np.diag(1.0 / reference_neutral)
        @ np.diag(1.0 / dng.analog_balance)
    )
    hue_sat_map = None
    if profile.hue_sat_maps:
        first, last = profile.hue_sat_maps[0], profile.hue_sat_maps[-1]
        hue_sat_map = HsvTable(
            (share * first.data + (1.0 - share) * last.data).astype(np.float32), first.srgb_value
        )
    return Development(
        camera_to_prophoto=XYZ_D50_TO_PROPHOTO @ camera_to_xyz,
        hue_sat_map=hue_sat_map,
        look_table=profile.look_table,
        tone_curve=profile.tone_curve,
        exposure_ev=dng.baseline_exposure + profile.baseline_exposure_offset,
        kelvin=kelvin,
        profile_name=profile.name,
    )


# ── Hue / saturation / value tables (dng_hue_sat_map) ────────────────────


def apply_hsv_table(rgb: F32, table: HsvTable) -> F32:
    """Apply a profile table to linear ProPhoto RGB ``(N, 3)`` in ``[0, 1]``.

    Each colour is taken to HSV, the table is read at its hue (which wraps),
    saturation and value by trilinear interpolation, the hue is shifted, the
    saturation and the value are scaled (and kept at 1 at most).
    """
    values, hues, sats, _ = table.data.shape
    flat = table.data.reshape(-1, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    top = rgb.max(axis=1)
    span = top - rgb.min(axis=1)
    safe = np.where(span > 0.0, span, np.float32(1.0))
    hue = np.where(
        top == red,
        (green - blue) / safe,
        np.where(top == green, 2.0 + (blue - red) / safe, 4.0 + (red - green) / safe),
    )
    hue = np.where(span > 0.0, np.mod(hue, 6.0), 0.0).astype(np.float32)
    sat = np.where(top > 0.0, span / np.where(top > 0.0, top, np.float32(1.0)), 0.0).astype(
        np.float32
    )

    hue_pos = hue * np.float32(hues / 6.0)
    hue_low = np.floor(hue_pos).astype(np.int64)
    hue_part = (hue_pos - hue_low).astype(np.float32)
    hue_low %= hues
    hue_high = (hue_low + 1) % hues
    sat_pos = sat * np.float32(sats - 1)
    sat_low = np.minimum(sat_pos.astype(np.int64), sats - 2)
    sat_part = (sat_pos - sat_low).astype(np.float32)
    if values > 1:
        level = top
        if table.srgb_value:
            level = np.where(
                top <= 0.0031308, 12.92 * top, 1.055 * np.power(top, 1.0 / 2.4) - 0.055
            ).astype(np.float32)
        value_pos = level * np.float32(values - 1)
        value_low = np.minimum(value_pos.astype(np.int64), values - 2)
        value_part = (value_pos - value_low).astype(np.float32)
        planes = [(value_low, 1.0 - value_part), (value_low + 1, value_part)]
    else:
        planes = [(np.zeros(len(rgb), dtype=np.int64), np.ones(len(rgb), dtype=np.float32))]

    change = np.zeros((len(rgb), 3), dtype=np.float32)
    for value_index, value_weight in planes:
        for hue_index, hue_weight in ((hue_low, 1.0 - hue_part), (hue_high, hue_part)):
            base = (value_index * hues + hue_index) * sats
            for sat_index, sat_weight in ((sat_low, 1.0 - sat_part), (sat_low + 1, sat_part)):
                weight = (value_weight * hue_weight * sat_weight).astype(np.float32)
                change += weight[:, None] * flat[base + sat_index]

    hue = np.mod(hue + change[:, 0] * np.float32(6.0 / 360.0), 6.0)
    sat = np.minimum(sat * change[:, 1], 1.0)
    top = np.minimum(top * change[:, 2], 1.0)

    sector = np.minimum(hue.astype(np.int64), 5)
    part = hue - sector
    low = top * (1.0 - sat)
    falling = top * (1.0 - sat * part)
    rising = top * (1.0 - sat * (1.0 - part))
    out = np.empty_like(rgb)
    out[:, 0] = np.choose(sector, [top, falling, low, low, rising, top])
    out[:, 1] = np.choose(sector, [rising, top, top, falling, low, low])
    out[:, 2] = np.choose(sector, [low, low, rising, top, top, falling])
    return out
