"""The picture the engine sees must be the picture the user sees.

Colour profiles (phones save Display P3) and full-resolution application of
the LUT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageCms

from stealzalut.core.lut_apply import FULL_CHUNK_ROWS, apply_lut, apply_lut_full
from stealzalut.core.lut_engine import generate_neutral_grid
from stealzalut.utils import image_io
from stealzalut.utils.image_io import _has_wide_gamut_profile, _pil_to_srgb, load_image_bgr


def _srgb_icc() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _picture() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), "RGB")


# ── Colour profiles ──────────────────────────────────────────────────────────


def test_image_without_profile_is_untouched() -> None:
    img = _picture()
    assert _pil_to_srgb(img) is img


def test_srgb_profile_is_untouched() -> None:
    img = _picture()
    img.info["icc_profile"] = _srgb_icc()
    assert _pil_to_srgb(img) is img


def test_other_profiles_are_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile that is not sRGB goes through the colour management module."""
    img = _picture()
    img.info["icc_profile"] = _srgb_icc()
    calls: list[Any] = []

    def convert(image: Image.Image, source: Any, target: Any, **kwargs: Any) -> Image.Image:
        calls.append(kwargs)
        return Image.new("RGB", image.size, (1, 2, 3))

    monkeypatch.setattr(ImageCms, "getProfileDescription", lambda profile: "Display P3")
    monkeypatch.setattr(ImageCms, "profileToProfile", convert)
    out = _pil_to_srgb(img)
    assert calls == [{"outputMode": "RGB"}]
    assert out.getpixel((0, 0)) == (1, 2, 3)


def test_broken_profile_does_not_break_loading() -> None:
    img = _picture()
    img.info["icc_profile"] = b"not a profile"
    assert _pil_to_srgb(img) is img


def test_wide_gamut_files_are_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain, tagged = tmp_path / "plain.jpg", tmp_path / "tagged.jpg"
    _picture().save(plain)
    _picture().save(tagged, icc_profile=_srgb_icc())
    assert not _has_wide_gamut_profile(plain)
    assert not _has_wide_gamut_profile(tagged)  # sRGB is not wide gamut
    monkeypatch.setattr(ImageCms, "getProfileDescription", lambda profile: "Display P3")
    assert _has_wide_gamut_profile(tagged)
    assert not _has_wide_gamut_profile(plain)  # still no profile at all


def test_wide_gamut_file_is_loaded_through_the_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "p3.png"
    _picture().save(path, icc_profile=_srgb_icc())
    monkeypatch.setattr(ImageCms, "getProfileDescription", lambda profile: "Display P3")
    seen: list[bool] = []
    original = image_io._pil_to_srgb

    def spy(img: Any) -> Any:
        seen.append(True)
        return original(img)

    monkeypatch.setattr(image_io, "_pil_to_srgb", spy)
    loaded = load_image_bgr(path)
    assert seen == [True]
    assert loaded.shape == (24, 32, 3)


def test_plain_file_still_loads_identically(tmp_path: Path) -> None:
    path = tmp_path / "plain.png"
    picture = _picture()
    picture.save(path)
    assert np.array_equal(load_image_bgr(path)[..., ::-1], np.asarray(picture))


# ── Full-resolution LUT ──────────────────────────────────────────────────────


def test_apply_lut_full_matches_apply_lut() -> None:
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (FULL_CHUNK_ROWS * 2 + 37, 40, 3), dtype=np.uint8)
    grid = np.clip(generate_neutral_grid(17) ** 1.3 + 0.02, 0.0, 1.0)
    assert np.array_equal(apply_lut_full(image, grid, 17), apply_lut(image, grid, 17))


def test_apply_lut_full_checks_the_grid() -> None:
    with pytest.raises(ValueError):
        apply_lut_full(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((10, 3)), 17)
