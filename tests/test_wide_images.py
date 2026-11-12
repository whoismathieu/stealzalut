"""The engine must analyse a picture's own colours, not their sRGB clipping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from stealzalut.core.color_transfer import fit_transfer
from stealzalut.core.colorspace import srgb_to_oklab
from stealzalut.core.look import fit_look
from stealzalut.core.palette import ANALYSIS_MAX_SIZE, analysis_oklab, analyze_zones, colour_stats
from stealzalut.core.rgb_spaces import DISPLAY_P3
from stealzalut.utils import image_io
from stealzalut.utils.image_io import (
    WIDE_ANALYSIS_SIZE,
    WideImage,
    create_thumbnail,
    load_image_rgb,
)
from tests.test_look import _green_dusk, _neutral_street


def _icc() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _neon() -> np.ndarray:
    """Grey picture with a pure red and a pure green patch (the gamut's corners)."""
    img = np.full((60, 80, 3), 90, dtype=np.uint8)
    img[10:30, 10:40] = (255, 0, 0)
    img[35:55, 40:70] = (0, 255, 0)
    return img


def _as_p3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every profile read as Display P3 (Pillow cannot build a P3 profile)."""
    monkeypatch.setattr(ImageCms, "getProfileDescription", lambda profile: "Display P3")


def _chroma(lab: np.ndarray) -> np.ndarray:
    return np.hypot(lab[..., 1], lab[..., 2])


# ── WideImage ────────────────────────────────────────────────────────────────


def test_sizes_agree() -> None:
    assert WIDE_ANALYSIS_SIZE == ANALYSIS_MAX_SIZE


def test_wide_image_is_the_srgb_picture() -> None:
    srgb = _neon()
    image = WideImage(srgb, np.zeros((60, 80, 3)), "Display P3")
    assert isinstance(image, np.ndarray) and image.dtype == np.uint8
    assert np.array_equal(np.asarray(image), srgb)
    assert image.analysis_lab is not None and image.source_space == "Display P3"


def test_derived_arrays_drop_the_analysis() -> None:
    """A crop or a copy is other pixels: the analysis copy no longer describes it."""
    image = WideImage(_neon(), np.zeros((60, 80, 3)), "Display P3")
    assert image[:10].analysis_lab is None
    assert image.copy().analysis_lab is None
    assert image.reshape(-1, 3).analysis_lab is None
    assert getattr(create_thumbnail(image, 32), "analysis_lab", None) is None
    assert getattr(np.asarray(image), "analysis_lab", None) is None


# ── Loading ──────────────────────────────────────────────────────────────────


def test_plain_file_has_no_wide_analysis(tmp_path: Path) -> None:
    path = tmp_path / "plain.png"
    Image.fromarray(_neon()).save(path)
    image = load_image_rgb(path)
    assert isinstance(image, WideImage) and image.analysis_lab is None
    assert np.array_equal(np.asarray(image), _neon())


def test_srgb_tagged_file_has_no_wide_analysis(tmp_path: Path) -> None:
    path = tmp_path / "tagged.png"
    Image.fromarray(_neon()).save(path, icc_profile=_icc())
    assert load_image_rgb(path).analysis_lab is None


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_p3_file_keeps_its_colours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    path = tmp_path / f"p3{suffix}"
    Image.fromarray(_neon()).save(path, icc_profile=_icc(), quality=98)
    _as_p3(monkeypatch)
    image = load_image_rgb(path)
    assert image.source_space == "Display P3"
    wide = image.analysis_lab
    assert wide is not None and wide.shape == create_thumbnail(image, 256).shape

    # The red patch is P3 red: more saturated than anything sRGB can hold
    p3_red = DISPLAY_P3.to_oklab(np.array([[1.0, 0.0, 0.0]]))[0]
    srgb_red = srgb_to_oklab(np.array([[1.0, 0.0, 0.0]]))[0]
    patch = wide[15:25, 15:35].reshape(-1, 3).mean(axis=0)
    assert np.allclose(patch, p3_red, atol=0.02)
    assert _chroma(patch) > _chroma(srgb_red) + 0.02
    # ... while the displayed picture stays within sRGB
    shown = srgb_to_oklab(np.asarray(image)[15:25, 15:35].reshape(-1, 3) / 255.0)
    assert _chroma(shown).max() <= _chroma(srgb_red) + 0.01


def test_unknown_profile_falls_back_to_srgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "odd.png"
    Image.fromarray(_neon()).save(path, icc_profile=_icc())
    monkeypatch.setattr(ImageCms, "getProfileDescription", lambda profile: "Scanner X")
    assert load_image_rgb(path).analysis_lab is None


def test_a_failing_wide_read_still_loads_the_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "p3.png"
    Image.fromarray(_neon()).save(path, icc_profile=_icc())

    def boom(path: Path) -> None:
        raise OSError("unreadable")

    monkeypatch.setattr(image_io, "_open_for_analysis", boom)
    image = load_image_rgb(path)
    assert image.analysis_lab is None and image.shape == (60, 80, 3)


# ── The engine reads the wide colours ────────────────────────────────────────


def test_analysis_oklab_prefers_the_wide_colours() -> None:
    srgb = _neon()
    plain_lab, h, w = analysis_oklab(srgb)
    assert (h, w) == (60, 80) and plain_lab.shape == (4800, 3)
    wide_lab = plain_lab.reshape(60, 80, 3) * np.array([1.0, 1.4, 1.4])
    got, h2, w2 = analysis_oklab(WideImage(srgb, wide_lab))
    assert (h2, w2) == (60, 80)
    assert np.allclose(got, wide_lab.reshape(-1, 3))


def test_colour_stats_and_zones_see_the_wide_colours() -> None:
    srgb = _neutral_street()
    lab, h, w = analysis_oklab(srgb)
    boosted = lab.reshape(h, w, 3) * np.array([1.0, 3.0, 3.0])
    wide = WideImage(srgb, boosted)
    assert colour_stats(wide).colourful > colour_stats(srgb).colourful + 0.05
    plain_zones, wide_zones = analyze_zones(srgb, 6), analyze_zones(wide, 6)
    assert wide_zones.labels.shape == plain_zones.labels.shape
    plain_c = max(float(_chroma(z.mean)) for z in plain_zones.zones)
    wide_c = max(float(_chroma(z.mean)) for z in wide_zones.zones)
    assert wide_c > plain_c * 1.5


def test_look_reads_the_reference_in_wide_gamut() -> None:
    """A reference whose colours exceed sRGB gives a stronger cast than its clipping."""
    source, reference = _neutral_street(), _green_dusk()
    lab, h, w = analysis_oklab(reference)
    vivid = lab.reshape(h, w, 3) * np.array([1.0, 1.5, 1.5])
    plain = fit_look(source, reference)
    wide = fit_look(source, WideImage(reference, vivid))
    assert np.abs(wide.cast).mean() > np.abs(plain.cast).mean() * 1.2
    assert np.array_equal(wide.tone_y, plain.tone_y)  # lightness was not touched


@pytest.mark.parametrize("method", ["look", "zones", "ambiance", "oklab"])
def test_oklab_methods_use_the_wide_colours(method: str) -> None:
    source, reference = _neutral_street(), _green_dusk()
    lab, h, w = analysis_oklab(reference)
    vivid = WideImage(reference, lab.reshape(h, w, 3) * np.array([1.0, 1.6, 1.6]))
    grey = np.array([[0.25, 0.25, 0.25]])
    plain_out = srgb_to_oklab(fit_transfer(source, reference, method=method).apply(grey))
    wide_out = srgb_to_oklab(fit_transfer(source, vivid, method=method).apply(grey))
    assert _chroma(wide_out)[0] > _chroma(plain_out)[0] + 0.003


def test_plain_arrays_behave_as_before() -> None:
    """No wide data: exactly the sRGB analysis (every existing test relies on it)."""
    source, reference = _neutral_street(), _green_dusk()
    a = fit_look(source, reference)
    b = fit_look(WideImage(source), WideImage(reference))
    assert np.array_equal(a.field, b.field) and np.array_equal(a.cast, b.cast)


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Display P3", "display-p3"),
        ("DISPLAY P3", "display-p3"),
        ("DCI-P3 D65", "display-p3"),
        ("Adobe RGB (1998)", "adobe-rgb"),
        ("AdobeRGB1998", "adobe-rgb"),
        ("adobe wide", "adobe-rgb"),
        ("ProPhoto RGB", "prophoto"),
        ("ROMM RGB: ISO 22028-2:2013", "prophoto"),
        ("sRGB IEC61966-2.1", None),
        ("", None),
        ("Scanner X", None),
    ],
)
def test_profile_names_are_matched_by_substring(description: str, expected: str | None) -> None:
    space = image_io._known_space(description)
    assert (None if space is None else space.key) == expected
