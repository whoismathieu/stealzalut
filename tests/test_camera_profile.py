"""Adobe camera profiles (.dcp): reading, white point, tables, and the DNG development."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stealzalut.utils import camera_profile as cp
from stealzalut.utils.image_io import (
    RAW_BLACK_POINT,
    RENDERING_ADOBE_PROFILE,
    RENDERING_NEUTRAL_RAW,
    base_curve,
    develop_with_profile,
    load_image_rgb,
)

#: Linear ProPhoto → XYZ D50: a camera whose channels *are* ProPhoto RGB.
PROPHOTO_TO_XYZ = np.linalg.inv(cp.XYZ_D50_TO_PROPHOTO)
MODEL = "Acme One back camera"


def _tiff(tags: dict[int, object], order: str = "<", magic: int = 42) -> bytes:
    """A one-directory TIFF-structured file holding *tags* (str, float or int arrays)."""
    entries, blob = [], b""
    data_at = 8 + 2 + 12 * len(tags) + 4
    for tag in sorted(tags):
        value = tags[tag]
        if isinstance(value, str):
            kind, raw, count = 2, value.encode("latin-1") + b"\0", len(value) + 1
        else:
            array = np.asarray(value)
            if array.dtype.kind == "f":
                kind, raw, count = 12, array.astype(order + "f8").tobytes(), array.size
            else:
                kind, raw, count = 4, array.astype(order + "u4").tobytes(), array.size
        if len(raw) <= 4:
            field = raw.ljust(4, b"\0")
        else:
            field = struct.pack(order + "I", data_at + len(blob))
            blob += raw
        entries.append(struct.pack(order + "HHI", tag, kind, count) + field)
    header = (b"II" if order == "<" else b"MM") + struct.pack(order + "HI", magic, 8)
    directory = struct.pack(order + "H", len(entries)) + b"".join(entries) + b"\0\0\0\0"
    return header + directory + blob


def _profile_tags(**extra: object) -> dict[int, object]:
    """A two-light profile of a camera whose channels are ProPhoto RGB."""
    tags: dict[int, object] = {
        cp.TAG_UNIQUE_CAMERA_MODEL: MODEL,
        cp.TAG_PROFILE_NAME: "Adobe Standard",
        cp.TAG_CALIBRATION_ILLUMINANT_1: np.array([17]),
        cp.TAG_CALIBRATION_ILLUMINANT_2: np.array([21]),
        cp.TAG_COLOR_MATRIX_1: cp.XYZ_D50_TO_PROPHOTO.ravel(),
        cp.TAG_COLOR_MATRIX_2: cp.XYZ_D50_TO_PROPHOTO.ravel(),
        cp.TAG_FORWARD_MATRIX_1: PROPHOTO_TO_XYZ.ravel(),
        cp.TAG_FORWARD_MATRIX_2: PROPHOTO_TO_XYZ.ravel(),
    }
    tags.update({int(k.removeprefix("t")): v for k, v in extra.items()})
    return tags


def _dng_tags(neutral: tuple[float, float, float] = (1.0, 1.0, 1.0), ev: float = 1.0) -> dict:
    return {
        cp.TAG_UNIQUE_CAMERA_MODEL: MODEL,
        cp.TAG_ANALOG_BALANCE: np.array([2.0, 1.0, 1.5]),
        cp.TAG_AS_SHOT_NEUTRAL: np.array(neutral),
        cp.TAG_BASELINE_EXPOSURE: np.array([ev]),
    }


def _identity_table(values: int = 1, hues: int = 6, sats: int = 2) -> np.ndarray:
    table = np.zeros((values, hues, sats, 3), dtype=np.float32)
    table[..., 1:] = 1.0
    return table


class TestReading:
    @pytest.mark.parametrize("order", ["<", ">"])
    def test_tags_are_read_in_both_byte_orders(self, tmp_path: Path, order: str) -> None:
        path = tmp_path / "file.dng"
        path.write_bytes(_tiff(_dng_tags((0.5, 1.0, 0.7), ev=2.25), order))
        dng = cp.read_dng_colour(path)
        assert dng is not None
        assert dng.camera_model == MODEL
        assert dng.baseline_exposure == 2.25
        np.testing.assert_allclose(dng.as_shot_neutral, [0.5, 1.0, 0.7])
        np.testing.assert_allclose(dng.analog_balance, [2.0, 1.0, 1.5])
        assert not dng.gain_table_map

    def test_a_dng_without_neutral_or_another_file_gives_nothing(self, tmp_path: Path) -> None:
        tags = _dng_tags()
        del tags[cp.TAG_AS_SHOT_NEUTRAL]
        (tmp_path / "a.dng").write_bytes(_tiff(tags))
        (tmp_path / "b.dng").write_bytes(b"not a tiff at all")
        (tmp_path / "c.nef").write_bytes(_tiff(_dng_tags()))
        assert cp.read_dng_colour(tmp_path / "a.dng") is None
        assert cp.read_dng_colour(tmp_path / "b.dng") is None
        assert cp.read_dng_colour(tmp_path / "c.nef") is None

    def test_a_profile_is_read_with_its_tables(self, tmp_path: Path) -> None:
        look = _identity_table(values=3)
        path = tmp_path / "p.dcp"
        path.write_bytes(
            _tiff(
                _profile_tags(
                    t50937=np.array([6, 2, 1]),
                    t50938=_identity_table().ravel().astype(np.float64),
                    t50939=_identity_table().ravel().astype(np.float64),
                    t50981=np.array([6, 2, 3]),
                    t50982=look.ravel().astype(np.float64),
                    t51109=np.array([-0.25]),
                ),
                magic=0x4352,  # the magic number of a .dcp file
            )
        )
        profile = cp.read_camera_profile(path)
        assert profile is not None
        assert profile.kelvin == (2850.0, 6500.0)
        assert len(profile.hue_sat_maps) == 2
        assert profile.look_table is not None and profile.look_table.data.shape == (3, 6, 2, 3)
        assert profile.tone_curve is None
        assert profile.baseline_exposure_offset == -0.25

    def test_a_profile_without_forward_matrix_is_not_usable(self, tmp_path: Path) -> None:
        tags = _profile_tags()
        del tags[cp.TAG_FORWARD_MATRIX_1], tags[cp.TAG_FORWARD_MATRIX_2]
        path = tmp_path / "p.dcp"
        path.write_bytes(_tiff(tags, magic=0x4352))
        assert cp.read_camera_profile(path) is None

    def test_the_profile_of_the_camera_is_found_by_its_model(self, tmp_path: Path) -> None:
        other = _profile_tags()
        other[cp.TAG_UNIQUE_CAMERA_MODEL] = "Acme Two back camera"
        (tmp_path / "Acme Two back camera Adobe Standard.dcp").write_bytes(
            _tiff(other, magic=0x4352)
        )
        # Adobe puts the maker in front of the model in the file name
        (tmp_path / f"Acme {MODEL} Adobe Standard.dcp").write_bytes(
            _tiff(_profile_tags(), magic=0x4352)
        )
        found = cp.find_adobe_standard(MODEL, [tmp_path, tmp_path / "missing"])
        assert found is not None and found.camera_model == MODEL
        assert cp.find_adobe_standard("Unknown camera", [tmp_path]) is None
        assert cp.find_adobe_standard("", [tmp_path]) is None


class TestWhitePoint:
    def test_temperature_of_known_whites(self) -> None:
        assert abs(cp.correlated_temperature(0.3127, 0.3290) - 6504) < 40  # D65
        assert abs(cp.correlated_temperature(0.3457, 0.3585) - 5003) < 40  # D50
        assert abs(cp.correlated_temperature(0.4476, 0.4074) - 2856) < 40  # A

    def _plan(self, tmp_path: Path, neutral: tuple[float, float, float]) -> cp.Development:
        (tmp_path / "p.dcp").write_bytes(_tiff(_profile_tags(), magic=0x4352))
        (tmp_path / "f.dng").write_bytes(_tiff(_dng_tags(neutral, ev=1.5)))
        profile = cp.read_camera_profile(tmp_path / "p.dcp")
        dng = cp.read_dng_colour(tmp_path / "f.dng")
        assert profile is not None and dng is not None
        return cp.plan_development(profile, dng)

    def test_the_neutral_of_the_shot_lands_on_white(self, tmp_path: Path) -> None:
        for neutral in ((1.0, 1.0, 1.0), (0.6, 1.0, 0.8)):
            plan = self._plan(tmp_path, neutral)
            white = plan.camera_to_prophoto @ np.array(neutral)
            np.testing.assert_allclose(white / white.max(), [1.0, 1.0, 1.0], atol=2e-4)
            assert plan.exposure_ev == 1.5

    def test_a_camera_that_is_prophoto_gets_the_identity(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path, (2.0, 1.0, 1.5))  # the neutral the analog balance gives
        # AnalogBalance (2, 1, 1.5) is already in the values: the neutral is grey once undone
        np.testing.assert_allclose(
            plan.camera_to_prophoto @ np.array([2.0, 1.0, 1.5]), [1.0, 1.0, 1.0], atol=2e-4
        )
        assert abs(plan.kelvin - 5000) < 60


class TestHsvTable:
    def _colours(self) -> np.ndarray:
        rng = np.random.default_rng(3)
        return rng.random((500, 3)).astype(np.float32)

    def test_an_identity_table_changes_nothing(self) -> None:
        rgb = self._colours()
        for values in (1, 4):
            out = cp.apply_hsv_table(rgb, cp.HsvTable(_identity_table(values)))
            np.testing.assert_allclose(out, rgb, atol=2e-6)

    def test_a_hue_shift_of_a_third_of_a_turn_rotates_the_channels(self) -> None:
        table = _identity_table()
        table[..., 0] = 120.0
        rgb = self._colours()
        out = cp.apply_hsv_table(rgb, cp.HsvTable(table))
        np.testing.assert_allclose(out, rgb[:, [2, 0, 1]], atol=2e-6)  # red becomes green

    def test_saturation_and_value_factors(self) -> None:
        rgb = self._colours()
        grey = _identity_table()
        grey[..., 1] = 0.0
        out = cp.apply_hsv_table(rgb, cp.HsvTable(grey))
        np.testing.assert_allclose(out, np.repeat(rgb.max(1, keepdims=True), 3, 1), atol=2e-6)
        darker = _identity_table()
        darker[..., 2] = 0.5
        np.testing.assert_allclose(
            cp.apply_hsv_table(rgb, cp.HsvTable(darker)), rgb * 0.5, atol=2e-6
        )
        brighter = _identity_table()
        brighter[..., 2] = 4.0
        assert float(cp.apply_hsv_table(rgb, cp.HsvTable(brighter)).max()) <= 1.0

    def test_the_table_is_read_along_its_value_axis(self) -> None:
        table = _identity_table(values=2)
        table[1, ..., 2] = 0.5  # bright colours halved, dark ones kept
        grey = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32)
        out = cp.apply_hsv_table(grey, cp.HsvTable(table))
        np.testing.assert_allclose(out[:, 0], [0.0, 0.5 * 0.75, 0.5], atol=1e-6)


class TestDevelopment:
    def _plan(self, ev: float = 1.0, **tables: cp.HsvTable | None) -> cp.Development:
        return cp.Development(
            camera_to_prophoto=np.eye(3),
            hue_sat_map=tables.get("hue_sat_map"),
            look_table=tables.get("look_table"),
            tone_curve=None,
            exposure_ev=ev,
            kelvin=5000.0,
            profile_name="Adobe Standard",
        )

    def test_exposure_black_point_and_curve_on_a_grey_ramp(self) -> None:
        ramp = np.repeat(np.linspace(0.0, 1.0, 400, dtype=np.float32)[:, None], 3, 1)
        out = develop_with_profile(ramp, self._plan(ev=1.0))
        exposed = np.minimum(ramp[:, 0].astype(np.float64) * 2.0, 1.0)
        lowered = np.clip((exposed - RAW_BLACK_POINT) / (1.0 - RAW_BLACK_POINT), 0.0, 1.0)
        np.testing.assert_allclose(out[:, 0], base_curve(lowered), atol=2e-4)
        np.testing.assert_allclose(out[:, 0], out[:, 2], atol=1e-6)
        assert out[0, 0] == 0.0 and out[-1, 0] == pytest.approx(1.0, abs=1e-4)

    def test_the_black_point_is_taken_off_each_channel(self) -> None:
        colour = np.array([[0.2, 0.01, 0.002]], dtype=np.float32)
        out = develop_with_profile(colour, self._plan(ev=0.0))
        assert out[0, 2] == 0.0  # under the black point: gone, the others are not scaled by it
        assert out[0, 0] > out[0, 1] > 0.0

    def test_the_tables_take_part(self) -> None:
        colour = np.array([[0.3, 0.1, 0.05]], dtype=np.float32)
        plain = develop_with_profile(colour, self._plan())
        grey = _identity_table()
        grey[..., 1] = 0.0
        before = develop_with_profile(colour, self._plan(hue_sat_map=cp.HsvTable(grey)))
        after = develop_with_profile(colour, self._plan(look_table=cp.HsvTable(grey)))
        assert np.ptp(plain) > 0.1
        assert np.ptp(before) < 1e-6 and np.ptp(after) < 1e-6

    def _raw(self, value: int = 6000) -> MagicMock:
        raw = MagicMock()
        raw.postprocess.return_value = np.full((40, 60, 3), value, dtype=np.uint16)
        raw.__enter__.return_value = raw
        raw.__exit__.return_value = None
        return raw

    def test_a_dng_with_an_installed_profile_is_developed_with_it(self, tmp_path: Path) -> None:
        import rawpy

        folder = tmp_path / "profiles"
        folder.mkdir()
        (folder / f"{MODEL} Adobe Standard.dcp").write_bytes(_tiff(_profile_tags(), magic=0x4352))
        path = tmp_path / "photo.dng"
        path.write_bytes(_tiff(_dng_tags((2.0, 1.0, 1.5), ev=1.0)))
        raw = self._raw()
        with (
            patch("rawpy.imread", return_value=raw),
            patch("stealzalut.utils.camera_profile.adobe_profile_folders", return_value=[folder]),
        ):
            image = load_image_rgb(path)
        assert image.rendering == RENDERING_ADOBE_PROFILE
        assert image.native is not None and image.native_space.key == "display-p3"
        settings = raw.postprocess.call_args.kwargs
        # The camera's own channels, no white balance: the profile does the colour
        assert settings["output_color"] == rawpy.ColorSpace.raw  # type: ignore[attr-defined]
        assert settings["user_wb"] == [1.0, 1.0, 1.0, 1.0] and not settings["use_camera_wb"]
        assert settings["adjust_maximum_thr"] == 0.0
        # A grey would read (2, 1, 1.5): equal camera values lack red most, then blue
        r, g, b = (int(c) for c in image.native[0, 0])
        assert r < b < g

    def test_without_an_installed_profile_the_file_matrices_are_used(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.dng"
        path.write_bytes(_tiff(_dng_tags()))
        raw = self._raw()
        with (
            patch("rawpy.imread", return_value=raw),
            patch("stealzalut.utils.camera_profile.adobe_profile_folders", return_value=[tmp_path]),
        ):
            image = load_image_rgb(path)
        assert image.rendering == RENDERING_NEUTRAL_RAW
        assert raw.postprocess.call_args.kwargs["use_camera_wb"] is True
