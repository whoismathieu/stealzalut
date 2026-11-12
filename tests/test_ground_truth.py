"""Tests for the ground-truth bench (development tool)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stealzalut.core import look as look_module
from stealzalut.core.colorspace import _in_gamut, oklab_to_linear, srgb_to_oklab
from stealzalut.core.rgb_spaces import DISPLAY_P3
from stealzalut.devtools.ground_truth import (
    BENCH_SPACE,
    GRADES,
    LOTS,
    BenchImage,
    Case,
    Grade,
    as_picture,
    bench_image,
    by_grade,
    evaluate,
    load_folder,
    look_constants,
    main,
    measure,
    pairs,
    render,
    report,
    split,
    summarize,
)
from stealzalut.utils.image_io import WideImage

U8 = np.ndarray


def _scene(seed: int, base: tuple[int, int, int], accent: tuple[int, int, int]) -> U8:
    """A small picture with a lightness ramp, a coloured area and some noise."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.15, 1.0, 96)[:, None, None]
    img = np.tile(np.array(base, dtype=np.float64)[None, None, :] * ramp, (1, 128, 1))
    img[20:60, 30:90] = accent
    img[70:90, 10:50] = (235, 232, 225)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


SCENES = (
    ("a", "nuit", _scene(1, (150, 140, 130), (200, 60, 50))),
    ("b", "nuit", _scene(2, (90, 110, 150), (60, 150, 80))),
    ("c", "jour", _scene(3, (170, 170, 175), (50, 80, 200))),
    ("d", "jour", _scene(4, (120, 150, 110), (210, 180, 40))),
)


@pytest.fixture(scope="module")
def images() -> list[BenchImage]:
    return [bench_image(name, scene, category) for name, category, scene in SCENES]


# ── Pictures ─────────────────────────────────────────────────────────────────


def test_bench_image_holds_the_analysis_colours(images: list[BenchImage]) -> None:
    first = images[0]
    assert first.lab.shape == (96, 128, 3) and first.flat.shape == (96 * 128, 3)
    assert np.allclose(first.flat, srgb_to_oklab(SCENES[0][2].reshape(-1, 3) / 255.0))


def test_bench_image_keeps_wide_colours() -> None:
    srgb = SCENES[0][2]
    wide = srgb_to_oklab(srgb.reshape(-1, 3) / 255.0).reshape(96, 128, 3) * [1.0, 1.5, 1.5]
    image = bench_image("w", WideImage(srgb, wide))
    assert np.allclose(image.lab, wide)


def test_as_picture_carries_unclipped_colours() -> None:
    lab = DISPLAY_P3.to_oklab(np.array([[1.0, 0.0, 0.0]] * 12)).reshape(3, 4, 3)
    picture = as_picture(lab)
    assert isinstance(picture, WideImage) and picture.shape == (3, 4, 3)
    assert picture.analysis_lab is not None and np.array_equal(picture.analysis_lab, lab)
    shown = srgb_to_oklab(np.asarray(picture).reshape(-1, 3) / 255.0)
    assert np.hypot(shown[0, 1], shown[0, 2]) < np.hypot(lab[0, 0, 1], lab[0, 0, 2]) - 0.01


# ── Grades ───────────────────────────────────────────────────────────────────


def test_neutral_grade_changes_nothing(images: list[BenchImage]) -> None:
    assert np.allclose(Grade("neutre").grade(images[0].lab), images[0].lab, atol=1e-9)


def test_grade_names_are_unique() -> None:
    assert len({grade.name for grade in GRADES}) == len(GRADES)


@pytest.mark.parametrize("grade", GRADES, ids=lambda g: g.name)
def test_grade_tone_is_monotone(grade: Grade) -> None:
    greys = np.column_stack([np.linspace(0.0, 1.0, 200), np.zeros(200), np.zeros(200)])
    assert np.all(np.diff(grade.apply_lab(greys)[:, 0]) >= -1e-12)


def test_split_tone_grade_tints_shadows_and_highlights() -> None:
    grade = Grade("essai", shadows=(-0.04, 0.0), highlights=(0.04, 0.0))
    out = grade.apply_lab(np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]))
    assert out[0, 1] < -0.02 and out[1, 1] > 0.02


def test_graded_colours_stay_wide_but_inside_the_bench_gamut() -> None:
    """Saturating a vivid sRGB colour leaves sRGB; it must not be clipped back."""
    vivid = srgb_to_oklab(np.array([[1.0, 0.05, 0.05], [0.05, 0.9, 0.1]]))
    graded = Grade("fort", saturation=1.35).grade(vivid)
    assert not _in_gamut(oklab_to_linear(graded)).all()  # outside sRGB
    assert _in_gamut(BENCH_SPACE.oklab_to_linear(graded)).all()  # inside Display P3
    assert np.all(np.hypot(graded[:, 1], graded[:, 2]) > np.hypot(vivid[:, 1], vivid[:, 2]))


# ── Pairs and lots ───────────────────────────────────────────────────────────


def test_pairs_cover_both_regimes() -> None:
    result = pairs(4, 2)
    assert result[:4] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert len(result) == 4 + 4 * 2
    assert all(source != reference for source, reference in result[4:])
    assert result == pairs(4, 2)  # deterministic


def test_pairs_with_few_images() -> None:
    assert pairs(1, 3) == [(0, 0)]
    assert len(pairs(2, 5)) == 2 + 2


def test_split_alternates_within_each_category(images: list[BenchImage]) -> None:
    tune, check = split(images)
    assert [i.name for i in tune] == ["c", "a"]  # categories sorted: "jour", then "nuit"
    assert [i.name for i in check] == ["d", "b"]
    assert {i.category for i in tune} == {i.category for i in check} == {"jour", "nuit"}
    assert not {i.name for i in tune} & {i.name for i in check}


def test_split_is_deterministic_and_complete(images: list[BenchImage]) -> None:
    shuffled = [images[2], images[0], images[3], images[1]]
    assert [[i.name for i in lot] for lot in split(shuffled)] == [
        [i.name for i in lot] for lot in split(images)
    ]
    assert sum(len(lot) for lot in split(images[:3])) == 3


# ── Measure ──────────────────────────────────────────────────────────────────


def test_measure_skips_a_grade_that_does_nothing(images: list[BenchImage]) -> None:
    assert measure("oklab", Grade("neutre"), images[0], images[0]) is None


def test_measure_reports_the_parts_of_the_error(images: list[BenchImage]) -> None:
    case = measure("look", GRADES[5], images[0], images[1], lot="reglage")
    assert isinstance(case, Case) and case.lot == "reglage"
    assert not case.same_content and case.source == "a" and case.reference == "b"
    assert case.residual == pytest.approx(case.error / case.nothing)
    assert case.error <= case.error_lightness + case.error_chroma + 1e-9
    assert case.nothing > 0.01


def test_same_content_is_recovered_far_better_than_doing_nothing(
    images: list[BenchImage],
) -> None:
    """With the same picture on both sides, the look method finds the grade."""
    residuals = [
        case.residual
        for grade in GRADES
        if (case := measure("look", grade, images[0], images[0])) is not None
    ]
    assert len(residuals) == len(GRADES)
    assert np.mean(residuals) < 0.35
    assert max(residuals) < 0.8


def test_render_gives_float_colours_of_the_source(images: list[BenchImage]) -> None:
    out = render("oklab", images[0], images[1].lab)
    assert out.shape == images[0].flat.shape and out.dtype == np.float64
    assert np.abs(out - images[0].flat).mean() > 0.005  # something was transferred
    assert _in_gamut(BENCH_SPACE.oklab_to_linear(out)).all()


# ── Summary ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cases(images: list[BenchImage]) -> list[Case]:
    return evaluate(images[:3], methods=("look", "oklab"), grades=GRADES[:3], others=1, lot="x")


def test_evaluate_runs_every_combination(cases: list[Case]) -> None:
    assert len(cases) == (3 + 3) * 3 * 2  # (same + other pairs) × grades × methods
    assert {case.method for case in cases} == {"look", "oklab"}
    assert sum(case.same_content for case in cases) == 3 * 3 * 2
    assert {case.lot for case in cases} == {"x"}


def test_summary_separates_the_regimes(cases: list[Case]) -> None:
    summary = summarize(cases)
    assert set(summary) == {"look", "oklab"}
    for regimes in summary.values():
        assert set(regimes) == {"same", "other"}
        assert regimes["same"]["cases"] == 9 and regimes["other"]["cases"] == 9
        assert regimes["same"]["worst"] >= regimes["same"]["residual"] >= 0.0
    # Copying statistics also copies the subject: other content is the harder regime
    assert summary["look"]["other"]["residual"] > summary["look"]["same"]["residual"]


def test_by_grade_and_report(cases: list[Case]) -> None:
    table = by_grade(cases, same_content=True)
    assert set(table) == {grade.name for grade in GRADES[:3]}
    assert set(table["contraste"]) == {"look", "oklab"}
    text = report(cases)
    assert "| look | même contenu |" in text and "| oklab | autre contenu |" in text
    assert "Résidu moyen par étalonnage, autre contenu :" in text


# ── Constants ────────────────────────────────────────────────────────────────


def test_look_constants_are_restored() -> None:
    original = look_module.SIGMA_FINE
    with look_constants(SIGMA_FINE=3.0):
        assert look_module.SIGMA_FINE == 3.0
    assert original == look_module.SIGMA_FINE


def test_look_constants_are_restored_after_an_error() -> None:
    original = look_module.BANDS
    with pytest.raises(RuntimeError), look_constants(BANDS=6):
        raise RuntimeError("boom")
    assert original == look_module.BANDS


def test_look_constants_reject_unknown_names() -> None:
    with pytest.raises(AttributeError, match="no constant"), look_constants(NOPE=1):
        pass
    with pytest.raises(AttributeError, match="no constant"), look_constants(fit_look=1):
        pass


def test_a_constant_changes_the_measure(images: list[BenchImage]) -> None:
    plain = measure("look", GRADES[5], images[0], images[1])
    with look_constants(BANDS=4):
        changed = measure("look", GRADES[5], images[0], images[1])
    assert plain is not None and changed is not None
    assert plain.error != changed.error


# ── Command line ─────────────────────────────────────────────────────────────


def _folder(tmp_path: Path, manifest: bool = True) -> Path:
    entries = []
    for name, category, scene in SCENES:
        cv2.imwrite(str(tmp_path / f"{name}.png"), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))
        entries.append({"file": f"{name}.png", "category": category})
    (tmp_path / "notes.txt").write_text("not a picture", encoding="utf-8")
    if manifest:
        (tmp_path / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return tmp_path


def test_load_folder_reads_pictures_and_categories(tmp_path: Path) -> None:
    loaded = load_folder(_folder(tmp_path))
    assert [image.name for image in loaded] == ["a.png", "b.png", "c.png", "d.png"]
    assert [image.category for image in loaded] == ["nuit", "nuit", "jour", "jour"]


def test_load_folder_without_manifest(tmp_path: Path) -> None:
    loaded = load_folder(_folder(tmp_path, manifest=False))
    assert len(loaded) == 4 and {image.category for image in loaded} == {""}


def test_main_reports_both_lots(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cases.json"
    code = main([str(_folder(tmp_path)), "--methods", "oklab", "--others", "1", "--json", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "4 images" in printed
    for lot in LOTS:
        assert f"## Lot « {lot} » : 2 images" in printed
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert {case["lot"] for case in saved} == set(LOTS)


def test_main_can_measure_one_lot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main([str(_folder(tmp_path)), "--methods", "oklab", "--others", "1", "--lot", "reglage"])
    printed = capsys.readouterr().out
    assert code == 0 and "« reglage »" in printed and "« controle »" not in printed


def test_main_needs_four_pictures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path)]) == 2
    assert "au moins 4 images" in capsys.readouterr().err
