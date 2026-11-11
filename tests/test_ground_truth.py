"""Tests for the ground-truth bench (development tool)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from stealzalut.core import look as look_module
from stealzalut.core.colorspace import srgb_to_oklab
from stealzalut.devtools.ground_truth import (
    GRADES,
    Case,
    Grade,
    by_grade,
    evaluate,
    load_folder,
    look_constants,
    main,
    measure,
    pairs,
    render,
    report,
    summarize,
)

U8 = np.ndarray


def _scene(seed: int, base: tuple[int, int, int], accent: tuple[int, int, int]) -> U8:
    """A small picture with a lightness ramp, a coloured area and some noise."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.15, 1.0, 96)[:, None, None]
    img = np.tile(np.array(base, dtype=np.float64)[None, None, :] * ramp, (1, 128, 1))
    img[20:60, 30:90] = accent
    img[70:90, 10:50] = (235, 232, 225)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def images() -> list[tuple[str, U8]]:
    return [
        ("a", _scene(1, (150, 140, 130), (200, 60, 50))),
        ("b", _scene(2, (90, 110, 150), (60, 150, 80))),
        ("c", _scene(3, (170, 170, 175), (50, 80, 200))),
    ]


# ── Grades ───────────────────────────────────────────────────────────────────


def test_neutral_grade_changes_nothing(images: list[tuple[str, U8]]) -> None:
    graded = Grade("neutre").apply(images[0][1])
    assert np.abs(graded.astype(int) - images[0][1].astype(int)).max() <= 1


def test_grade_names_are_unique() -> None:
    assert len({grade.name for grade in GRADES}) == len(GRADES)


@pytest.mark.parametrize("grade", GRADES, ids=lambda g: g.name)
def test_grade_tone_is_monotone(grade: Grade) -> None:
    greys = np.column_stack([np.linspace(0.0, 1.0, 200), np.zeros(200), np.zeros(200)])
    assert np.all(np.diff(grade.apply_lab(greys)[:, 0]) >= -1e-12)


def test_grade_is_a_function_of_colour_only(images: list[tuple[str, U8]]) -> None:
    """Same colour, same result, wherever it sits: what a LUT can represent."""
    grade = GRADES[5]
    image = images[0][1]
    shuffled = np.random.default_rng(0).permutation(image.reshape(-1, 3)).reshape(image.shape)
    assert sorted(map(tuple, grade.apply(image).reshape(-1, 3))) == sorted(
        map(tuple, grade.apply(shuffled).reshape(-1, 3))
    )


def test_split_tone_grade_tints_shadows_and_highlights() -> None:
    grade = Grade("essai", shadows=(-0.04, 0.0), highlights=(0.04, 0.0))
    out = grade.apply_lab(np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]))
    assert out[0, 1] < -0.02 and out[1, 1] > 0.02


# ── Pairs ────────────────────────────────────────────────────────────────────


def test_pairs_cover_both_regimes() -> None:
    result = pairs(4, 2)
    assert result[:4] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert len(result) == 4 + 4 * 2
    assert all(source != reference for source, reference in result[4:])
    assert result == pairs(4, 2)  # deterministic


def test_pairs_with_few_images() -> None:
    assert pairs(1, 3) == [(0, 0)]
    assert len(pairs(2, 5)) == 2 + 2


# ── Measure ──────────────────────────────────────────────────────────────────


def test_measure_skips_a_grade_that_does_nothing(images: list[tuple[str, U8]]) -> None:
    assert measure("oklab", Grade("neutre"), images[0], images[0]) is None


def test_measure_reports_the_parts_of_the_error(images: list[tuple[str, U8]]) -> None:
    case = measure("look", GRADES[5], images[0], images[1])
    assert isinstance(case, Case)
    assert not case.same_content and case.source == "a" and case.reference == "b"
    assert case.residual == pytest.approx(case.error / case.nothing)
    assert case.error <= case.error_lightness + case.error_chroma + 1e-9
    assert case.nothing > 0.01


def test_same_content_is_recovered_far_better_than_doing_nothing(
    images: list[tuple[str, U8]],
) -> None:
    """With the same picture on both sides, the look method finds the grade."""
    residuals = [
        case.residual
        for grade in GRADES
        if (case := measure("look", grade, images[0], images[0])) is not None
    ]
    assert len(residuals) == len(GRADES)
    assert np.mean(residuals) < 0.35
    assert max(residuals) < 0.7


def test_render_is_what_the_lut_gives(images: list[tuple[str, U8]]) -> None:
    out = render("oklab", images[0][1], images[1][1])
    assert out.shape == images[0][1].shape and out.dtype == np.uint8
    before = srgb_to_oklab(images[0][1].reshape(-1, 3) / 255.0)
    after = srgb_to_oklab(out.reshape(-1, 3) / 255.0)
    assert np.abs(after - before).mean() > 0.005  # something was transferred


# ── Summary ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cases(images: list[tuple[str, U8]]) -> list[Case]:
    return evaluate(images, methods=("look", "oklab"), grades=GRADES[:3], others=1)


def test_evaluate_runs_every_combination(cases: list[Case]) -> None:
    assert len(cases) == (3 + 3) * 3 * 2  # (same + other pairs) × grades × methods
    assert {case.method for case in cases} == {"look", "oklab"}
    assert sum(case.same_content for case in cases) == 3 * 3 * 2


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


def test_a_constant_changes_the_measure(images: list[tuple[str, U8]]) -> None:
    plain = measure("look", GRADES[5], images[0], images[1])
    with look_constants(BANDS=4):
        changed = measure("look", GRADES[5], images[0], images[1])
    assert plain is not None and changed is not None
    assert plain.error != changed.error


# ── Command line ─────────────────────────────────────────────────────────────


def _folder(tmp_path: Path, images: list[tuple[str, U8]]) -> Path:
    for name, image in images:
        cv2.imwrite(str(tmp_path / f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    (tmp_path / "notes.txt").write_text("not a picture", encoding="utf-8")
    return tmp_path


def test_load_folder_reads_pictures_only(tmp_path: Path, images: list[tuple[str, U8]]) -> None:
    loaded = load_folder(_folder(tmp_path, images))
    assert [name for name, _ in loaded] == ["a.png", "b.png", "c.png"]


def test_main_prints_a_report(
    tmp_path: Path, images: list[tuple[str, U8]], capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _folder(tmp_path, images[:2])
    out = tmp_path / "cases.json"
    code = main([str(folder), "--methods", "oklab", "--others", "1", "--json", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "2 images" in printed and "| oklab | même contenu |" in printed
    assert out.exists() and '"residual"' in out.read_text(encoding="utf-8")


def test_main_needs_two_pictures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path)]) == 2
    assert "au moins 2 images" in capsys.readouterr().err
