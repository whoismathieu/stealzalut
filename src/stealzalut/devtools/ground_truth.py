"""Ground-truth bench: how well does a method recover a grade it is shown?

Judging a colour transfer by eye on a handful of pairs tunes the engine to
those pairs.  This bench gives a number instead, on any set of pictures:

1. take a picture ``P`` and apply a **known grade** ``G`` to it;
2. give the engine another picture ``Q`` as source and ``G(P)`` as reference;
3. compare its output with ``G(Q)``, which is the exact answer.

Two regimes are reported separately, because they answer different
questions:

* **same content** (``Q = P``): the grade can be recovered exactly in
  principle.  The residual measures what the method loses on its own.
* **other content** (``Q ≠ P``): a method that copies the reference's
  statistics also copies its *subject* (a dark reference darkens a bright
  source).  The residual adds that over-transfer to the first one.

The score is the **residual**: error of the method divided by the error of
doing nothing.  0 = the grade is recovered, 1 = no better than leaving the
source alone, above 1 = worse than doing nothing.

What this bench does **not** measure: taste.  A grade here is one global
function of colour; it says nothing about what makes a look beautiful.

Run::

    python -m stealzalut.devtools.ground_truth <folder of pictures>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stealzalut.core import look as look_module
from stealzalut.core.color_transfer import SUPPORTED_METHODS, fit_transfer
from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.finishing import eval_curve
from stealzalut.core.grading import build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.utils.image_io import (
    SUPPORTED_EXTENSIONS,
    create_thumbnail,
    load_image_rgb,
)

F64 = NDArray[np.float64]
U8 = NDArray[np.uint8]

#: Longest edge of the pictures the bench works on.
BENCH_SIZE = 384
#: LUT resolution used to render (the preview's).
BENCH_LUT = 33
#: Below this mean error of "doing nothing", a case has no grade to recover.
MIN_GRADE_SIZE = 1e-3


# ── Known grades ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Grade:
    """A grade whose effect on any colour is known exactly.

    Attributes:
        name: Identifier.
        tone: Points of the monotone tone curve (Oklab L in, L out).
        shadows: Tint ``(a, b)`` added to the darkest tones.
        midtones: Tint added to the mid-tones.
        highlights: Tint added to the brightest tones.
        saturation: Factor applied to chroma before the tints.
    """

    name: str
    tone: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
    shadows: tuple[float, float] = (0.0, 0.0)
    midtones: tuple[float, float] = (0.0, 0.0)
    highlights: tuple[float, float] = (0.0, 0.0)
    saturation: float = 1.0

    def apply_lab(self, lab: F64) -> F64:
        """The grade on Oklab colours ``(N, 3)`` (before any gamut limit)."""
        out = np.empty_like(lab)
        lightness = np.clip(eval_curve(self.tone, lab[:, 0]), 0.0, 1.0)
        out[:, 0] = lightness
        weights = (
            ((1.0 - lightness) ** 2, self.shadows),
            (2.0 * lightness * (1.0 - lightness), self.midtones),
            (lightness**2, self.highlights),
        )
        out[:, 1:] = lab[:, 1:] * self.saturation
        for weight, tint in weights:
            out[:, 1:] += weight[:, None] * np.array(tint)[None, :]
        return out

    def apply(self, image_rgb: U8) -> U8:
        """The grade on an RGB uint8 picture, brought back into sRGB."""
        lab = srgb_to_oklab(image_rgb.reshape(-1, 3).astype(np.float64) / 255.0)
        rgb = oklab_to_srgb(gamut_compress(self.apply_lab(lab)))
        graded: U8 = (rgb * 255.0 + 0.5).astype(np.uint8).reshape(image_rgb.shape)
        return graded


#: Grades of the bench: tone only, colour only, and both, mild to strong.
GRADES: tuple[Grade, ...] = (
    Grade("contraste", tone=((0.0, 0.0), (0.25, 0.17), (0.75, 0.83), (1.0, 1.0))),
    Grade("sombre", tone=((0.0, 0.0), (0.5, 0.33), (1.0, 0.92))),
    Grade("delave", tone=((0.0, 0.08), (0.5, 0.52), (1.0, 0.94)), saturation=0.8),
    Grade("sature", saturation=1.35),
    Grade("sarcelle-orange", shadows=(-0.03, -0.02), highlights=(0.02, 0.03), saturation=1.1),
    Grade(
        "nuit-verte",
        tone=((0.0, 0.0), (0.5, 0.40), (1.0, 0.95)),
        shadows=(-0.04, 0.02),
        midtones=(-0.02, 0.01),
        highlights=(0.02, -0.01),
        saturation=1.15,
    ),
    Grade("dominante-violette", shadows=(0.03, -0.06), midtones=(0.04, -0.07), saturation=0.9),
    Grade(
        "chaud-doux",
        tone=((0.0, 0.03), (0.5, 0.54), (1.0, 0.97)),
        midtones=(0.015, 0.03),
        highlights=(0.01, 0.02),
    ),
)


# ── Measure ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Case:
    """One measurement.

    Attributes:
        method: Transfer method.
        grade: Name of the grade.
        source: Name of the source picture ``Q``.
        reference: Name of the picture ``P`` the reference was made from.
        same_content: True when ``Q`` is ``P``.
        error: Mean Oklab distance between the output and ``G(Q)``.
        error_lightness: Mean absolute lightness error.
        error_chroma: Mean distance in the ``(a, b)`` plane.
        nothing: Mean Oklab distance between ``Q`` and ``G(Q)``.
        residual: ``error / nothing``.
    """

    method: str
    grade: str
    source: str
    reference: str
    same_content: bool
    error: float
    error_lightness: float
    error_chroma: float
    nothing: float
    residual: float


def _lab(image_rgb: U8) -> F64:
    return srgb_to_oklab(image_rgb.reshape(-1, 3).astype(np.float64) / 255.0)


def render(method: str, source: U8, reference: U8) -> U8:
    """What the application would show: the method's 33³ LUT applied to *source*."""
    model = fit_transfer(source, reference, method=method)
    return apply_lut(source, build_grid(model, BENCH_LUT).grid, BENCH_LUT)


def measure(
    method: str,
    grade: Grade,
    source: tuple[str, U8],
    reference: tuple[str, U8],
) -> Case | None:
    """Run one case; ``None`` when the grade barely changes the source."""
    truth = _lab(grade.apply(source[1]))
    nothing = float(np.sqrt(((_lab(source[1]) - truth) ** 2).sum(axis=1)).mean())
    if nothing < MIN_GRADE_SIZE:
        return None
    output = _lab(render(method, source[1], grade.apply(reference[1])))
    diff = output - truth
    error = float(np.sqrt((diff**2).sum(axis=1)).mean())
    return Case(
        method=method,
        grade=grade.name,
        source=source[0],
        reference=reference[0],
        same_content=source[0] == reference[0],
        error=error,
        error_lightness=float(np.abs(diff[:, 0]).mean()),
        error_chroma=float(np.hypot(diff[:, 1], diff[:, 2]).mean()),
        nothing=nothing,
        residual=error / nothing,
    )


def pairs(count: int, others: int) -> list[tuple[int, int]]:
    """Deterministic ``(source, reference)`` index pairs.

    Each picture is its own reference once (same content), then the
    reference of the *others* next pictures, cyclically (other content).
    """
    result = [(i, i) for i in range(count)]
    for i in range(count):
        for step in range(1, min(others, count - 1) + 1):
            result.append(((i + step) % count, i))
    return result


def evaluate(
    images: Sequence[tuple[str, U8]],
    methods: Sequence[str] = ("look", "zones", "oklab"),
    grades: Sequence[Grade] = GRADES,
    others: int = 2,
) -> list[Case]:
    """Measure every method on every grade and pair of *images*."""
    small = [(name, create_thumbnail(image, BENCH_SIZE)) for name, image in images]
    cases: list[Case] = []
    for source_index, reference_index in pairs(len(small), others):
        for grade in grades:
            for method in methods:
                case = measure(method, grade, small[source_index], small[reference_index])
                if case is not None:
                    cases.append(case)
    return cases


# ── Summary ──────────────────────────────────────────────────────────────────


def summarize(cases: Sequence[Case]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean scores per method and regime.

    Returns:
        ``{method: {"same" | "other": {"residual", "median", "worst",
        "error", "lightness", "chroma", "cases"}}}``.
    """
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method in sorted({case.method for case in cases}):
        summary[method] = {}
        for regime, same in (("same", True), ("other", False)):
            chosen = [c for c in cases if c.method == method and c.same_content is same]
            if not chosen:
                continue
            residuals = np.array([c.residual for c in chosen])
            summary[method][regime] = {
                "residual": float(residuals.mean()),
                "median": float(np.median(residuals)),
                "worst": float(residuals.max()),
                "error": float(np.mean([c.error for c in chosen])),
                "lightness": float(np.mean([c.error_lightness for c in chosen])),
                "chroma": float(np.mean([c.error_chroma for c in chosen])),
                "cases": float(len(chosen)),
            }
    return summary


def by_grade(cases: Sequence[Case], same_content: bool) -> dict[str, dict[str, float]]:
    """Mean residual per grade and method, for one regime."""
    table: dict[str, dict[str, float]] = {}
    for case in cases:
        if case.same_content is not same_content:
            continue
        table.setdefault(case.grade, {}).setdefault(case.method, 0.0)
    for grade, row in table.items():
        for method in row:
            values = [
                c.residual
                for c in cases
                if c.grade == grade and c.method == method and c.same_content is same_content
            ]
            row[method] = float(np.mean(values))
    return table


def report(cases: Sequence[Case]) -> str:
    """Markdown report of *cases*."""
    lines = [
        "Résidu = erreur de la méthode / erreur de ne rien faire "
        "(0 : étalonnage retrouvé ; 1 : inutile ; > 1 : pire que rien).",
        "",
        "| Méthode | Régime | Résidu moyen | Médian | Pire | Erreur L | Erreur couleur | Cas |",
        "|---|---|---|---|---|---|---|---|",
    ]
    labels = {"same": "même contenu", "other": "autre contenu"}
    for method, regimes in summarize(cases).items():
        for regime, s in regimes.items():
            lines.append(
                f"| {method} | {labels[regime]} | {s['residual']:.2f} | {s['median']:.2f} "
                f"| {s['worst']:.2f} | {s['lightness']:.3f} | {s['chroma']:.3f} "
                f"| {int(s['cases'])} |"
            )
    for same, title in ((True, "même contenu"), (False, "autre contenu")):
        table = by_grade(cases, same)
        if not table:
            continue
        methods = sorted({m for row in table.values() for m in row})
        lines += ["", f"Résidu moyen par étalonnage, {title} :", ""]
        lines.append("| Étalonnage | " + " | ".join(methods) + " |")
        lines.append("|---|" + "---|" * len(methods))
        for grade in (g.name for g in GRADES):
            if grade in table:
                cells = " | ".join(f"{table[grade].get(m, float('nan')):.2f}" for m in methods)
                lines.append(f"| {grade} | {cells} |")
    return "\n".join(lines)


# ── Constants of the look method, overridden for one run ─────────────────────


@contextmanager
def look_constants(**values: Any) -> Iterator[None]:
    """Temporarily replace module constants of :mod:`stealzalut.core.look`.

    Raises:
        AttributeError: If a name is not a constant of that module.
    """
    previous: dict[str, Any] = {}
    for name in values:
        if not name.isupper() or not hasattr(look_module, name):
            raise AttributeError(f"stealzalut.core.look has no constant '{name}'")
    try:
        for name, value in values.items():
            previous[name] = getattr(look_module, name)
            setattr(look_module, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(look_module, name, value)


# ── Command line ─────────────────────────────────────────────────────────────


def load_folder(folder: Path) -> list[tuple[str, U8]]:
    """Every readable picture of *folder* (not recursive), sorted by name."""
    images: list[tuple[str, U8]] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            image = np.asarray(load_image_rgb(path))  # the bench grades sRGB pictures
        except (OSError, ValueError):
            continue
        images.append((path.name, create_thumbnail(image, BENCH_SIZE)))
    return images


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="folder of pictures (not recursive)")
    parser.add_argument("--methods", default="look,zones,oklab", help="comma-separated")
    parser.add_argument("--others", type=int, default=2, help="other-content pairs per picture")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a constant of core/look.py for this run")
    parser.add_argument("--json", type=Path, help="also write every case to this file")
    args = parser.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)}")
    overrides: dict[str, Any] = {}
    for item in args.set:
        name, _, value = item.partition("=")
        overrides[name.strip()] = _parse_value(value.strip())

    images = load_folder(args.folder)
    if len(images) < 2:
        print(f"Il faut au moins 2 images lisibles dans {args.folder}.", file=sys.stderr)
        return 2
    with look_constants(**overrides):
        cases = evaluate(images, methods, others=args.others)
    print(f"{len(images)} images, {len(cases)} cas" + (f", {overrides}" if overrides else ""))
    print()
    print(report(cases))
    if args.json is not None:
        args.json.write_text(
            json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
