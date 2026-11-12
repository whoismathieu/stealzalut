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

Everything is measured in Oklab, in floating point, and nothing is clipped
to sRGB: graded pictures are only brought into the Display P3 gamut, the
method's LUT is baked for Display P3 and applied to float colours.

The pictures are split in two lots, **tuning** and **control**, alternating
within each category.  Constants are to be changed looking at the first
lot only; the second says whether the change generalises.

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
from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb
from stealzalut.core.finishing import eval_curve
from stealzalut.core.grading import build_grid
from stealzalut.core.lut_apply import apply_lut_colors
from stealzalut.core.palette import analysis_oklab
from stealzalut.core.rgb_spaces import DISPLAY_P3
from stealzalut.utils.image_io import SUPPORTED_EXTENSIONS, WideImage, load_image_rgb

F64 = NDArray[np.float64]
U8 = NDArray[np.uint8]

#: LUT resolution used to render (the preview's).
BENCH_LUT = 33
#: Space the bench works in: wide enough for graded colours, with real primaries.
BENCH_SPACE = DISPLAY_P3
#: Below this mean error of "doing nothing", a case has no grade to recover.
MIN_GRADE_SIZE = 1e-3
#: Names of the two lots.
LOTS: tuple[str, str] = ("reglage", "controle")


# ── Pictures ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, eq=False)
class BenchImage:
    """A picture as the engine analyses it.

    Attributes:
        name: File name.
        category: Group used to balance the lots ("" if unknown).
        lab: Oklab colours of the analysis copy, ``(h, w, 3)``, unclipped.
    """

    name: str
    category: str
    lab: F64

    @property
    def flat(self) -> F64:
        """The colours as ``(h × w, 3)``."""
        return self.lab.reshape(-1, 3)


def bench_image(name: str, image_rgb: U8, category: str = "") -> BenchImage:
    """A :class:`BenchImage` from a loaded picture (wide colours if it has them)."""
    lab, h, w = analysis_oklab(image_rgb)
    return BenchImage(name, category, lab.reshape(h, w, 3).copy())


def as_picture(lab: F64) -> WideImage:
    """Oklab colours ``(h, w, 3)`` as the picture the engine would be handed.

    The array is the sRGB rendition (what a screen shows); the unclipped
    colours ride along, as for a picture loaded from a wide-gamut file.
    """
    h, w = lab.shape[:2]
    srgb = oklab_to_srgb(gamut_compress(lab.reshape(-1, 3)))
    pixels: U8 = (srgb * 255.0 + 0.5).astype(np.uint8).reshape(h, w, 3)
    return WideImage(pixels, lab, BENCH_SPACE.label)


def _into_gamut(lab: F64) -> F64:
    return gamut_compress(lab, to_linear=BENCH_SPACE.oklab_to_linear)


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

    def grade(self, lab: F64) -> F64:
        """The grade on colours of any shape ``(..., 3)``, kept inside the bench's gamut."""
        return _into_gamut(self.apply_lab(lab.reshape(-1, 3))).reshape(lab.shape)


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
        lot: "reglage" or "controle".
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

    lot: str
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


def render(method: str, source: BenchImage, reference_lab: F64) -> F64:
    """The method's LUT applied to the source: Oklab colours ``(h × w, 3)``.

    The grid is baked for :data:`BENCH_SPACE` and applied to float colours,
    so nothing is rounded to 8 bits or clipped to sRGB on the way.
    """
    model = fit_transfer(as_picture(source.lab), as_picture(reference_lab), method=method)
    grid = build_grid(model, BENCH_LUT, None, None, BENCH_SPACE).grid
    encoded = BENCH_SPACE.from_oklab(_into_gamut(source.flat))
    return BENCH_SPACE.to_oklab(apply_lut_colors(encoded, grid, BENCH_LUT))


def measure(
    method: str,
    grade: Grade,
    source: BenchImage,
    reference: BenchImage,
    lot: str = "",
) -> Case | None:
    """Run one case; ``None`` when the grade barely changes the source."""
    truth = grade.grade(source.flat)
    nothing = float(np.sqrt(((source.flat - truth) ** 2).sum(axis=1)).mean())
    if nothing < MIN_GRADE_SIZE:
        return None
    diff = render(method, source, grade.grade(reference.lab)) - truth
    error = float(np.sqrt((diff**2).sum(axis=1)).mean())
    return Case(
        lot=lot,
        method=method,
        grade=grade.name,
        source=source.name,
        reference=reference.name,
        same_content=source is reference,
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


def split(images: Sequence[BenchImage]) -> tuple[list[BenchImage], list[BenchImage]]:
    """Tuning and control lots: pictures alternate within each category, by name."""
    lots: tuple[list[BenchImage], list[BenchImage]] = ([], [])
    for category in sorted({image.category for image in images}):
        members = sorted((i for i in images if i.category == category), key=lambda i: i.name)
        for index, image in enumerate(members):
            lots[index % 2].append(image)
    return lots


def evaluate(
    images: Sequence[BenchImage],
    methods: Sequence[str] = ("look", "zones", "oklab"),
    grades: Sequence[Grade] = GRADES,
    others: int = 2,
    lot: str = "",
) -> list[Case]:
    """Measure every method on every grade and pair of *images* (one lot)."""
    cases: list[Case] = []
    for source_index, reference_index in pairs(len(images), others):
        for grade in grades:
            for method in methods:
                case = measure(method, grade, images[source_index], images[reference_index], lot)
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
    values: dict[str, dict[str, list[float]]] = {}
    for case in cases:
        if case.same_content is same_content:
            values.setdefault(case.grade, {}).setdefault(case.method, []).append(case.residual)
    return {
        grade: {method: float(np.mean(items)) for method, items in row.items()}
        for grade, row in values.items()
    }


def report(cases: Sequence[Case]) -> str:
    """Markdown report of *cases* (one lot)."""
    lines = [
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


def _categories(folder: Path) -> dict[str, str]:
    """``{file name: category}`` from the folder's ``manifest.json``, if it has one."""
    manifest = folder / "manifest.json"
    if not manifest.exists():
        return {}
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        return {str(e["file"]): str(e.get("category", "")) for e in entries}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def load_folder(folder: Path) -> list[BenchImage]:
    """Every readable picture of *folder* (not recursive), sorted by name."""
    categories = _categories(folder)
    images: list[BenchImage] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            picture = load_image_rgb(path)
        except (OSError, ValueError):
            continue
        images.append(bench_image(path.name, picture, categories.get(path.name, "")))
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
    parser.add_argument("--lot", choices=(*LOTS, "tous"), default="tous",
                        help="measure one lot only (tune on 'reglage', check on 'controle')")
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
    if len(images) < 4:
        print(f"Il faut au moins 4 images lisibles dans {args.folder}.", file=sys.stderr)
        return 2
    print(f"{len(images)} images" + (f", constantes : {overrides}" if overrides else ""))
    print(
        "Résidu = erreur de la méthode / erreur de ne rien faire "
        "(0 : étalonnage retrouvé ; 1 : inutile ; > 1 : pire que rien)."
    )
    cases: list[Case] = []
    for name, members in zip(LOTS, split(images), strict=True):
        if args.lot not in (name, "tous"):
            continue
        with look_constants(**overrides):
            lot_cases = evaluate(members, methods, others=args.others, lot=name)
        cases += lot_cases
        print(f"\n## Lot « {name} » : {len(members)} images, {len(lot_cases)} cas\n")
        print(report(lot_cases))
        sys.stdout.flush()
    if args.json is not None:
        args.json.write_text(
            json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
