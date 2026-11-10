"""Shared helpers of the prototypes: real cases, baselines, metrics, sheets.

Paths point to the user's machine (same convention as docs/audit/diagnostics).
Run from the repository root:

    PYTHONPATH=src .venv/Scripts/python.exe docs/approach-v3/prototypes/<script>.py <out_dir>
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from stealzalut.core.color_transfer import TransferModel, fit_transfer
from stealzalut.core.colorspace import oklab_to_oklch, srgb_to_oklab
from stealzalut.core.finishing import Finishing
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.palette import colour_stats
from stealzalut.core.recipe import apply_global, apply_recipe, full_dose, parse_recipe
from stealzalut.core.zone_transfer import fit_zone_transfer
from stealzalut.utils.image_io import bgr_to_rgb, create_thumbnail, load_image_bgr

U8 = NDArray[np.uint8]
PICS = Path(r"C:\Users\mathi\Pictures")
REPO = Path(__file__).resolve().parents[3]
SOURCE = PICS / "IMG_6295.DNG"
REF_A = PICS / "ben-blennerhassett-CptmwIhxfo4-unsplash.jpg"
REF_B = PICS / "841121141_18631647988026044_4768591964563047386_n.jpg"
AI_ANSWER_A = REPO / "docs" / "audit" / "data" / "ai_answer_tokyo_green_trimmed.txt"
VIEW = 640


def load(path: Path) -> U8:
    return bgr_to_rgb(load_image_bgr(path))


def metrics(img: U8) -> dict[str, float]:
    lab = srgb_to_oklab(img.reshape(-1, 3) / 255.0)
    lch = oklab_to_oklch(lab)
    dark = lab[:, 0] < 0.35
    stats = colour_stats(img)
    return {
        "L": float(lab[:, 0].mean()),
        "C": float(lch[:, 1].mean()),
        "vivid": float(np.mean(lch[:, 1] > 0.06)),
        "coloured": float(np.mean(lch[:, 1] >= 0.05)),
        "shadow_a": float(lab[dark, 1].mean()) if dark.any() else 0.0,
        "shadow_b": float(lab[dark, 2].mean()) if dark.any() else 0.0,
        "shadow_C": float(lch[dark, 1].mean()) if dark.any() else 0.0,
        "dark_share": float(dark.mean()),
        "hue_conc": stats.concentration,
        "L_p05": float(np.quantile(lab[:, 0], 0.05)),
        "L_p95": float(np.quantile(lab[:, 0], 0.95)),
    }


def fmt(name: str, m: dict[str, float]) -> str:
    return (
        f"{name:30s} L={m['L']:.3f} [{m['L_p05']:.2f}-{m['L_p95']:.2f}] C={m['C']:.3f} "
        f"vivid={m['vivid'] * 100:4.1f}% col={m['coloured'] * 100:4.1f}% "
        f"sh(a={m['shadow_a']:+.3f} b={m['shadow_b']:+.3f} C={m['shadow_C']:.3f}) "
        f"R={m['hue_conc']:.2f}"
    )


def render(
    model: TransferModel,
    small: U8,
    params: GradingParams | None = None,
    finishing: Finishing | None = None,
    size: int = 33,
) -> tuple[U8, float, bool]:
    result = build_grid(model, size, params, finishing)
    return apply_lut(small, result.grid, size), result.fold_fraction, result.auto_smoothed


def baseline_zones_ai(source: U8, reference: U8) -> tuple[TransferModel, GradingParams]:
    """Case A baseline: the AI recipe dosed at 100 %, luminance 1.0 (brief §5)."""
    model = fit_zone_transfer(source, reference, 8)
    recipe = parse_recipe(AI_ANSWER_A.read_text(encoding="utf-8"))
    table = apply_recipe(recipe, model.table, model.reference.ids)
    params, _tau = apply_global(recipe, GradingParams(), 1.0)
    table, params = full_dose(table, params)
    return model.with_table(table), replace(params, luminance_strength=1.0)


def baseline_ambiance(source: U8, reference: U8) -> TransferModel:
    return fit_transfer(source, reference, method="ambiance")


def label(img: U8, text: str) -> U8:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (14 + 10 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit_height(img: U8, height: int) -> U8:
    width = round(img.shape[1] * height / img.shape[0])
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def sheet(tiles: list[tuple[str, U8]], columns: int, height: int = 480) -> U8:
    cells = [label(fit_height(im, height), name) for name, im in tiles]
    rows = [cells[i : i + columns] for i in range(0, len(cells), columns)]
    gap = 6
    lines = []
    for row in rows:
        parts: list[U8] = []
        for cell in row:
            parts += [cell, np.full((height, gap, 3), 20, np.uint8)]
        lines.append(np.hstack(parts[:-1]))
    width = max(line.shape[1] for line in lines)
    padded = []
    for line in lines:
        fill = np.full((height, width - line.shape[1], 3), 20, np.uint8)
        padded += [np.hstack([line, fill]), np.full((gap, width, 3), 20, np.uint8)]
    return np.vstack(padded[:-1])


def save(path: Path, img: U8, quality: int = 86) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    path.write_bytes(buf.tobytes())


def small(img: U8) -> U8:
    return create_thumbnail(img, VIEW)
