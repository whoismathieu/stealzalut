"""Reproduce the Tokyo import and measure what weakens the look."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\mathi\Documents\StealZaLut\src")
from stealzalut.core.colorspace import oklab_to_oklch, srgb_to_oklab
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.recipe import apply_global, apply_recipe, parse_recipe
from stealzalut.core.zone_transfer import ZoneBakeParams, fit_zone_transfer
from stealzalut.utils.image_io import bgr_to_rgb, create_thumbnail, load_image_bgr

S = Path(sys.argv[1])
PICS = Path(r"C:\Users\mathi\Pictures")
src = bgr_to_rgb(load_image_bgr(PICS / "IIIIIIIMG_6295 - Copie.DNG"))
ref = bgr_to_rgb(load_image_bgr(PICS / "ben-blennerhassett-CptmwIhxfo4-unsplash.jpg"))
pack = Path(r"C:\Users\mathi\Downloads") / (
    "IIIIIIIMG_6295 - Copie_x_ben-blennerhassett-CptmwIhxfo4-unsplash_stealzalut_pack"
)
analysis = json.loads((pack / "analysis.json").read_text(encoding="utf-8"))
k = analysis["zone_count"]
model = fit_zone_transfer(src, ref, k)
print("zone_count", k, "analysis", model.analysis_id, "pack", analysis["analysis_id"])

recipe = parse_recipe((S / "ai_answer.txt").read_text(encoding="utf-8"))
table = apply_recipe(recipe, model.table, model.reference.ids)
params, tau = apply_global(recipe, GradingParams(), 1.0)
print("recipe params", params, "tau", tau)

small = create_thumbnail(src, 640)


def render(m, p):
    r = build_grid(m, 33, p)
    return apply_lut(small, r.grid, 33), r


def stats(img_rgb, label):
    lab = srgb_to_oklab(img_rgb.reshape(-1, 3).astype(np.float64) / 255.0)
    L = lab[:, 0]
    out = [label]
    for name, lo, hi in (("ombres", 0.0, 0.35), ("medium", 0.35, 0.65), ("hautes", 0.65, 1.01)):
        sel = (L >= lo) & (L < hi)
        a, b = lab[sel, 1].mean(), lab[sel, 2].mean()
        chroma = oklab_to_oklch(lab[sel])[:, 1].mean()
        out.append(f"{name}: a={a:+.3f} b={b:+.3f} C={chroma:.3f} ({sel.mean()*100:.0f}%)")
    print(" | ".join(out))


ref_small = create_thumbnail(ref, 640)
stats(small, "SOURCE   ")
stats(ref_small, "REFERENCE")

variants = {}
recipe_model = model.with_table(table).with_params(ZoneBakeParams(tau=tau))
variants["A recette importee"] = render(recipe_model, params)
variants["B liens IA, globaux neutres"] = render(recipe_model, GradingParams())
full = table
for link in table.links:
    full = full.with_link(replace(link, strength=1.0))
variants["C liens IA force 1, globaux neutres"] = render(
    model.with_table(full).with_params(ZoneBakeParams(tau=1.0)), GradingParams()
)
variants["D auto, globaux neutres"] = render(model, GradingParams())
variants["E transfert global Oklab"] = render(model.global_model, GradingParams())

for name, (img, r) in variants.items():
    stats(img, name[:9].ljust(9))
    print("    ", name, f"folds={r.fold_fraction:.4f} auto_smooth={r.auto_smoothed}")

# Effective chroma gain per link
print("\nDilution par zone (part du deplacement de couleur appliquee):")
for link in table.links:
    eff = (
        (0 if link.locked else link.strength)
        * (1.0 if link.chroma_strength is None else link.chroma_strength)
        * params.chroma_strength
        * params.mix
    )
    print(f"  S{link.source_zone}: strength={link.strength:.2f} -> {eff*100:.0f} % (avant protection neutres {params.preserve_neutrals*100:.0f} %)")

# Contact sheet
h = small.shape[0]
tiles = [("SOURCE", small)] + [(n, im) for n, (im, _) in variants.items()]
ref_fit = cv2.resize(ref_small, (round(ref_small.shape[1] * h / ref_small.shape[0]), h))
rows = []
row = []
for name, im in [("REFERENCE", ref_fit)] + tiles:
    im = im.copy()
    cv2.rectangle(im, (0, 0), (min(im.shape[1], 12 + 11 * len(name)), 26), (0, 0, 0), -1)
    cv2.putText(im, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    row.append(im)
    if len(row) == 3:
        rows.append(row)
        row = []
if row:
    rows.append(row)
width = max(sum(i.shape[1] for i in r) + 6 * (len(r) - 1) for r in rows)
lines = []
for r in rows:
    line = np.full((h, width, 3), 20, np.uint8)
    x = 0
    for im in r:
        line[:, x : x + im.shape[1]] = im
        x += im.shape[1] + 6
    lines.append(line)
sheet = np.vstack([np.vstack([ln, np.full((6, width, 3), 20, np.uint8)]) for ln in lines])
cv2.imwrite(str(S / "tokyo_compare.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("sheet", sheet.shape)
