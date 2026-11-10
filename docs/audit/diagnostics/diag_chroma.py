"""Where is chroma lost? Per zone: target vs pure zone transfer vs blended vs final LUT."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Users\mathi\Documents\StealZaLut\src")
import stealzalut.core.zone_transfer as zt
from stealzalut.core.colorspace import gamut_compress, oklab_to_oklch, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.palette import zone_mask
from stealzalut.core.recipe import apply_global, apply_recipe, full_dose, parse_recipe
from stealzalut.utils.image_io import bgr_to_rgb, create_thumbnail, load_image_bgr

S = Path(sys.argv[1])
PICS = Path(r"C:\Users\mathi\Pictures")
src = bgr_to_rgb(load_image_bgr(PICS / "IMG_6295 - Copie.DNG"))
ref = bgr_to_rgb(load_image_bgr(PICS / "ben-blennerhassett-CptmwIhxfo4-unsplash.jpg"))
model = zt.fit_zone_transfer(src, ref, 8)
recipe = parse_recipe((S / "ai_answer.txt").read_text(encoding="utf-8"))
table = apply_recipe(recipe, model.table, model.reference.ids)
params, _ = apply_global(recipe, GradingParams(), 1.0)
ftable, fparams = full_dose(table, params)
fparams = replace(fparams, luminance_strength=1.0)
m = model.with_table(ftable).with_params(zt.ZoneBakeParams(tau=1.0))

small = create_thumbnail(src, 256)
h, w = small.shape[:2]
lab = srgb_to_oklab(small.reshape(-1, 3) / 255.0)
global_dl = srgb_to_oklab(m.global_model.apply(small.reshape(-1, 3) / 255.0)) - lab

blended = lab + m.displacement(lab)
blended_srgb = oklab_to_srgb(gamut_compress(blended))
lut = apply_lut(small, build_grid(m, 33, fparams).grid, 33).reshape(-1, 3) / 255.0


def chroma(x):
    return oklab_to_oklch(x)[:, 1].mean()


print("zone | aire | cible C | pur C | melange C | apres gamut C | LUT final C")
tot = {"target": 0.0, "pure": 0.0, "blend": 0.0, "gamut": 0.0, "lut": 0.0, "area": 0.0}
for zone in m.source.zones:
    link = m.table.link(zone.id)
    mask = zone_mask(m.source, zone.id, (w, h)).ravel()
    if not mask.any():
        continue
    if link.locked or not link.targets:
        continue
    t_mean, _ = zt._target_moments(m.reference, link.targets)
    target_c = float(np.hypot(*t_mean[1:]))
    pure = lab[mask] + zt.zone_delta(lab[mask], zone, link, m.reference, global_dl[mask])
    row = (
        target_c,
        chroma(pure),
        chroma(blended[mask]),
        chroma(srgb_to_oklab(blended_srgb[mask])),
        chroma(srgb_to_oklab(lut[mask])),
    )
    a = zone.area
    for k, v in zip(("target", "pure", "blend", "gamut", "lut"), row, strict=True):
        tot[k] += a * v
    tot["area"] += a
    print(f"S{zone.id}  | {a*100:4.0f}% | " + " | ".join(f"{v:.3f}" for v in row))
print("moy  |      | " + " | ".join(f"{tot[k]/tot['area']:.3f}" for k in ("target", "pure", "blend", "gamut", "lut")))

# Membership: how much of each pixel's displacement comes from its own zone?
zones = m.source.zones
logits = np.stack([np.full(len(lab), m.params.background_logit)] + [m._zone_logit(lab, z) for z in zones], 1)
logits -= logits.max(1, keepdims=True)
wts = np.exp(logits)
wts /= wts.sum(1, keepdims=True)
own = []
for i, zone in enumerate(zones, start=1):
    mask = zone_mask(m.source, zone.id, (w, h)).ravel()
    if mask.any():
        own.append(f"S{zone.id}:{wts[mask, i].mean():.2f}")
print("poids de sa propre zone (tau 1):", " ".join(own), "| global:", f"{wts[:, 0].mean():.2f}")
