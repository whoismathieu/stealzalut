"""Prototype: protect the source's vivid colours (neons, lights), let dull areas take the cast."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\mathi\Documents\StealZaLut\src")
import stealzalut.core.grading as grading
import stealzalut.core.zone_transfer as zt
from stealzalut.core.colorspace import oklab_to_oklch, srgb_to_oklab
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.utils.image_io import bgr_to_rgb, load_image_bgr

S = Path(sys.argv[1])
PICS = Path(r"C:\Users\mathi\Pictures")
original_adjust = grading.adjust_displacement
VIVID = {"amount": 0.0}


def adjust_with_vivid(lab_in, lab_target, params):
    delta = original_adjust(lab_in, lab_target, params)
    chroma_in = oklab_to_oklch(lab_in)[:, 1]
    keep = VIVID["amount"] * grading.smoothstep(chroma_in, 0.04, 0.12)
    delta[:, 1:] *= (1.0 - keep)[:, None]
    return delta


grading.adjust_displacement = adjust_with_vivid


def concentration(img):
    lab = srgb_to_oklab(img.reshape(-1, 3) / 255.0)
    lch = oklab_to_oklch(lab)
    col = lch[:, 1] > 0.05
    h, w = np.radians(lch[col, 2]), lch[col, 1]
    r = np.hypot((w * np.cos(h)).sum(), (w * np.sin(h)).sum()) / w.sum()
    return f"C={lch[:, 1].mean():.3f} colores={col.mean()*100:.0f}% concentration R={r:.2f}"


def case(src_name, ref_name, label):
    src = bgr_to_rgb(load_image_bgr(PICS / src_name))
    ref = bgr_to_rgb(load_image_bgr(PICS / ref_name))
    model = zt.fit_zone_transfer(src, ref, 8)
    small = cv2.resize(src, (640, round(640 * src.shape[0] / src.shape[1])), interpolation=cv2.INTER_AREA)
    refs = cv2.resize(ref, (round(ref.shape[1] * small.shape[0] / ref.shape[0]), small.shape[0]), interpolation=cv2.INTER_AREA)
    print(f"--- {label}")
    print("  source    ", concentration(small))
    print("  reference ", concentration(refs))
    tiles = [("REFERENCE", refs), ("SOURCE", small)]
    for amount in (0.0, 0.6, 0.9):
        VIVID["amount"] = amount
        img = apply_lut(small, build_grid(model, 33, GradingParams()).grid, 33)
        print(f"  vives protegees {amount:.0%}:", concentration(img))
        tiles.append((f"COULEURS VIVES PROTEGEES {amount:.0%}", img))
    return tiles


def sheet(tiles, path):
    h = tiles[0][1].shape[0]
    out = []
    for name, im in tiles:
        im = im.copy()
        cv2.rectangle(im, (0, 0), (12 + 11 * len(name), 26), (0, 0, 0), -1)
        cv2.putText(im, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(im)
    gap = np.full((h, 6, 3), 20, np.uint8)
    top = np.hstack([out[0], gap, out[1], gap, out[2]])
    bottom = np.hstack([out[3], gap, out[4]])
    w = max(top.shape[1], bottom.shape[1])
    pad = lambda a: np.hstack([a, np.full((h, w - a.shape[1], 3), 20, np.uint8)])
    img = np.vstack([pad(top), np.full((6, w, 3), 20, np.uint8), pad(bottom)])
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])


sheet(case("IMG_6295.DNG", "841121141_18631647988026044_4768591964563047386_n.jpg", "VIOLET"), S / "vivid_purple.jpg")
sheet(case("IMG_6295.DNG", "ben-blennerhassett-CptmwIhxfo4-unsplash.jpg", "TOKYO VERT"), S / "vivid_green.jpg")
